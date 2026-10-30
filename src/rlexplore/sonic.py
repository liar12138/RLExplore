"""Default-release SONIC ONNX interface; no model optimization or fallback.

Ordering, centered positions, raw velocities and 6D rotations follow NVIDIA's
g1_deploy_onnx_ref.cpp and policy_parameters.hpp. Only the g1 reference mode
and the exact released observation schema below are supported.
"""
from collections import deque
from pathlib import Path
import hashlib
import re
import numpy as np
import onnxruntime as ort
import yaml

BODY_JOINTS = (
    [f"{side}_{name}_joint" for side in ("left", "right") for name in
     ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{side}_{name}_joint" for side in ("left", "right") for name in
       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
# Policy/IsaacLab index -> MuJoCo kinematic-tree index.
POLICY_ORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                         4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
DEFAULT = np.array([-.312, 0, 0, .669, -.363, 0] * 2 + [0, 0, 0]
                   + [.2, .2, 0, .6, 0, 0, 0] + [.2, -.2, 0, .6, 0, 0, 0])
ARMATURE = np.array([.025101925, .025101925, .010177520, .025101925,
                     .003609725, .003609725] * 2 + [.010177520, .003609725, .003609725]
                    + [.003609725] * 5 + [.00425] * 2 + [.003609725] * 5 + [.00425] * 2)
KP = ARMATURE * (20 * np.pi) ** 2
KD = 4 * ARMATURE * (20 * np.pi)
for _i in (4, 5, 10, 11, 13, 14):
    KP[_i] *= 2
    KD[_i] *= 2
EFFORT = np.array([139, 139, 88, 139, 50, 50] * 2 + [88, 50, 50]
                  + [25] * 5 + [5] * 2 + [25] * 5 + [5] * 2)
# Official ankle/waist scales use single-motor stiffness, unlike PD gains.
SCALE = .25 * np.array([139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25]
                       + [25] * 5 + [5] * 2 + [25] * 5 + [5] * 2) / (ARMATURE * (20 * np.pi) ** 2)
ENCODER_DIMS = [4, 290, 290, 10, 1, 6, 60, 120, 120, 9, 12, 720, 60, 60]
ENCODER_NAMES = ["encoder_mode_4", "motion_joint_positions_10frame_step5",
                 "motion_joint_velocities_10frame_step5", "motion_root_z_position_10frame_step5",
                 "motion_root_z_position", "motion_anchor_orientation", "motion_anchor_orientation_10frame_step5",
                 "motion_joint_positions_lowerbody_10frame_step5", "motion_joint_velocities_lowerbody_10frame_step5",
                 "vr_3point_local_target", "vr_3point_local_orn_target", "smpl_joints_10frame_step1",
                 "smpl_anchor_orientation_10frame_step1", "motion_joint_positions_wrists_10frame_step1"]
DECODER_NAMES = ["token_state", "his_base_angular_velocity_10frame_step1",
                 "his_body_joint_positions_10frame_step1", "his_body_joint_velocities_10frame_step1",
                 "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"]


class FrozenSonic:
    def __init__(self, directory):
        directory = Path(directory)
        config = yaml.safe_load((directory / "observation_config.yaml").read_text())
        active = lambda items: [i["name"] for i in items if i.get("enabled", True)]
        if active(config["observations"]) != DECODER_NAMES or active(config["encoder"]["encoder_observations"]) != ENCODER_NAMES:
            raise ValueError("SONIC observation schema differs from the supported default release")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(directory / "model_encoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(directory / "model_decoder.onnx"), options, providers=["CPUExecutionProvider"])
        if self.encoder.get_inputs()[0].shape != [1, 1762] or self.decoder.get_inputs()[0].shape != [1, 994]:
            raise ValueError("checkpoint dimensions do not match the default SONIC release")
        self.hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")}
        self.history = deque(maxlen=10)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.calls = 0

    def reset(self):
        self.history.clear()
        self.last_action.fill(0)

    def command(self, q, dq, angular_velocity, base_rotation, reference, reference_velocity=None):
        """q/dq/reference in BODY_JOINTS order; references cover 10 future frames."""
        q, dq = np.asarray(q), np.asarray(dq)
        reference = np.broadcast_to(reference, (10, 29))
        velocity = np.zeros_like(reference) if reference_velocity is None else np.broadcast_to(reference_velocity, (10, 29))
        orientation = (base_rotation.T @ np.eye(3))[:, :2].reshape(-1)
        arrays = [np.zeros(n, np.float32) for n in ENCODER_DIMS]
        # Official GatherEncoderMode writes a scalar mode ID followed by zeros,
        # not a one-hot vector. g1=0, teleop=1, smpl=2.
        arrays[0][0] = 0
        arrays[1][:] = reference[:, POLICY_ORDER].reshape(-1)
        arrays[2][:] = velocity[:, POLICY_ORDER].reshape(-1)
        arrays[5][:] = orientation
        arrays[6][:] = np.tile(orientation, 10)
        token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: np.concatenate(arrays)[None]})[0].reshape(-1)
        if token.shape != (64,):
            raise ValueError(f"invalid motion token: {token.shape}")
        entry = (np.asarray(angular_velocity), (q - DEFAULT)[POLICY_ORDER], dq[POLICY_ORDER],
                 self.last_action.copy(), base_rotation.T @ np.array([0., 0., -1.]))
        if not self.history:
            self.history.extend([entry] * 9)
        self.history.append(entry)
        observation = np.concatenate([token] + [np.concatenate([h[i] for h in self.history]) for i in range(5)]).astype(np.float32)
        action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: observation[None]})[0].reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise FloatingPointError("SONIC returned invalid actions")
        self.last_action[:] = action
        self.calls += 1
        mujoco_action = np.empty(29)
        mujoco_action[POLICY_ORDER] = action
        return DEFAULT + SCALE * mujoco_action
