"""Pinned-pelvis G1 Dex3 task. No kinematic robot teleportation or hidden-grid planner."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .sonic import BODY_JOINTS, DEFAULT, KP, KD, EFFORT, ARMATURE, FrozenSonic

GRASP_OFFSET = np.array([.235, 0., 0.])

def make_scene_spec(asset_xml, object_halfsize):
    asset_xml = Path(asset_xml)
    root = ET.parse(asset_xml).getroot()
    root.find("compiler").set("meshdir", str(asset_xml.parent / "assets"))
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast")
    ET.SubElement(root, "visual")
    world = root.find("worldbody")
    for g in list(world.findall("geom")):
        if g.get("type") == "plane":
            world.remove(g)
    pelvis = world.find("body[@name='pelvis']")
    for j in list(pelvis):
        if j.tag == "freejoint" or (j.tag == "joint" and j.get("type") == "free"):
            pelvis.remove(j)
    pelvis.set("pos", "0 0 0.76")
    # Asset has duplicate mesh geoms for visual/collision. Group 1 is visual-only.
    for geom in root.findall(".//geom"):
        if geom.get("group") == "1":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        else:
            geom.set("group", "0")
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
    torso = root.find(".//body[@name='torso_link']")
    # Camera axes: x=right (-world-y), y=up, looking along +x with downward tilt.
    a = np.deg2rad(45)
    xyaxes = [0, -1, 0, np.sin(a), 0, np.cos(a)]
    ET.SubElement(torso, "camera", name="ego", pos="0.08 0 0.39", xyaxes=" ".join(map(str, xyaxes)), fovy="64")
    palm = root.find(".//body[@name='left_hand_palm_link']")
    # Idealized fixed grasp beyond the open Dex3 fingertips; avoid an
    # interpenetrating weld that would inject impulses at release.
    ET.SubElement(palm, "site", name="held_center", pos=" ".join(map(str, GRASP_OFFSET)), size="0.005", rgba="0 0 0 0")
    ET.SubElement(world, "geom", name="task_floor", type="plane", size="3 3 .1", rgba=".18 .2 .22 1", friction=".8 .01 .001")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse=".8 .8 .8")
    # Models are compiled once; scene bodies move only between episodes.
    for i in range(2):
        body = ET.SubElement(world, "body", name=f"table_{i}", pos=f".48 {(-1) ** i * .26} .65")
        ET.SubElement(body, "geom", name=f"support_{i}", type="box", size=".12 .13 .025", rgba=".5 .37 .22 1", friction="1 .01 .001")
        block = ET.SubElement(world, "body", name=f"block_{i}", pos=f".5 {(-1) ** i * .26} .73")
        ET.SubElement(block, "geom", name=f"obstacle_{i}", type="box", size=".065 .08 .06", rgba=".7 .2 .15 1")
    occluder = ET.SubElement(world, "body", name="occluder", pos=".33 .18 .85")
    ET.SubElement(occluder, "geom", name="occluder_geom", type="box", size=".015 .07 .17", rgba=".25 .3 .55 1")
    obj = ET.SubElement(world, "body", name="payload", pos=".3 .2 1.0")
    ET.SubElement(obj, "freejoint", name="payload_free")
    ET.SubElement(obj, "geom", name="payload_geom", type="box", size=" ".join(map(str, object_halfsize)),
                  mass=".15", rgba=".15 .65 .3 1", friction="1 .01 .001")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="hold", body1="left_hand_palm_link", body2="payload",
                  relpose=" ".join(map(str, GRASP_OFFSET)) + " 1 0 0 0", solref=".005 1")
    actuator = ET.SubElement(root, "actuator")
    for j in root.findall(".//joint"):
        name = j.get("name")
        if name not in BODY_JOINTS and "hand_" not in (name or ""):
            continue
        if name in BODY_JOINTS:
            i = BODY_JOINTS.index(name)
            kp, kd, effort, armature = KP[i], KD[i], EFFORT[i], ARMATURE[i]
        else:
            kp, kd, effort, armature = 30., 1., 1.4, .01
        j.set("damping", "0")
        j.set("armature", str(armature))
        ET.SubElement(actuator, "general", name=f"motor_{name}", joint=name, biastype="affine",
                      gainprm=f"{kp} 0 0", biasprm=f"0 {-kp} {-kd}", forcelimited="true", forcerange=f"{-effort} {effort}")
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def make_scene(asset_xml, object_halfsize):
    return make_scene_spec(asset_xml, object_halfsize).compile()


class Physics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.halfsize = np.array([.04, .04, .035])
        self.scene = None
        if cfg.simulator == "mjlab":
            from mjlab.scene import Scene, SceneCfg
            source = make_scene_spec(cfg.asset_xml, self.halfsize)
            def attach_task(spec):
                spec.option.timestep = .005
                spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                spec.attach(source, prefix="", frame=spec.worldbody.add_frame())
            # The task is one composite spec; preserve SONIC's named joints.
            self.scene = Scene(SceneCfg(num_envs=1, spec_fn=attach_task), device=cfg.device)
            self.spec = self.scene.spec
            self.model = self.scene.compile()
        else:
            self.spec = make_scene_spec(cfg.asset_xml, self.halfsize)
            self.model = self.spec.compile()
        self.runtime = None
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in BODY_JOINTS])
        self.dids = np.array([self.model.joint(n).dofadr[0] for n in BODY_JOINTS])
        self.aids = np.array([self.model.actuator("motor_" + n).id for n in BODY_JOINTS])
        self.jids = np.array([self.model.joint(n).id for n in BODY_JOINTS])
        self.pelvis = self.model.body("pelvis").id
        self.site = self.model.site("held_center").id
        self.payload = self.model.body("payload").id
        self.obj_q = self.model.joint("payload_free").qposadr[0]
        self.hold = self.model.equality("hold").id
        self.renderer = None
        self.sonic = FrozenSonic(cfg.sonic_dir) if cfg.backend == "sonic" else None
        self.reference = DEFAULT.copy()
        self.last_tracking_error = 0.
        self.last_stability = {}
        self.failure = ""
        self.low_steps = 0
        robot_bodies = {self.pelvis}
        for b in range(self.model.nbody):
            if int(self.model.body_parentid[b]) in robot_bodies:
                robot_bodies.add(b)
        self.robot_bodies = robot_bodies
        self.scene_bodies = {self.model.body(n).id for n in ("table_0", "table_1", "block_0", "block_1", "occluder")}
        self.collision_ticks = 0

    def reset(self, rng, suite):
        mujoco.mj_resetData(self.model, self.data)
        self.reference = DEFAULT.copy()
        self.reference[15] = -.25
        self.reference[18] = 1.3
        self.data.qpos[self.qids] = self.reference
        self.data.ctrl[self.aids] = self.reference
        self.halfsize = np.array([rng.uniform(.025, .055), rng.uniform(.025, .05), rng.uniform(.025, .05)])
        if self.cfg.object_variant != "random":
            # Random draws above are retained, so changing only object_variant
            # preserves the layout for matched seed comparisons.
            # Grow across the palm axis so every preset remains beyond the
            # open fingertips at the fixed grasp offset (min x >= .18 m).
            self.halfsize = np.array({"small": [.025, .025, .03], "large": [.055, .075, .04],
                                      "long": [.025, .09, .035]}[self.cfg.object_variant])
        self.spec.geom("payload_geom").size = self.halfsize
        for i in range(2):
            table = self.model.body(f"table_{i}").id
            self.model.body_pos[table] = [rng.uniform(.30, .38), (-1) ** i * rng.uniform(.24, .32), rng.uniform(.76, .82)]
            block = self.model.body(f"block_{i}").id
            self.model.body_pos[block] = self.model.body_pos[table] + [rng.uniform(-.06, .06), rng.uniform(-.08, .08), .085]
            if suite in {"E0", "E1"}:
                self.model.body_pos[block, 0] = 4.
        self.model.body_pos[self.model.body("occluder").id] = [4, 0, 0] if suite in {"E0", "E1"} else [.28, rng.uniform(.12, .22), .91]
        if suite == "E1":
            self.model.body_pos[self.model.body("table_1").id, 0] = 4.
        for name in ("table_0", "table_1", "block_0", "block_1", "occluder"):
            self.spec.body(name).pos = self.model.body_pos[self.model.body(name).id].copy()
        # MuJoCo 3.10 setConst does not rebuild size-dependent collision bounds.
        # Compile the randomized spec so bounds, inertias and static BVH agree;
        # the next runtime uploads exactly this model, with unchanged topology.
        self.runtime = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.model = self.scene.compile() if self.scene else self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.data.qpos[self.qids] = self.reference
        self.data.ctrl[self.aids] = self.reference
        # Initialize a held object above the support surfaces. This defines the
        # episode's initial grasp, not an action-time kinematic state change.
        initial, _ = self.ik([.30, .29, 1.02], np.eye(3))
        self.reference = initial
        self.data.qpos[self.qids] = self.reference
        self.data.ctrl[self.aids] = self.reference
        if suite == "E1":
            self.reference[12] = -.8
            self.data.qpos[self.qids] = self.reference
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[self.obj_q:self.obj_q+3] = self.data.site_xpos[self.site]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        self.data.qpos[self.obj_q+3:self.obj_q+7] = quat
        self.data.eq_active[self.hold] = True
        mujoco.mj_forward(self.model, self.data)
        if self.sonic:
            self.sonic.reset()
        if self.cfg.simulator == "mjlab":
            from .mjlab_runtime import MjlabRuntime
            # Reset-time geometry changes require a fresh device model; never
            # leave stale collision bounds in a captured CUDA graph.
            self.runtime = MjlabRuntime(self.cfg, self.model, self.data, self.scene)
        self.failure, self.low_steps, self.collision_ticks = "", 0, 0
        self.last_stability = {}
        self.run(self.reference, .4)

    def run(self, target, seconds):
        """Smooth references, SONIC at 50 Hz and physics at 200 Hz."""
        start = self.reference.copy()
        n = max(1, int(round(seconds / .02)))
        for k in range(n):
            f = min(1., (k + 1) / n)
            alpha = f*f*(3 - 2*f)
            refs = np.stack([start + (target-start) * (lambda t: t*t*(3-2*t))(min(1., (k+1+j*5)/n)) for j in range(10)])
            velocities = np.gradient(refs, .1, axis=0)
            if self.sonic:
                control = self.sonic.command(self.data.qpos[self.qids], self.data.qvel[self.dids],
                                             np.zeros(3), self.data.xmat[self.pelvis].reshape(3, 3), refs, velocities)
            else:
                control = start + (target-start) * alpha
            limits = self.model.jnt_range[self.jids]
            self.data.ctrl[self.aids] = np.clip(control, limits[:, 0], limits[:, 1])
            if self.runtime is not None:
                self.runtime.advance(self.data, 4)
            else:
                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)
            # Reward/diagnostic event only; never used by candidate generation.
            # Foot-ground and payload-support contacts are expected contacts.
            for c in self.data.contact:
                b1, b2 = (int(self.model.geom_bodyid[g]) for g in c.geom)
                if c.dist < -.001 and ((b1 in self.robot_bodies and b2 in self.scene_bodies)
                                      or (b2 in self.robot_bodies and b1 in self.scene_bodies)):
                    self.collision_ticks += 1
                    break
            self.low_steps += 1
            if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel)) > 250:
                self.failure = "unstable_physics"
                raise FloatingPointError(self.failure)
        self.reference = target.copy()
        self.last_tracking_error = float(np.sqrt(np.mean((self.data.qpos[self.qids] - target)**2)))

    def observe(self, yaw, pitch):
        target = self.reference.copy()
        target[12], target[14] = yaw, pitch
        self.run(target, self.cfg.observe_seconds)

    def ik(self, position, rotation=None):
        """Damped least squares, left arm only, no privileged collision geometry."""
        d = self.ikdata
        d.qpos[:] = self.data.qpos
        columns = self.dids[15:22]
        qids = self.qids[15:22]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        for _ in range(100):
            mujoco.mj_forward(self.model, d)
            dp = np.asarray(position) - d.site_xpos[self.site]
            if rotation is not None:
                r = d.site_xmat[self.site].reshape(3, 3)
                dr = .5 * sum(np.cross(r[:, i], rotation[:, i]) for i in range(3))
                error = np.r_[dp, .25 * dr]
            else:
                error = dp
            if np.linalg.norm(error) < .008:
                break
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, columns], .25*jacr[:, columns]]) if rotation is not None else jacp[:, columns]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .003*np.eye(len(error)), error)
            d.qpos[qids] += np.clip(delta, -.12, .12)
            bounds = self.model.jnt_range[self.jids[15:22]]
            d.qpos[qids] = np.clip(d.qpos[qids], bounds[:, 0], bounds[:, 1])
        mujoco.mj_forward(self.model, d)
        result = self.data.qpos[self.qids].copy()
        result[15:22] = d.qpos[qids]
        error = float(np.linalg.norm(np.asarray(position) - d.site_xpos[self.site]))
        return result, error

    def place(self, center):
        """Actual contact release; pre-release failures permit a later retry."""
        center = np.asarray(center)
        pre = center + [0, 0, self.halfsize[2] + .13]
        target = center + [0, 0, self.halfsize[2] + .012]
        for stage, waypoint in enumerate((pre, target)):
            offset = np.zeros(3)
            for attempt in range(3):
                pose, error = self.ik(waypoint + offset, np.eye(3))
                if error > .045:
                    return False, "ik_unreachable", False
                self.run(pose, self.cfg.move_seconds)
                residual = waypoint - self.data.site_xpos[self.site]
                if np.linalg.norm(residual) < (.06 if stage == 0 else .045):
                    break
                # Fixed, bounded Cartesian feedback in the skill adapter. It
                # changes SONIC references, never joint controls or model weights.
                offset = np.clip(offset + .25*residual, -.06, .06)
            if np.linalg.norm(waypoint - self.data.site_xpos[self.site]) > .07:
                return False, "tracking_error", False
        self.data.eq_active[self.hold] = False
        # Settle before retracting, then require stable physical support.
        self.run(self.reference, .4)
        withdraw = self.data.site_xpos[self.site] + [0, 0, .18]
        pose, _ = self.ik(withdraw)
        self.run(pose, self.cfg.move_seconds)
        success = False
        for window in range(3):
            before = self.data.xpos[self.payload].copy()
            before_quat = self.data.xquat[self.payload].copy()
            max_displacement, max_angle = 0., 0.
            supported_samples, hand_samples = 0, 0
            for _ in range(8):
                self.run(self.reference, .1)
                max_displacement = max(max_displacement, float(np.linalg.norm(self.data.xpos[self.payload] - before)))
                dot = abs(float(np.dot(before_quat, self.data.xquat[self.payload])))
                max_angle = max(max_angle, float(2*np.arccos(np.clip(dot, 0., 1.))))
                support, hand_contact = False, False
                for contact in self.data.contact:
                    names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in contact.geom]
                    if "payload_geom" in names:
                        support |= any(n.startswith("support_") for n in names)
                        other = int(contact.geom[1] if contact.geom[0] == self.model.geom("payload_geom").id else contact.geom[0])
                        hand_contact |= int(self.model.geom_bodyid[other]) in self.robot_bodies
                supported_samples += int(support)
                hand_samples += int(hand_contact)
            freevel = self.model.joint("payload_free").dofadr[0]
            # Allow bounded settling after retract, then require a full 0.8 s
            # window of supported, hand-free, bounded pose. Actual waiting time
            # remains part of the macro action duration and reward.
            stable = max_displacement < .01 and max_angle < .035
            success = bool(stable and supported_samples == 8 and hand_samples == 0)
            self.last_stability = {"max_displacement": max_displacement, "max_angle_radians": max_angle,
                                   "supported_samples": supported_samples, "hand_samples": hand_samples,
                                   "windows_checked": window+1,
                                   "instant_linear_speed": float(np.linalg.norm(self.data.qvel[freevel:freevel+3]))}
            if success:
                break
        return success, "success" if success else "unstable_or_unsupported", True

    def capture(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.cfg.image_height, width=self.cfg.image_width)
        self.renderer.update_scene(self.data, camera="ego")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        camera = self.model.camera("ego").id
        return rgb, depth, self.data.cam_xpos[camera].copy(), self.data.cam_xmat[camera].reshape(3, 3).copy(), float(self.model.cam_fovy[camera])

    def close(self):
        if self.renderer:
            self.renderer.close()
        self.runtime = None
