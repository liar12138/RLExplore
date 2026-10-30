"""Actual mjlab/MuJoCo-Warp integration; one world for initial bring-up.

Host MjData is a readback for ONNX, IK, contact reports and EGL RGB-D rendering.
It never advances physics when this backend is selected. Every physics step is
Simulation.step() on the selected CUDA device.
"""
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjwarp
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg


class MjlabRuntime:
    def __init__(self, cfg, model, host, scene):
        if not torch.cuda.is_available():
            raise RuntimeError("mjlab requires working CUDA; no CPU physics fallback")
        self.device = "cuda:0" if cfg.device == "cuda" else cfg.device
        self.scene = scene
        self.steps = 0
        self.sim = Simulation(
            num_envs=1, model=model, device=self.device,
            cfg=SimulationCfg(nconmax=512, njmax=1024,
                              mujoco=MujocoCfg(timestep=.005, integrator="implicitfast",
                                               iterations=50, ls_iterations=20)),
        )
        self.scene.initialize(model, self.sim.wp_model, self.sim.wp_data)
        self.sim.reset()
        for name in ("qpos", "qvel", "ctrl", "eq_active"):
            field = getattr(self.sim.data, name)
            field[0] = torch.as_tensor(np.array(getattr(host, name), copy=True), device=self.device, dtype=field.dtype)
        self.sim.forward()
        self.pull(host)

    def advance(self, host, substeps):
        for name in ("ctrl", "eq_active"):
            field = getattr(self.sim.data, name)
            field[0] = torch.as_tensor(np.array(getattr(host, name), copy=True), device=self.device, dtype=field.dtype)
        self.scene.write_data_to_sim()
        for _ in range(substeps):
            self.sim.step()
            self.scene.update(dt=.005)
            self.steps += 1
        self.pull(host)

    def pull(self, host):
        # Includes GPU contact pairs and constraint forces, not CPU recomputed
        # collision labels. The single-world readback prioritizes verification.
        wp.synchronize_device(self.device)
        mjwarp.get_data_into(host, self.sim.mj_model, self.sim.wp_data, world_id=0)
