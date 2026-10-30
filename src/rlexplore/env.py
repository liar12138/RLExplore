import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .memory import EvidenceMap
from .physics import Physics, GRASP_OFFSET

VIEWS = [(y, p) for p in (0., .18, .35) for y in (-.65, 0., .65)]
STATE_DIM = 94
PRIV_DIM = 20


class PlacementEnv(gym.Env):
    """Level A physical placement. All action candidates are observation-derived."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.physics = Physics(cfg)
        self.memory = EvidenceMap()
        self.n_actions = len(VIEWS) + cfg.max_candidates + 1
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, (4, cfg.image_height, cfg.image_width), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            "candidates": spaces.Box(-np.inf, np.inf, (cfg.max_candidates, 14), np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
        })
        self.episode_index = 0
        self.needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[self.cfg.split]
            seed = self.cfg.seed + offset + self.episode_index
        self.episode_seed = seed
        self.episode_index += 1
        self.rng = np.random.default_rng(seed)
        self.physics.reset(self.rng, self.cfg.suite)
        self.memory.reset()
        self.steps, self.elapsed, self.observations, self.attempts = 0, 0., 0, 0
        self.feedback = np.zeros(4)
        self.last_skill = np.zeros(len(VIEWS)+2)
        self.last_duration = 0.
        self.success = False
        self.needs_reset = False
        return self._observation(), self._info("reset")

    def _observation(self):
        rgb, depth, position, rotation, fovy = self.physics.capture()
        if self.cfg.suite == "E9":
            depth = np.maximum(.01, depth + self.rng.normal(0, .008, depth.shape))
            position += self.rng.normal(0, .005, 3)
        if not self.cfg.use_map:
            self.memory.reset()
        self.memory.integrate(depth, position, rotation, fovy)
        halfsize = self.physics.halfsize if self.cfg.use_object else np.array([.04, .04, .035])
        self.candidate_features, self.candidate_mask = self.memory.candidates(halfsize, self.cfg.max_candidates)
        mask = np.r_[np.ones(len(VIEWS), bool), self.candidate_mask, True]
        state = np.r_[self.physics.data.qpos[self.physics.qids], self.physics.data.qvel[self.physics.dids],
                      halfsize*2 if self.cfg.use_object else np.zeros(3), GRASP_OFFSET if self.cfg.use_object else [0, 0, 0],
                      position, rotation[:, :2].reshape(-1), self.elapsed/self.cfg.time_budget,
                      self.feedback if self.cfg.use_feedback else np.zeros(4), self.memory.summary(),
                      self.last_skill, self.last_duration/self.cfg.time_budget]
        return {"image": np.concatenate([rgb.transpose(2, 0, 1)/255., np.clip(depth[None], 0, 3)/3]).astype(np.float32),
                "state": state.astype(np.float32), "candidates": self.candidate_features.copy(), "action_mask": mask}

    def privileged_state(self):
        """Training critic only. This is deliberately outside the Gym observation."""
        p = self.physics
        return np.r_[p.model.body_pos[p.model.body("table_0").id], p.model.body_pos[p.model.body("table_1").id],
                     p.model.body_pos[p.model.body("block_0").id], p.model.body_pos[p.model.body("block_1").id],
                     p.model.body_pos[p.model.body("occluder").id], p.data.xpos[p.payload],
                     p.data.eq_active[p.hold], self.elapsed/self.cfg.time_budget].astype(np.float32)

    def _info(self, reason):
        return {"success": bool(self.success), "reason": reason, "elapsed": self.elapsed,
                "observations": self.observations, "attempts": self.attempts, "seed": self.episode_seed,
                "suite": self.cfg.suite, "level": "A_pinned_pelvis", "backend": self.cfg.backend,
                "simulator": self.cfg.simulator,
                "tracking_rmse": self.physics.last_tracking_error, "low_level_steps": self.physics.low_steps,
                "stability": dict(self.physics.last_stability),
                "collision_seconds": self.physics.collision_ticks*.02}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("reset is required after episode termination")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action index: {action}")
        action = int(action)
        before = self.physics.data.time
        collisions_before = self.physics.collision_ticks
        terminated, penalty, reason = False, 0., "observed"
        self.feedback.fill(0)
        if action < len(VIEWS):
            self.physics.observe(*VIEWS[action])
            self.observations += 1
            self.feedback[0] = 1
        elif action == self.n_actions-1:
            terminated, reason = True, "abort"
            penalty = 1.
        else:
            index = action-len(VIEWS)
            self.attempts += 1
            if not self.candidate_mask[index]:
                penalty, reason = self.cfg.invalid_cost, "unknown_candidate"
                self.feedback[1] = 1
            elif self.cfg.suite == "E7" and self.rng.random() < .2:
                self.physics.run(self.physics.reference, .4)
                penalty, reason = self.cfg.invalid_cost, "injected_skill_timeout"
                self.feedback[2] = 1
            else:
                self.success, reason, released = self.physics.place(self.candidate_features[index, :3])
                terminated = self.success or released
                if not self.success:
                    penalty = self.cfg.invalid_cost
                    self.feedback[1] = 1
        self.steps += 1
        dt = max(.02, self.physics.data.time-before)
        self.last_duration = dt
        self.last_skill.fill(0)
        self.last_skill[action if action < len(VIEWS) else len(VIEWS)+(action == self.n_actions-1)] = 1
        self.feedback[3] = min(1., self.physics.last_tracking_error)
        self.elapsed += dt
        truncated = not terminated and (self.steps >= self.cfg.max_decisions or self.elapsed >= self.cfg.time_budget)
        collision_time = .02*(self.physics.collision_ticks-collisions_before)
        reward = self.cfg.success_reward*float(self.success) - self.cfg.time_cost*dt - penalty - self.cfg.collision_cost*collision_time
        observation = self._observation()
        self.needs_reset = terminated or truncated
        info = self._info("budget" if truncated else reason)
        info["duration"] = dt
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.physics.close()
