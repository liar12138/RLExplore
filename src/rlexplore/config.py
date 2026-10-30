from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    asset_xml: str = "/home/tcs/8t01/lhx/weightlifterc2p/src/assets/robots/unitree_g1_dex3/xmls/g1_dex3.xml"
    sonic_dir: str = "/home/tcs/8t01/tactile-wbc/thirdparty/sonic/gear_sonic_deploy/policy/release"
    backend: str = "sonic"
    simulator: str = "mjlab"
    suite: str = "E2"
    split: str = "train"
    seed: int = 0
    image_width: int = 96
    image_height: int = 72
    max_decisions: int = 24
    time_budget: float = 60.0
    observe_seconds: float = 0.6
    move_seconds: float = 1.2
    max_candidates: int = 16
    history_length: int = 16
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    backbone: str = "transformer"
    use_map: bool = True
    use_object: bool = True
    use_feedback: bool = True
    object_variant: str = "random"
    privileged_critic: bool = True
    device: str = "cuda:0"
    updates: int = 100
    rollout_steps: int = 64
    ppo_epochs: int = 4
    minibatch: int = 16
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    time_cost: float = 0.02
    invalid_cost: float = 0.2
    collision_cost: float = 0.1
    success_reward: float = 10.0
    run_dir: str = "runs/default"

    def validate(self):
        if self.simulator not in {"mjlab", "mujoco_debug"}:
            raise ValueError("simulator must be mjlab or explicit mujoco_debug")
        if self.simulator == "mjlab" and not self.device.startswith("cuda"):
            raise ValueError("mjlab physics requires device=cuda:0; CPU reference is simulator=mujoco_debug")
        if self.backend not in {"sonic", "pd_debug"}:
            raise ValueError("backend must be sonic or explicit pd_debug")
        if self.suite not in {"E0", "E1", "E2", "E3", "E7", "E9"}:
            raise ValueError("implemented suites: E0/E1/E2/E3/E7/E9; paired E4-E6 and E8 are not implemented")
        if self.object_variant not in {"random", "small", "large", "long"}:
            raise ValueError("unknown object variant")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("unknown layout split")
        if self.backbone not in {"transformer", "gru", "reactive", "ttt"}:
            raise ValueError("unknown backbone")
        if self.heads < 1 or self.hidden_dim < 2 or self.hidden_dim % 2 or self.hidden_dim % self.heads or self.history_length < 1:
            raise ValueError("invalid model dimensions")
        if min(self.layers, self.updates, self.ppo_epochs, self.max_candidates, self.max_decisions, self.rollout_steps, self.minibatch) < 1:
            raise ValueError("counts must be positive")
        if min(self.image_width, self.image_height) < 24 or min(self.time_budget, self.observe_seconds, self.move_seconds, self.learning_rate) <= 0:
            raise ValueError("invalid image dimensions or timing/learning rate")
        if not (0 < self.gamma <= 1 and 0 < self.gae_lambda <= 1 and 0 < self.clip < 1):
            raise ValueError("invalid PPO discount or clip")
        return self


def load_config(path, overrides=()):
    data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in Config.__dataclass_fields__:
            raise ValueError(f"unknown setting: {key}")
        data[key] = yaml.safe_load(value)
    return Config(**data).validate()


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
