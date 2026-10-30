"""E0 execution diagnostics with privileged target coordinates, never training data.

Run via scripts/python.sh scripts/calibrate.py --config configs/smoke.yaml.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rlexplore.config import load_config
from rlexplore.physics import Physics
from rlexplore.sonic import BODY_JOINTS

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/smoke.yaml")
parser.add_argument("--set", nargs="*", default=[])
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--output", default="runs/e0_calibration.json")
args = parser.parse_args()
cfg = load_config(args.config, args.set)
physics = Physics(cfg)
results = []
try:
    for seed in range(args.episodes):
        physics.reset(np.random.default_rng(seed), "E0")
        physics.run(physics.reference, 1.)
        error = physics.data.qpos[physics.qids] - physics.reference
        table = physics.model.body_pos[physics.model.body("table_0").id].copy()
        target = table + [0, 0, .025]
        success, reason, released = physics.place(target)
        results.append({"seed": seed, "success": success, "reason": reason, "released": released,
                        "target": target.tolist(), "tracking_before_attempt": dict(zip(BODY_JOINTS, error.tolist())),
                        "payload_final": physics.data.xpos[physics.payload].tolist(),
                        "tracking_rmse_after": physics.last_tracking_error})
        print(json.dumps(results[-1]), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"backend": cfg.backend, "simulator": cfg.simulator, "level": "A_pinned_pelvis", "oracle_targets": True,
                                           "success_rate": float(np.mean([r["success"] for r in results])), "results": results}, indent=2))
finally:
    physics.close()
