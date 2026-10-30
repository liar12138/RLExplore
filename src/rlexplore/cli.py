import argparse
from pathlib import Path
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Frozen SONIC upper-policy active placement")
    parser.add_argument("command", choices=["doctor", "smoke", "train", "eval", "manifest"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--baseline", choices=["scan", "geometry", "reactive"])
    parser.add_argument("--output", default="runs/eval.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .config import load_config
    cfg = load_config(args.config, args.set)
    if cfg.device == "cpu" and cfg.simulator != "mjlab":
        # Avoid CUDA/Triton driver probing in this process on the damaged host.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "train" and os.environ.get("MUJOCO_GL") == "osmesa":
        # Import Triton's LLVM before Mesa's LLVM. Reverse order segfaults in
        # this host's borrowed Torch runtime when Adam imports Dynamo lazily.
        import torch._dynamo
    if args.command == "doctor":
        import importlib.metadata
        import torch
        import mujoco
        from .sonic import FrozenSonic
        sonic = FrozenSonic(cfg.sonic_dir)
        from .physics import make_scene
        model = make_scene(cfg.asset_xml, [.04, .04, .035])
        data = {"python": sys.version, "torch": torch.__version__, "mujoco": mujoco.__version__,
                "simulator": cfg.simulator,
                "mjlab": importlib.metadata.version("mjlab"),
                "mujoco_warp": importlib.metadata.version("mujoco-warp"),
                "cuda_available": torch.cuda.is_available(), "asset": cfg.asset_xml,
                "joints": model.njnt, "actuators": model.nu, "sonic_hashes": sonic.hashes,
                "level": "A_pinned_pelvis"}
        print(json.dumps(data, indent=2))
    elif args.command == "smoke":
        from .env import PlacementEnv
        from .history import History, batch_histories
        from .model import ActorCritic
        from .ppo import resolve_device
        import numpy as np
        import torch
        torch.set_num_threads(2)
        env = PlacementEnv(cfg)
        try:
            obs, _ = env.reset(seed=cfg.seed)
            history = History(cfg.history_length)
            history.reset(obs)
            device = resolve_device(cfg.device)
            model = ActorCritic(cfg).to(device)
            log = []
            for action in [0, 2, 6, 8]:
                obs, _, terminated, truncated, info = env.step(action)
                history.append(obs)
                log.append(info)
                if terminated or truncated:
                    break
            batch, valid = batch_histories([history.snapshot()], device)
            dist, value = model(batch, valid)
            loss = -dist.log_prob(dist.sample()).mean()+value.square().mean()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            from PIL import Image
            run = Path(cfg.run_dir)
            run.mkdir(parents=True, exist_ok=True)
            Image.fromarray((obs["image"][:3].transpose(1, 2, 0)*255).astype(np.uint8)).save(run / "ego.png")
            report = {"level": "A_pinned_pelvis", "backend": cfg.backend, "backbone": cfg.backbone,
                      "simulator": cfg.simulator, "device": str(device),
                      "mjlab_gpu_steps": env.physics.runtime.steps if env.physics.runtime else 0,
                      "sonic_calls": env.physics.sonic.calls if env.physics.sonic else 0,
                      "observed_candidates": int(env.candidate_mask.sum()), "gradient_check": "finite", "steps": log}
            (run / "smoke.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            env.close()
    elif args.command == "train":
        from .ppo import train
        train(cfg, args.resume)
    elif args.command == "eval":
        from .ppo import evaluate
        evaluate(cfg, args.checkpoint, args.episodes, args.baseline, args.output)
    else:
        from .experiments import write_manifest
        write_manifest(args.output)


if __name__ == "__main__":
    main()
