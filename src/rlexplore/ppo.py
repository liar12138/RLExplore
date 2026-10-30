from dataclasses import asdict
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from .env import PlacementEnv
from .history import History, batch_histories
from .model import ActorCritic
from .config import save_config


def returns_gae(rewards, values, next_values, terminated, episode_ends, durations, gamma, lam):
    """Time-aware bootstrap; a timeout bootstraps but never leaks into next reset."""
    advantage = np.zeros(len(rewards), np.float32)
    running = 0.
    for t in reversed(range(len(rewards))):
        discount = gamma ** durations[t]
        delta = rewards[t] + discount * next_values[t] * (not terminated[t]) - values[t]
        running = delta + discount * lam ** durations[t] * (not episode_ends[t]) * running
        advantage[t] = running
    return advantage, advantage + np.asarray(values, np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix server driver/device access or explicitly use device=cpu; no silent fallback.")
    return torch.device(name)


def train(cfg, resume=None):
    set_seed(cfg.seed)
    torch.set_num_threads(2)
    device = resolve_device(cfg.device)
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")
    model = ActorCritic(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start_update = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt["config"].get("simulator", "mujoco_debug") != cfg.simulator:
            raise ValueError("checkpoint simulator differs; legacy MuJoCo runs cannot resume as mjlab runs")
        architecture = ("backbone", "hidden_dim", "layers", "heads", "max_candidates", "history_length", "privileged_critic")
        if any(ckpt["config"][k] != getattr(cfg, k) for k in architecture):
            raise ValueError("resume architecture/context differs from checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"]
        # Resume at a fresh episode, not at an un-restored simulator state.
        set_seed(cfg.seed + start_update)
    env = PlacementEnv(cfg)
    history = History(cfg.history_length)
    observation, _ = env.reset(seed=cfg.seed+start_update*100_000)
    history.reset(observation)
    steps, start = 0, time.monotonic()
    metadata = {"level": "A_pinned_pelvis", "backend": cfg.backend,
                "simulator": cfg.simulator,
                "sonic_hashes": env.physics.sonic.hashes if env.physics.sonic else None,
                "torch": torch.__version__, "device": str(device), "trained_modules": "upper_policy_only",
                "resume_semantics": "optimizer restored; simulator starts a new episode",
                "parameters": sum(p.numel() for p in model.parameters()),
                "observation_protocol": "rgbd_proprio_object_grasp_previous_skill_duration_feedback_evidence_v2",
                "render_backend": __import__("os").environ.get("MUJOCO_GL")}
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        with (run / "metrics.jsonl").open("a") as log:
            for update in range(start_update, cfg.updates):
                contexts, privileged, actions, old_logp = [], [], [], []
                rewards, values, next_values, terms, ends, durations, episodes = [], [], [], [], [], [], []
                model.eval()
                for _ in range(cfg.rollout_steps):
                    context = history.snapshot()
                    priv = env.privileged_state()
                    inputs, valid = batch_histories([context], device)
                    with torch.no_grad():
                        dist, value = model(inputs, valid, torch.as_tensor(priv[None], device=device))
                        action = dist.sample()
                        logp = dist.log_prob(action)
                    observation, reward, terminated, truncated, info = env.step(int(action.item()))
                    history.append(observation)
                    inputs, valid = batch_histories([history.snapshot()], device)
                    with torch.no_grad():
                        _, bootstrap = model(inputs, valid, torch.as_tensor(env.privileged_state()[None], device=device))
                    contexts.append(context)
                    privileged.append(priv)
                    actions.append(action.item())
                    old_logp.append(logp.item())
                    rewards.append(reward)
                    values.append(value.item())
                    next_values.append(bootstrap.item())
                    terms.append(terminated)
                    ends.append(terminated or truncated)
                    durations.append(info["duration"])
                    if terminated or truncated:
                        episodes.append(info)
                        observation, _ = env.reset()
                        history.reset(observation)
                advantage, returns = returns_gae(rewards, values, next_values, terms, ends, durations, cfg.gamma, cfg.gae_lambda)
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                old_logp, actions = torch.tensor(old_logp, device=device), torch.tensor(actions, device=device)
                advantage, returns = torch.tensor(advantage, device=device), torch.tensor(returns, device=device)
                stats = []
                model.train()
                for _ in range(cfg.ppo_epochs):
                    for ids in np.array_split(np.random.permutation(cfg.rollout_steps), max(1, int(np.ceil(cfg.rollout_steps/cfg.minibatch)))):
                        inputs, valid = batch_histories([contexts[i] for i in ids], device)
                        priv = torch.as_tensor(np.stack([privileged[i] for i in ids]), device=device)
                        dist, value = model(inputs, valid, priv)
                        logp = dist.log_prob(actions[ids])
                        ratio = (logp-old_logp[ids]).exp()
                        policy_loss = -torch.minimum(ratio*advantage[ids], ratio.clamp(1-cfg.clip, 1+cfg.clip)*advantage[ids]).mean()
                        value_loss = .5*(value-returns[ids]).square().mean()
                        entropy = dist.entropy().mean()
                        loss = policy_loss + cfg.value_coef*value_loss-cfg.entropy_coef*entropy
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite PPO objective")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError("nonfinite PPO gradient")
                        optimizer.step()
                        stats.append([policy_loss.item(), value_loss.item(), entropy.item()])
                steps += cfg.rollout_steps
                avg = np.mean(stats, axis=0)
                record = {"update": update+1, "decisions_this_session": steps, "wall_seconds": time.monotonic()-start,
                          "reward_mean": float(np.mean(rewards)), "policy_loss": float(avg[0]), "value_loss": float(avg[1]),
                          "entropy": float(avg[2]), "episodes": len(episodes),
                          "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else None,
                          "episode_results": episodes}
                log.write(json.dumps(record)+"\n")
                log.flush()
                checkpoint = {"config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "update": update+1, "metadata": metadata}
                tmp = run / "checkpoint.tmp"
                torch.save(checkpoint, tmp)
                tmp.replace(run / "last.pt")
                print(json.dumps({k: v for k, v in record.items() if k != "episode_results"}), flush=True)
    finally:
        env.close()


def evaluate(cfg, checkpoint=None, episodes=10, baseline=None, output=None):
    torch.set_num_threads(2)
    device = resolve_device(cfg.device)
    model = ActorCritic(cfg).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if ckpt["config"].get("simulator", "mujoco_debug") != cfg.simulator:
            raise ValueError("checkpoint simulator differs from evaluation configuration")
        architecture = ("backbone", "hidden_dim", "layers", "heads", "history_length", "max_candidates", "privileged_critic")
        if any(ckpt["config"][k] != getattr(cfg, k) for k in architecture):
            raise ValueError("evaluation architecture/context differs from checkpoint; use its saved config.yaml")
        model.load_state_dict(ckpt["model"])
    elif baseline is None:
        raise ValueError("evaluation requires a checkpoint or an explicit baseline")
    model.eval()
    env = PlacementEnv(cfg)
    results = []
    try:
        for episode in range(episodes):
            offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[cfg.split]
            obs, _ = env.reset(seed=offset+cfg.seed+episode)
            history = History(cfg.history_length)
            history.reset(obs)
            trajectory = []
            for step in range(cfg.max_decisions):
                if baseline:
                    from .baselines import choose_action
                    action = choose_action(baseline, obs, step)
                else:
                    batch, valid = batch_histories([history.snapshot()], device)
                    with torch.no_grad():
                        # No privileged-state call in evaluation.
                        dist, _ = model(batch, valid)
                    action = int(dist.probs.argmax(-1).item())
                obs, reward, term, trunc, info = env.step(action)
                trajectory.append({"action": action, "reward": reward, "reason": info["reason"], "duration": info["duration"]})
                history.append(obs)
                if term or trunc:
                    break
            results.append({**info, "trajectory": trajectory})
        from collections import Counter
        reasons = Counter(r["reason"] for r in results)
        summary = {"episodes": len(results), "success_rate": float(np.mean([r["success"] for r in results])),
                   "mean_elapsed": float(np.mean([r["elapsed"] for r in results])), "level": "A_pinned_pelvis",
                   "backend": cfg.backend, "simulator": cfg.simulator, "suite": cfg.suite, "split": cfg.split, "checkpoint": checkpoint,
                   "baseline": baseline, "terminal_reasons": dict(reasons),
                   "mean_observations": float(np.mean([r["observations"] for r in results])),
                   "mean_attempts": float(np.mean([r["attempts"] for r in results])), "results": results}
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
        return summary
    finally:
        env.close()
