"""Generate reviewable commands only; never launch a research sweep implicitly."""
import csv
import json
from pathlib import Path


def write_manifest(output):
    rows = []
    for suite in ("E0", "E1", "E2", "E3", "E7", "E9"):
        for backbone in ("reactive", "gru", "transformer", "ttt"):
            for seed in range(5):
                name = f"{suite}_{backbone}_{seed}"
                rows.append({"id": name, "suite": suite, "backbone": backbone, "seed": seed,
                             "overrides": {"suite": suite, "backbone": backbone, "seed": seed}, "kind": "architecture"})
    for ablation, overrides in {
        "no_map": {"use_map": False}, "no_object": {"use_object": False},
        "no_feedback": {"use_feedback": False}, "observable_critic": {"privileged_critic": False},
        "history_1": {"history_length": 1}, "time_cost_low": {"time_cost": .005},
        "time_cost_high": {"time_cost": .08},
    }.items():
        for seed in range(5):
            name = f"E2_transformer_{ablation}_{seed}"
            rows.append({"id": name, "suite": "E2", "backbone": "transformer", "seed": seed,
                         "overrides": {"suite": "E2", "backbone": "transformer", "seed": seed, **overrides}, "kind": "ablation"})
    for row in rows:
        args = {**row.pop("overrides"), "run_dir": f"runs/{row['id']}"}
        values = [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in args.items()]
        row["status"] = "planned"
        row["train_command"] = "./scripts/run.sh train --config configs/default.yaml --set " + " ".join(values)
        row["eval_command"] = (f"./scripts/run.sh eval --config runs/{row['id']}/config.yaml "
                               f"--checkpoint runs/{row['id']}/last.pt --episodes 100 --set split=test "
                               f"--output runs/{row['id']}/eval_test.json")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} planned runs to {path}; no training dispatched")
