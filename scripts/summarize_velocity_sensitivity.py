"""Pool compatible sensitivity runs across random seeds without hiding failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_velocity_sensitivity import make_plots, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = [json.loads((p / "metadata.json").read_text()) for p in args.runs]
    keys = ("checkpoint_sha256", "task", "physics_dt", "control_dt", "duration_s",
            "tracking_warmup_s", "command", "noise_half_widths_mps", "delays_ms",
            "other_conditions", "trials_per_condition", "noise_model", "perturbation_target")
    for meta in metadata[1:]:
        for key in keys:
            if meta[key] != metadata[0][key]:
                raise ValueError(f"Mismatched experiment setting: {key}")
        if meta.get("measurement_pipeline") != metadata[0].get("measurement_pipeline"):
            raise ValueError("Mismatched experiment setting: measurement_pipeline")
    seeds = [m["seed"] for m in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Cannot pool duplicate random seeds")
    rows = []
    for path, seed in zip(args.runs, seeds):
        rows.extend({**r, "seed": seed} for r in json.loads((path / "trials.json").read_text()))
    # Both axes were measured on exactly the same time samples. Reconstruct
    # each trial's vector RMSE before aggregating; do not combine axis means.
    for row in rows:
        vx, vy = row["vx_rmse_mps"], row["vy_rmse_mps"]
        row["vxy_rmse_mps"] = float(np.hypot(vx, vy)) if vx is not None and vy is not None else None
    meta = dict(metadata[0])
    meta.pop("seed")
    meta["random_seeds"] = seeds
    meta["source_runs"] = [str(p.resolve()) for p in args.runs]
    meta["trials_per_condition"] *= len(seeds)
    meta["trials"] = "Pooled randomized rollout replicas from one fixed policy, not independent training runs"
    meta["primary_metric"] = "vxy_rmse_mps: sqrt(mean_t((vx-vx_cmd)^2 + (vy-vy_cmd)^2)); no division by number of axes; summary reports mean and SD of per-trial vector RMSE; yaw rate separate"
    summaries = []
    for amplitude in meta["noise_half_widths_mps"]:
        for delay in meta["delays_ms"]:
            subset = [r for r in rows if r["noise_mps"] == amplitude and r["delay_ms"] == delay]
            assert len(subset) == meta["trials_per_condition"]
            result = {"noise_mps": amplitude, "delay_ms": delay, "trials": len(subset),
                      "fall_rate": sum(r["fell"] for r in subset) / len(subset)}
            for key in ("vxy_rmse_mps", "vx_rmse_mps", "vy_rmse_mps", "yaw_rate_rmse_radps", "lateral_drift_m",
                        "heading_drift_deg", "forward_displacement_m", "observed_survival_s"):
                result[key] = stats([r[key] for r in subset])
            summaries.append(result)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("metadata", meta), ("trials", rows), ("summary", summaries)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    make_plots(args.output, summaries, meta["noise_half_widths_mps"], meta["delays_ms"],
               "actor + observer" if meta["perturbation_target"] == "shared" else "actor only")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [(0, 0), (.05, 0), (.05, 120), (.5, 120)]
    grid = [(a, d) for a in meta["noise_half_widths_mps"] for d in meta["delays_ms"]]
    n = metadata[0]["trials_per_condition"]
    trajectories = [np.load(p / "trajectories.npz") for p in args.runs]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True, sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, selected):
        if condition not in grid:
            ax.set_visible(False)
            continue
        ci = grid.index(condition)
        for archive in trajectories:
            for trial in range(n):
                i = ci * n + trial
                alive = archive["alive"][:, i]
                xy = archive["position"][alive, i, :2]
                ax.plot(xy[:, 0], xy[:, 1], alpha=.4, linewidth=1)
        ax.plot([0, meta["command"][0] * meta["duration_s"]], [0, 0], "k--", label="Ideal path")
        ax.set_title(f"Noise ±{condition[0]:g} m/s | delay {condition[1]} ms")
        ax.set_xlabel("World forward displacement (m)")
        ax.set_ylabel("World lateral displacement (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.25)
    fig.suptitle(f"30 s walking paths | {meta['trials_per_condition']} trials/condition")
    fig.savefig(args.output / "walking_paths.png", dpi=170)
    plt.close(fig)
    for archive in trajectories:
        archive.close()
    print(f"Pooled {len(rows)} trials across seeds {seeds}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
