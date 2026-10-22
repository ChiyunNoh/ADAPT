"""Re-evaluate saved complete rollouts using ADAPT Appendix D metrics.

Velocity RMSE retains the full-rate sufficient statistic. Drift endpoints must
be present at the exact first and last tracking samples; no interpolation or
velocity estimation from the downsampled trajectories is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PAPER_URL = "https://arxiv.org/html/2606.16542v1#A4"
METRICS = (
    ("vx_rmse_mps", "Velocity error (m/s)"),
    ("lateral_drift_m", "Lateral drift (m)"),
    ("yaw_drift_rad", "Yaw drift (rad)"),
)


def endpoint_metrics(position_start, position_end, heading_start, heading_end):
    delta = np.asarray(heading_end, dtype=np.float64) - np.asarray(heading_start, dtype=np.float64)
    return (np.abs(np.asarray(position_end, dtype=np.float64)[..., 1]
                   - np.asarray(position_start, dtype=np.float64)[..., 1]),
            np.abs(np.arctan2(np.sin(delta), np.cos(delta))))


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty, finite complete-trial measurements")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0., "n": len(values)}


def make_plots(output, summaries, noises, delays, n, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    for ax, (metric, label) in zip(axes, METRICS):
        for noise in noises:
            rows = [r for r in summaries if r["noise_mps"] == noise]
            ax.errorbar(delays, [r[metric]["mean"] for r in rows],
                        yerr=[r[metric]["std"] for r in rows],
                        label=f"±{noise:g} m/s", marker="o", capsize=3, linewidth=1.7)
        ax.set(xlabel="Fixed linear-velocity delay (ms)", ylabel=label, xticks=delays)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
    axes[0].legend(title="Uniform noise half-width per xyz axis", ncol=3, fontsize=10)
    fig.suptitle(f"ADAPT Appendix D metrics | actor + observer shared input\n"
                 f"{n} trials/condition | {window[0]:.2f}–{window[1]:.2f} s | mean ± 1 SD", fontsize=14)
    fig.savefig(output / "paper_metrics_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (metric, label) in zip(axes, METRICS):
        matrix = np.array([r[metric]["mean"] for r in summaries]).reshape(len(noises), len(delays))
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if im.norm(value) < .55 else "black", fontsize=10)
        ax.set(title=label, xlabel="Delay (ms)", ylabel="Noise half-width (m/s)",
               xticks=range(len(delays)), yticks=range(len(noises)))
        ax.set_xticklabels(delays)
        ax.set_yticklabels([f"±{a:g}" for a in noises])
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(output / "paper_metrics_heatmaps.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metas = [json.loads((p / "metadata.json").read_text()) for p in args.runs]
    keys = ("checkpoint_sha256", "task", "physics_dt", "control_dt", "duration_s",
            "tracking_warmup_s", "command", "noise_half_widths_mps", "delays_ms",
            "other_conditions", "trials_per_condition", "noise_model", "perturbation_target", "measurement_pipeline")
    for meta in metas:
        for key in keys:
            if meta[key] != metas[0][key]:
                raise ValueError(f"Mismatched experiment setting: {key}")
        if meta["perturbation_target"] != "shared":
            raise ValueError("This analysis expects actor/observer shared measurements")
    seeds = [m["seed"] for m in metas]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds")
    all_rows, source_hashes = [], {}
    window = None
    for path, meta in zip(args.runs, metas):
        rows = json.loads((path / "trials.json").read_text())
        if any(r["fell"] or not r["tracking_samples"] for r in rows):
            raise ValueError("Exact reanalysis requires complete trials; rerun failed trials with full-rate endpoints")
        with np.load(path / "trajectories.npz") as archive:
            times = archive["time"].astype(np.float64)
            # Match the original strict warmup cutoff and its full-rate samples.
            dt = meta["control_dt"]
            control_times = np.arange(1, round(meta["duration_s"] / dt) + 1) * dt
            tracking_times = control_times[control_times > meta["tracking_warmup_s"]]
            start, end = float(tracking_times[0]), float(tracking_times[-1])
            start_ids = np.flatnonzero(np.isclose(times, start, atol=1e-9, rtol=0))
            end_ids = np.flatnonzero(np.isclose(times, end, atol=1e-9, rtol=0))
            if len(start_ids) != 1 or len(end_ids) != 1:
                raise ValueError("Exact evaluation endpoints absent; rerun with full-rate logging")
            if not archive["alive"].all() or any(r["tracking_samples"] != len(tracking_times) for r in rows):
                raise ValueError("Incomplete tracking window")
            first, last = int(start_ids[0]), int(end_ids[0])
            positions = archive["position"].astype(np.float64)
            heading = archive["heading"].astype(np.float64)
            if positions.shape[1] != len(rows) or not np.isfinite(positions).all() or not np.isfinite(heading).all():
                raise ValueError("Invalid trajectory data")
            lateral, yaw = endpoint_metrics(positions[first], positions[last], heading[first], heading[last])
            window = [start, end]
            for i, old in enumerate(rows):
                ci, trial = divmod(i, meta["trials_per_condition"])
                noise_index, delay_index = divmod(ci, len(meta["delays_ms"]))
                if (old["noise_mps"], old["delay_ms"], old["trial"]) != (
                    meta["noise_half_widths_mps"][noise_index], meta["delays_ms"][delay_index], trial
                ):
                    raise ValueError("Trajectory/trial order mismatch")
                all_rows.append({
                    "seed": meta["seed"], "trial": trial,
                    "noise_mps": old["noise_mps"], "delay_ms": old["delay_ms"],
                    "vx_rmse_mps": old["vx_rmse_mps"],
                    "lateral_drift_m": float(lateral[i]), "yaw_drift_rad": float(yaw[i]),
                    "vxy_rmse_mps": float(np.hypot(old["vx_rmse_mps"], old["vy_rmse_mps"])),
                    "lateral_start_m": float(positions[first, i, 1]),
                    "lateral_end_m": float(positions[last, i, 1]),
                    "unwrapped_yaw_delta_rad": float(heading[last, i] - heading[first, i]),
                    "evaluation_start_s": start, "evaluation_end_s": end,
                    "tracking_samples": len(tracking_times), "fell": False,
                })
        for filename in ("metadata.json", "trials.json", "trajectories.npz"):
            src = path / filename
            source_hashes[str(src.resolve())] = hashlib.sha256(src.read_bytes()).hexdigest()
    meta = dict(metas[0])
    for key in ("seed", "primary_metric", "drift_definition", "tracking_definition"):
        meta.pop(key, None)
    meta.update({
        "analysis": "Exact reanalysis of saved complete rollouts, no new simulation",
        "paper_url": PAPER_URL, "metric_reference": "Appendix D equations 14–16",
        "random_seeds": seeds, "trials_per_condition": metas[0]["trials_per_condition"] * len(seeds),
        "source_runs": [str(p.resolve()) for p in args.runs],
        "evaluation_window_s": window,
        "primary_metrics": [key for key, _ in METRICS],
        "metric_definitions": {"vx_rmse_mps": "sqrt(mean((true body root vx - command vx)^2))",
            "lateral_drift_m": "abs(world y at last tracking sample - world y at first tracking sample)",
            "yaw_drift_rad": "abs(atan2(sin(heading_end-heading_start), cos(heading_end-heading_start)))"},
        "aggregation": "mean and sample SD of per-trial metrics, wrap each trial before averaging",
        "protocol_difference_from_paper": "Velocity-input noise/delay sweep, no torso force; 30 s rollout with 2 s excluded, versus paper Figure 3's 1 s zero-command settling then 10 s forced walking; metric definitions only are aligned",
        "trials": "Repeated complete rollouts of one fixed trained policy, not independent training seeds",
    })
    summaries = []
    for a in meta["noise_half_widths_mps"]:
        for d in meta["delays_ms"]:
            selected = [r for r in all_rows if r["noise_mps"] == a and r["delay_ms"] == d]
            assert len(selected) == meta["trials_per_condition"]
            result = {"noise_mps": a, "delay_ms": d, "trials": len(selected), "fall_rate": 0.}
            for key in [key for key, _ in METRICS] + ["vxy_rmse_mps"]:
                result[key] = statistics([r[key] for r in selected])
            summaries.append(result)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("metadata", meta), ("trials", all_rows), ("summary", summaries), ("source_hashes", source_hashes)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    make_plots(args.output, summaries, meta["noise_half_widths_mps"], meta["delays_ms"], meta["trials_per_condition"], window)
    print(f"Re-evaluated {len(all_rows)} trials on exact window {window}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
