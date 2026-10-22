"""Evaluate a fixed policy under shared linear-velocity noise and latency."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from common.training import make_runner, restore
from scripts.play import set_physics_dt
from tasks.registry import load_env_cfg, load_env_cls, load_rl_cfg


class VelocityInput:
    """Noise at measurement time, then fixed per-environment FIFO delay.

    Conditions share standardized noise draws for a given replicate. The caller
    supplies an episode-reset mask so history never leaks across resets.
    """

    def __init__(self, amplitudes, lags, trials, seed, device):
        self.trials = trials
        self.amplitudes = torch.tensor(amplitudes, device=device).repeat_interleave(trials)
        self.lags = torch.tensor(lags, device=device).repeat_interleave(trials)
        self.n = len(self.lags)
        self.size = max(lags) + 1
        self.buffer = torch.zeros(self.size, self.n, 3, device=device)
        self.ids = torch.arange(self.n, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.index = -1
        self.last = None

    def _measure(self, values):
        # Broadcast the same random realization across conditions, not trials.
        draws = torch.rand(self.trials, 3, generator=self.generator, device=values.device)
        noise = (2 * draws - 1).repeat(self.n // self.trials, 1)
        return values + noise * self.amplitudes[:, None]

    def reset(self, values, env_ids):
        """Replace selected histories without advancing other environments."""
        if self.last is None:
            return self.push(values, torch.ones(self.n, device=values.device, dtype=torch.bool))
        measured = self._measure(values)
        self.buffer[:, env_ids] = measured[env_ids]
        self.last[env_ids] = measured[env_ids]
        return self.last

    def push(self, values, reset_mask):
        measured = self._measure(values)
        self.index = (self.index + 1) % self.size
        if self.last is None:
            self.buffer[:] = measured
        self.buffer[:, reset_mask] = measured[reset_mask]
        self.buffer[self.index] = measured
        self.last = self.buffer[(self.index - self.lags) % self.size, self.ids].clone()
        return self.last


def velocity_observation(env, sensor_name, amplitudes, lags, trials, seed, target):
    # Observation-manager initialization may query shapes before the first reset.
    # All real samples are produced once, before the momentum observer update.
    if not hasattr(env, "velocity_sensitivity_input"):
        return builtin_sensor(env, sensor_name).clone()
    return env.velocity_sensitivity_input.last.clone()


class VelocitySensitivityMixin:
    def _prepare_observer_inputs(self, env_ids=None):
        params = self.cfg.observations["actor"].terms["base_lin_vel"].params
        if not hasattr(self, "velocity_sensitivity_input"):
            self.velocity_sensitivity_input = VelocityInput(
                params["amplitudes"], params["lags"], params["trials"],
                params["seed"], self.device,
            )
            # G1's velocimeter is body-aligned but offset from the free-joint
            # origin. Confirm the layout instead of treating sensor and qvel
            # linear velocities as interchangeable.
            model = self.sim.mj_model
            site_id = int(model.sensor(params["sensor_name"]).objid[0])
            body_id = int(model.site_bodyid[site_id])
            joint_id = int(model.body_jntadr[body_id])
            if (model.jnt_type[joint_id] != 0 or model.jnt_qposadr[joint_id] != 0
                    or model.jnt_dofadr[joint_id] != 0
                    or not np.allclose(model.site_quat[site_id], [1, 0, 0, 0])):
                raise ValueError("Evaluation requires a body-aligned sensor on the root free joint")
            self.velocity_sensor_offset = torch.as_tensor(
                model.site_pos[site_id].copy(), device=self.device, dtype=self.sim.data.qvel.dtype
            ).expand(self.num_envs, 3)
            self.velocity_clean_sample = torch.zeros(self.num_envs, 3, device=self.device)
        quat = self.sim.data.qpos[:, 3:7]
        qvel = self.sim.data.qvel
        # Reconstruct the current velocimeter sample before observer dynamics;
        # sim.sense() is later in step(). Checked against that sensor every step.
        clean = quat_apply_inverse(quat, qvel[:, :3]) + torch.cross(
            qvel[:, 3:6], self.velocity_sensor_offset, dim=-1
        )
        channel = self.velocity_sensitivity_input
        if env_ids is None:
            measured = channel.push(clean, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
            self.velocity_clean_sample[:] = clean
            selected = slice(None)
        else:
            measured = channel.reset(clean, env_ids)
            self.velocity_clean_sample[env_ids] = clean[env_ids]
            selected = env_ids
        if params["target"] == "shared":
            # Apply the SAME sensor-space error to observer free-joint velocity.
            # Subtracting the clean sensor cancels its omega x offset term and
            # preserves the original observer input exactly when error is zero.
            delta = measured - clean
            self.ideal_qvel_view[selected, :3] = (
                qvel[selected, :3] + quat_apply(quat[selected], delta[selected])
            )
            observed_delta = quat_apply_inverse(
                quat[selected], self.ideal_qvel_view[selected, :3] - qvel[selected, :3]
            )
            torch.testing.assert_close(observed_delta, delta[selected], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(self.ideal_qvel_view[selected, 3:6], qvel[selected, 3:6])


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))
            if len(values) > 1 else 0.0, "n": len(values)}


def make_plots(output, results, amplitudes, delays, target="actor only"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("vxy_rmse_mps", "Planar velocity vector RMSE (m/s)", "viridis", None),
        ("fall_rate", "Fallen trials (%)", "magma", 100),
        ("lateral_drift_m", "Final lateral drift |dy| (m)", "viridis", None),
        ("heading_drift_deg", "Final unwrapped heading drift (deg)", "viridis", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, title, cmap, vmax) in zip(axes.flat, panels):
        data = np.full((len(amplitudes), len(delays)), np.nan)
        for i, row in enumerate(results):
            value = row[metric] * 100 if metric == "fall_rate" else row[metric]["mean"]
            if value is not None:
                data[i // len(delays), i % len(delays)] = value
        im = ax.imshow(data, cmap=cmap, origin="lower", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(delays)), delays)
        ax.set_yticks(range(len(amplitudes)), [f"±{a:g}" for a in amplitudes])
        ax.set_xlabel("Fixed observation delay (ms)")
        ax.set_ylabel("Uniform noise half-width (m/s), each xyz axis")
        ax.set_title(title)
        for (y, x), value in np.ndenumerate(data):
            label = "—" if np.isnan(value) else f"{value:.2f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=10,
                    color="white" if np.isnan(value) or im.norm(value) < .55 else "black")
        fig.colorbar(im, ax=ax, shrink=.82)
    fig.suptitle(f"ADAPT: {target} linear-velocity sensitivity | command = 0.8 m/s", fontsize=15)
    fig.savefig(output / "sensitivity_heatmaps.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric, ylabel in zip(
        axes, ("vxy_rmse_mps", "yaw_rate_rmse_radps"),
        ("Planar velocity vector RMSE (m/s)", "Yaw-rate RMSE (rad/s)"),
    ):
        for amplitude in amplitudes:
            rows = [r for r in results if r["noise_mps"] == amplitude]
            ax.errorbar([r["delay_ms"] for r in rows],
                        [r[metric]["mean"] for r in rows],
                        yerr=[r[metric]["std"] for r in rows],
                        label=f"±{amplitude:g} m/s", marker="o", capsize=3, linewidth=1.6)
        ax.set_xlabel("Fixed linear-velocity observation delay (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(delays)
        ax.grid(alpha=.25)
        ax.legend(title="Uniform linear-velocity noise", fontsize=9)
    fig.suptitle(f"ADAPT: {target} | {results[0]['trials']} trials/condition | mean ± 1 SD")
    fig.savefig(output / "sensitivity_curves.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale")
    parser.add_argument("--physics-dt", type=float, default=.001)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--warmup", type=float, default=2.)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["shared", "actor"], default="shared",
                        help="Share one corrupted measurement with actor and observer (default), or actor-only control")
    parser.add_argument("--noise", type=float, nargs="+", default=[0, .05, .1, .2, .5])
    parser.add_argument("--delay-ms", type=int, nargs="+", default=[0, 20, 40, 80, 120])
    args = parser.parse_args()
    if args.trials < 1 or not 0 <= args.warmup < args.seconds:
        parser.error("Need positive trials and 0 <= warmup < seconds")
    if any(not np.isfinite(a) or a < 0 for a in args.noise):
        parser.error("Noise amplitudes must be finite and non-negative")
    cfg = load_env_cfg(args.task, play=True)
    set_physics_dt(cfg, args.physics_dt)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    if any(d < 0 or not np.isclose(d / (dt * 1000), round(d / (dt * 1000)))
           for d in args.delay_ms):
        parser.error("Delays must be non-negative integer multiples of the control period")
    conditions = [(a, d) for a in args.noise for d in args.delay_ms]
    cfg.scene.num_envs = n = len(conditions) * args.trials
    cfg.seed = args.seed
    actor = deepcopy(cfg.observations["actor"])
    actor.enable_corruption = True  # Fixed angular/gravity noise, as in training.
    actor.terms["base_lin_vel"] = replace(
        actor.terms["base_lin_vel"], func=velocity_observation, noise=None,
        delay_min_lag=0, delay_max_lag=0,
        params={"sensor_name": "robot/imu_lin_vel",
                "amplitudes": [a for a, _ in conditions],
                "lags": [round(d / (dt * 1000)) for _, d in conditions],
                "trials": args.trials, "seed": args.seed + 1000, "target": args.target},
    )
    cfg.observations["actor"] = actor
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "task": args.task, "seed": args.seed, "trials_per_condition": args.trials,
        "physics_dt": args.physics_dt, "control_dt": dt,
        "duration_s": args.seconds, "tracking_warmup_s": args.warmup,
        "command": [.8, 0, 0], "noise_half_widths_mps": args.noise,
        "delays_ms": args.delay_ms, "perturbation_target": args.target,
        "measurement_pipeline": "One body-frame IMU linear-velocity sample per control tick, noise then FIFO; actor uses sample directly; shared observer uses same sensor error rotated to world root qvel before dynamics and momentum; current orientation and angular velocity retained; lever-arm correction included",
        "noise_model": "iid uniform per axis/sample; measurement noise -> FIFO -> five-frame history",
        "other_conditions": {"actor_angular_noise_radps": .2,
                             "actor_projected_gravity_noise": .05,
                             "joint_velocity_noise_radps": .5,
                             "joint_position_noise_rad": .01,
                             "encoder_bias_rad": .015,
                             "observer_linear_noise": "same sample as actor" if args.target == "shared" else 0,
                             "observer_linear_delay": "same FIFO as actor" if args.target == "shared" else 0,
                             "observer_angular_noise_radps": 0,
                             "external_load_N": 0},
        "trials": "10 randomized replicas by default, not independently trained models; initial encoder biases and standardized linear noise paired across conditions",
        "failure_definition": "First existing fell_over termination; trial censored thereafter",
        "tracking_definition": "Per-trial true body-frame velocity RMSE after warmup, before first fall; always report fall rate alongside",
        "primary_metric": "vxy_rmse_mps: sqrt(mean_t((vx-vx_cmd)^2 + (vy-vy_cmd)^2)); no division by number of axes; summary reports mean and SD of per-trial vector RMSE; yaw rate separate",
        "drift_definition": "Absolute world y displacement and absolute unwrapped yaw change from initial pose to last nonterminal sample; failed trials reported separately",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Evaluating {len(conditions)} conditions x {args.trials} trials on CUDA; {args.seconds}s simulated", flush=True)
    evaluation_cls = type("VelocitySensitivityEnv", (VelocitySensitivityMixin, load_env_cls(args.task)), {})
    env, runner = make_runner(args.task, cfg, load_rl_cfg(args.task), "cuda:0", env_cls=evaluation_cls)
    try:
        restore(runner, env, args.checkpoint, "cuda:0")
        raw = env.unwrapped
        # Same initial sensor bias for each replicate across all grid conditions.
        raw._encoder_bias[:] = raw._encoder_bias[:args.trials].clone().repeat(len(conditions), 1)
        policy = runner.get_inference_policy(device="cuda:0")
        obs = env.get_observations()
        robot = raw.scene["robot"]
        start_pos = robot.data.root_link_pos_w.clone()
        previous_yaw = robot.data.heading_w.clone()
        final_pos = start_pos.clone()
        yaw_sum = torch.zeros(n, device=raw.device)
        alive = torch.ones(n, device=raw.device, dtype=torch.bool)
        time_to_fall = torch.full((n,), args.seconds, device=raw.device)
        sums = torch.zeros(n, 3, device=raw.device)
        counts = torch.zeros(n, device=raw.device)
        trajectories, headings, alive_samples, sample_times = [], [], [], []
        steps = round(args.seconds / dt)
        started = time.monotonic()
        with torch.inference_mode():
            for step in range(steps):
                obs, reward, _, _ = env.step(policy(obs))
                if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite rollout values")
                command = raw.command_manager.get_command("twist")
                torch.testing.assert_close(command, command.new_tensor([.8, 0, 0]).expand(n, 3))
                # Verify the corrupted, delayed sample enters the newest history slot.
                torch.testing.assert_close(obs["actor"][:, 12:15], raw.velocity_sensitivity_input.last)
                torch.testing.assert_close(builtin_sensor(raw, "robot/imu_lin_vel"),
                                           raw.velocity_clean_sample, atol=2e-5, rtol=2e-4)
                done = raw.reset_terminated
                just_failed = alive & done
                time_to_fall[just_failed] = (step + 1) * dt
                alive &= ~done
                pos = robot.data.root_link_pos_w
                yaw = robot.data.heading_w
                delta = torch.atan2(torch.sin(yaw - previous_yaw), torch.cos(yaw - previous_yaw))
                yaw_sum[alive] += delta[alive]
                previous_yaw = yaw.clone()
                final_pos[alive] = pos[alive]
                if (step + 1) * dt > args.warmup:
                    vel = robot.data.root_link_lin_vel_b
                    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
                    errors = torch.stack([vel[:, 0] - .8, vel[:, 1], yaw_rate], dim=1)
                    sums[alive] += errors[alive].square()
                    counts[alive] += 1
                if step % 10 == 0 or step == steps - 1:
                    trajectories.append((final_pos - start_pos).cpu().numpy())
                    headings.append(yaw_sum.cpu().numpy())
                    alive_samples.append(alive.cpu().numpy())
                    sample_times.append((step + 1) * dt)
                if (step + 1) % 250 == 0 or step == steps - 1:
                    print(f"Progress {step+1}/{steps}, elapsed={time.monotonic()-started:.1f}s, first falls={(~alive).sum().item()}/{n}", flush=True)
        rms = (sums / counts.clamp(min=1)[:, None]).sqrt().cpu().numpy()
        alive_np = alive.cpu().numpy()
        valid = counts.cpu().numpy() > 0
        disp = (final_pos - start_pos).cpu().numpy()
        yaw_deg = np.rad2deg(yaw_sum.cpu().numpy())
        times = time_to_fall.cpu().numpy()
        rows, summaries = [], []
        for ci, (amplitude, delay) in enumerate(conditions):
            condition_rows = []
            for trial in range(args.trials):
                i = ci * args.trials + trial
                row = {"noise_mps": amplitude, "delay_ms": delay, "trial": trial,
                       "fell": not bool(alive_np[i]), "observed_survival_s": float(times[i]),
                       "tracking_samples": int(counts[i].item()),
                       "vx_rmse_mps": float(rms[i, 0]) if valid[i] else None,
                       "vy_rmse_mps": float(rms[i, 1]) if valid[i] else None,
                       "vxy_rmse_mps": float(np.hypot(rms[i, 0], rms[i, 1])) if valid[i] else None,
                       "yaw_rate_rmse_radps": float(rms[i, 2]) if valid[i] else None,
                       "lateral_drift_m": float(abs(disp[i, 1])),
                       "heading_drift_deg": float(abs(yaw_deg[i])),
                       "forward_displacement_m": float(disp[i, 0])}
                rows.append(row)
                condition_rows.append(row)
            summary = {"noise_mps": amplitude, "delay_ms": delay,
                       "fall_rate": sum(r["fell"] for r in condition_rows) / args.trials,
                       "trials": args.trials}
            for key in ("vxy_rmse_mps", "vx_rmse_mps", "vy_rmse_mps", "yaw_rate_rmse_radps", "lateral_drift_m",
                        "heading_drift_deg", "forward_displacement_m", "observed_survival_s"):
                summary[key] = stats([r[key] for r in condition_rows])
            summaries.append(summary)
        (args.output / "trials.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(args.output / "trajectories.npz", position=trajectories,
                            heading=headings, alive=alive_samples, time=sample_times)
        make_plots(args.output, summaries, args.noise, args.delay_ms,
                   "actor + observer" if args.target == "shared" else "actor only")
        print(f"Evaluation complete: {args.output.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
