"""Inspect a trusted .pt policy in its training environment."""

import argparse
import math
import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from common.training import make_runner, restore


def set_physics_dt(cfg, physics_dt: float) -> None:
    """Change physics resolution while preserving the trained control period."""
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("--physics-dt must be finite and positive")
    control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    ratio = control_dt / physics_dt
    if not math.isfinite(ratio):
        raise ValueError("--physics-dt is too small")
    decimation = round(ratio)
    if decimation < 1 or not math.isclose(ratio, decimation, rel_tol=1e-9):
        raise ValueError(
            f"--physics-dt must divide the policy period ({control_dt:g} s) exactly"
        )
    cfg.sim.mujoco.timestep = physics_dt
    cfg.decimation = decimation


def apply_hand_load(env, env_ids, mass_kg: float, asset_cfg: SceneEntityCfg) -> None:
    """Apply weight-equivalent world-frame force, without adding payload inertia."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    gravity = torch.as_tensor(
        env.sim.mj_model.opt.gravity.copy(),
        device=env.device,
        dtype=asset.data.body_external_wrench.dtype,
    )
    forces = (mass_kg * gravity).expand(len(env_ids), len(asset_cfg.body_ids), 3)
    asset.write_external_wrench_to_sim(
        forces,
        torch.zeros_like(forces),
        env_ids=env_ids,
        body_ids=asset_cfg.body_ids,
    )


def set_hand_loads(cfg, left_kg: float, right_kg: float) -> None:
    """Attach persistent wrist loads at each reset of a playback environment."""
    for mass_kg in (left_kg, right_kg):
        if not math.isfinite(mass_kg) or mass_kg < 0:
            raise ValueError("Hand loads must be finite, non-negative values in kg")
    for side, mass_kg in (("left", left_kg), ("right", right_kg)):
        if mass_kg == 0:
            continue
        # External wrenches persist through physics steps; simulation reset clears
        # them, so reapply on every reset (including falls and viewer resets).
        cfg.events[f"play_{side}_hand_load"] = EventTermCfg(
            func=apply_hand_load,
            mode="reset",
            params={
                "mass_kg": mass_kg,
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=(f"{side}_wrist_yaw_link",)
                ),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list_tasks())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--viewer", choices=["native", "viser", "none"], default="native")
    parser.add_argument("--steps", type=int, default=100, help="Headless step count")
    parser.add_argument(
        "--physics-dt",
        type=float,
        help="Physics timestep in seconds; preserves the trained policy period",
    )
    parser.add_argument(
        "--left-hand-load-kg",
        type=float,
        default=0.0,
        help="Constant left wrist load in kg-equivalent weight (default: 0)",
    )
    parser.add_argument(
        "--right-hand-load-kg",
        type=float,
        default=0.0,
        help="Constant right wrist load in kg-equivalent weight (default: 0)",
    )
    args = parser.parse_args()
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    try:
        if args.physics_dt is not None:
            set_physics_dt(cfg, args.physics_dt)
        set_hand_loads(cfg, args.left_hand_load_kg, args.right_hand_load_kg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.left_hand_load_kg or args.right_hand_load_kg:
        print(
            f"Hand loads (constant weight-equivalent force): "
            f"left={args.left_hand_load_kg:g} kg, right={args.right_hand_load_kg:g} kg"
        )
    env, runner = make_runner(args.task, cfg, load_rl_cfg(args.task), args.device)
    try:
        restore(runner, env, args.checkpoint, args.device)
        policy = runner.get_inference_policy(device=args.device)
        if args.viewer == "none":
            obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, reward, _, _ = env.step(policy(obs))
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("Non-finite rollout reward")
            print(f"Completed {args.steps} policy steps")
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

            viewer_cls = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
            viewer_cls(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
