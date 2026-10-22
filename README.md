# ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion

Project page: [ADAPT](https://blyu413.github.io/adapt-locomotion/)

Paper: [arXiv:2606.16542](https://arxiv.org/abs/2606.16542)

![ADAPT overview](images/overview.png)

## Abstract

Humanoids deployed in human-centered environments must handle force-interactive tasks, where external contacts introduce unexpected disturbances that disrupt locomotion accuracy and stability. Existing learning-based approaches rely on broad domain randomization, task-specific force objectives, or learning-based force estimators from motion history, each of which compromises accuracy, task transferability, or out-of-distribution robustness. We present Analytical Disturbance-Aware Policy Training (ADAPT), a framework that equips humanoid policies with a physically grounded disturbance observer. ADAPT estimates residual force/torque online from accessible robot dynamics without requiring force/torque sensors, feeds these estimates directly into the policy, and uses them as a physics-derived signal for robust behavior under external forces, payloads, and contact-induced disturbances. Experiments on a Unitree G1 humanoid show accurate disturbance prediction, stronger robustness than a proprioception-only baseline, improved velocity tracking under out-of-distribution disturbances, and the ability to encourage lighter locomotion by penalizing inferred lower-body disturbances.

## Project Structure

The main directories and files are organized as follows:

```text
ADAPT/
├── common/         # Shared utilities for training and deployment
├── deployment/     # MuJoCo and Unitree G1 deployment code
├── images/
├── logs/rsl_rl/    # Released checkpoints and training outputs
├── models/g1/      # G1 MuJoCo models and mesh assets
├── observer/       # Momentum observer implementation
├── scripts/        # Training, playback, export, and monitoring scripts
├── tasks/          # MJLab task environments and configurations
├── pyproject.toml  # Project metadata, dependencies, and commands
└── uv.lock         # Locked dependency versions
```

## Installation

ADAPT uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

### Requirements

| Item | Details |
| --- | --- |
| Platform | Linux (Ubuntu 22.04) |
| Python | 3.10 |
| CUDA | 12.8 |

### 1. Install `uv`

Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/blyu413/ADAPT.git
cd ADAPT
```

### 3. Install the dependencies for your use case

```bash
uv sync --frozen
```

The following optional dependency groups are available:

| Use case | Command | Main additions |
| --- | --- | --- |
| Train, play, or export a PyTorch policy | `uv sync --frozen --extra train` | mjlab, PyTorch 2.9.1 + CUDA 12.8, MuJoCo Warp, RSL-RL |
| Use a controller in MuJoCo | `uv sync --frozen --extra controller` | pygame |
| Connect to a Unitree G1 | `uv sync --frozen --extra robot` | Unitree SDK2, CycloneDDS, pygame |
| Plot live or recorded telemetry | `uv sync --frozen --extra monitor` | Matplotlib |
| Install everything | `uv sync --frozen --all-extras` | All of the above |

Extras can be combined, for example:

```bash
uv sync --frozen --extra train --extra monitor
```

`uv sync` creates `.venv/` and performs an exact sync. When running it again,
include every extra that should remain installed. Use `uv run --frozen ...` for
the commands in this repository; activating the environment manually is
optional.

### 4. Verify the installation

Run the released ADAPT ONNX policy in MuJoCo:

```bash
uv run --frozen adapt-sim --seconds 10
```

Use the `--headless` option when running without a display.

## Unitree G1 Code Configuration

The on-robot setup for the Unitree G1 requires
[FAST-LIO](https://github.com/hku-mars/FAST_LIO) to provide the base linear
velocity estimate used by the deployment code.

Detailed installation and configuration instructions for our G1 setup are
still in preparation.

## Train & Deploy

### Tasks

| Task name | Description |
| --- | --- |
| `MOMO-Lumped-Flat-Unitree-G1-EffortScale` | ADAPT Stage 1 locomotion training |
| `MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale` | ADAPT Stage 2 hand-load training |
| `Ref-NoMoMo-Flat-Unitree-G1` | Proprioception-only baseline Stage 1 |
| `Ref-NoMoMo-Flat-Unitree-G1-HandLoad-CleanMix` | Proprioception-only baseline Stage 2 |
| `MOMO-Lumped-Flat-Unitree-G1-LegEffortSoftStep-NoActorFeedback` | Light-step training without observer feedback to the actor |

### Training

Run the two-stage ADAPT training pipeline:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --num-envs 4096 \
  --stage1-iterations 15000 \
  --stage2-iterations 20000 \
  --log-root logs/rsl_rl
```

The first stage trains the locomotion policy, and the second stage continues
training with randomized hand loads. The final Stage 1 checkpoint is loaded
automatically before Stage 2 begins.

ADAPT retains MJLab's multi-GPU launcher. Select multiple GPUs with a quoted
list; `--num-envs` is the number of environments on each GPU:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --gpu-ids "[0, 1]" \
  --num-envs 4096
```

TensorBoard is the offline-safe default. To log both stages to Weights & Biases,
select the W&B logger explicitly:

```bash
uv run --extra train python scripts/train_two_stage.py adapt \
  --logger wandb \
  --wandb-project adapt
```

Add `--upload-model` if checkpoints and exported models should also be uploaded.

The two stages can also be trained separately. Train Stage 1 with:

```bash
uv run --extra train python scripts/train.py \
  MOMO-Lumped-Flat-Unitree-G1-EffortScale \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 15000
```

The single-task entry point accepts MJLab's `--gpu-ids` and RSL-RL logger
configuration directly, for example `--gpu-ids "[0, 1]"` or
`--agent.logger wandb --agent.wandb-project adapt`.

Then continue with the Stage 2 hand-load task:

```bash
uv run --extra train python scripts/train.py \
  MOMO-Lumped-Flat-Unitree-G1-HandLoad-CleanMix-EffortScale \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 20000 \
  --resume-checkpoint <STAGE1_CHECKPOINT>
```

### Play and Export

Inspect a PyTorch checkpoint in its training environment:

```bash
uv run --frozen --extra train adapt-play <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

Alternatively, run the Python script directly:

```bash
uv run --extra train python scripts/play.py <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

Playback can use a finer physics timestep while keeping the trained 20 ms
policy/observer period. For 1 ms physics and a constant 2 kg-equivalent load
on the left wrist:

```bash
uv run --frozen --extra train python -m scripts.play <TASK_NAME> \
  --checkpoint <CHECKPOINT_PATH> \
  --physics-dt 0.001 \
  --left-hand-load-kg 2 \
  --viewer native
```

Use `--right-hand-load-kg` for the right wrist, or pass both options for
independent loads on both wrists. Each defaults to zero. Loads are constant
world-frame forces equal to mass times gravity (2 kg is 19.62 N downward
under default gravity), applied from the start and reapplied after resets.
They do not add a physical object, collision geometry, or payload inertia.
These options affect playback only; training settings remain unchanged.

Evaluate shared actor/observer linear-velocity input noise and fixed latency with a frozen
checkpoint (GPU required):

```bash
uv run --frozen --extra train python -m scripts.evaluate_velocity_sensitivity \
  --checkpoint <CHECKPOINT_PATH> \
  --output logs/evaluation/velocity_sensitivity
```

The default sweep uses 1 ms physics, the trained 20 ms control period, a fixed
0.8 m/s forward command, no external loads, and 10 randomized trials of 30 s
per condition. Uniform noise half-widths are 0, 0.05, 0.1, 0.2, and 0.5 m/s;
fixed delays are 0, 20, 40, 80, and 120 ms. Noise is applied before delay and
observation history. One corrupted sample feeds both the actor's three linear
velocity channels and the momentum observer, before the observer's dynamics and
momentum computation. The observer input accounts for the IMU site offset and
converts the same measurement error to world-frame free-joint velocity. Delay
applies to linear velocity only; orientation and angular velocity remain current.
Other observation noise stays at the training settings (actor angular velocity
±0.2 rad/s; observer angular velocity unperturbed). Pass `--target actor` for an
actor-only control experiment. Training behavior is unchanged.
The output directory must be new; it receives experiment
metadata, per-trial metrics, summaries, sampled trajectories, and heatmaps.
Tracking RMSE excludes the first 2 s and is censored at the first fall; report
it alongside the fall rate. The primary tracking metric is planar velocity
vector RMSE: `sqrt(mean((vx-vx_cmd)^2 + (vy-vy_cmd)^2))`, in m/s, without
dividing by the number of axes. Per-trial values are aggregated as mean and
standard deviation. Axis-specific RMSE remains available for diagnosis;
yaw-rate RMSE is reported separately in rad/s. Older runs can be re-summarized
without rerunning the simulation, using each trial's saved axis RMSE values.
Drift is measured up to the last nonterminal
sample, using unwrapped heading so full turns are not hidden.

Repeat with different `--seed` values and new output directories to check
repeatability. Pool compatible runs and plot their mean and standard deviation:

```bash
uv run --frozen --extra train python -m scripts.summarize_velocity_sensitivity \
  <RUN_DIRECTORY_1> <RUN_DIRECTORY_2> <RUN_DIRECTORY_3> \
  --output logs/evaluation/velocity_sensitivity_combined
```

To evaluate the three metrics in ADAPT Appendix D (equations 14–16), use:

```bash
.venv/bin/python -m scripts.evaluate_paper_metrics \
  <RUN_DIRECTORY_1> <RUN_DIRECTORY_2> <RUN_DIRECTORY_3> \
  --output logs/evaluation/velocity_sensitivity_paper_metrics
```

This preserves each trial's full-rate forward-velocity RMSE and computes
absolute world-y drift and absolute wrapped yaw drift in radians over the same
tracking window. With the default settings, the first and last samples are
2.02 s and 30.00 s. Exact endpoint samples must exist in the saved trajectories;
the analyzer rejects incomplete trials or missing endpoints instead of
interpolating. Wrapping is applied per trial before averaging. This matches the
paper's metric definitions, while retaining this experiment's sensor noise and
delay protocol; it does not reproduce Figure 3's torso-force experiment.

Export a checkpoint for deployment:

```bash
uv run --frozen --extra train adapt-export <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --output <OUTPUT_PATH>
```

Alternatively, run the Python script directly:

```bash
uv run --extra train python scripts/export_onnx.py \
  <TASK_NAME> \
  --checkpoint <CHECKPOINT_PATH> \
  --output <OUTPUT_PATH>
```

The export command creates a validated ONNX policy and its matching JSON
configuration. Released checkpoints and task names are listed in
[`logs/rsl_rl/README.md`](logs/rsl_rl/README.md).

### MuJoCo Deployment

Run one of the released policies in MuJoCo:

```bash
uv run --frozen adapt-sim --policy adapt --seconds 20
```

Alternatively, run the Python script directly:

```bash
uv run python deployment/deploy_mujoco_mjlab.py --policy adapt --seconds 20
```

Available released policies are `adapt`, `baseline`, and `lightstep`. The policy
name resolves the matching ONNX and JSON files from `logs/rsl_rl/`; a task name
is not required for deployment. To load a custom export, use
`--checkpoint <ONNX_PATH>` and keep its matching JSON file in the same directory.

`MujocoRobot` reads a robot-state snapshot and applies controls,
`ObservationGenerator` builds observations and their history, and
`PolicyManager` selects the active policy. The control loop in
[`deployment/deploy_mujoco_mjlab.py`](deployment/deploy_mujoco_mjlab.py) runs
observation → ONNX inference → action processing → MuJoCo step → MoMo update.
ADAPT uses the approximate MoMo calculation in
[`observer/momo.py`](observer/momo.py).

To load an ONNX/JSON pair directly without using `manifest.json`:

```bash
uv run --frozen adapt-sim --checkpoint <ONNX_PATH> --seconds 20
```

```bash
uv run python deployment/deploy_mujoco_mjlab.py \
  --checkpoint <ONNX_PATH> \
  --seconds 20
```

To use a wired Xbox-compatible controller in MuJoCo, first install the
`controller` extra. Add policies with `--add-policy`, then pass `--controller`:
the left stick sets forward/lateral velocity, the right stick sets yaw rate,
and releasing START/SELECT switches to the next/previous policy. Switching
keeps the robot pose but resets the selected policy's observation history and
disturbance observer.

```bash
uv run --frozen --extra controller adapt-sim --policy adapt --add-policy baseline --controller --seconds 20
```

```bash
uv run --extra controller python deployment/deploy_mujoco_mjlab.py \
  --policy adapt --add-policy baseline --controller --seconds 20
```

For a custom ONNX/JSON pair, use `--add-policy <NAME>=<ONNX_PATH>`. Runtime
switching is available in MuJoCo; the real-robot entry point remains
single-policy and requires explicit control enablement. Without `--controller`,
MuJoCo uses the fixed `--command` value.

### Real-Robot Deployment

Start with read-only state and LIO inspection:

```bash
uv run --frozen --extra robot adapt-real --net eno1 --lidar-frame raw --state-only
```

Alternatively, run the Python script directly:

```bash
uv run --extra robot python deployment/deploy_real_lio_mjlab.py \
  --net eno1 \
  --lidar-frame raw \
  --state-only
```

To load an ONNX/JSON pair directly without using `manifest.json`:

```bash
uv run --frozen --extra robot adapt-real \
  --net eno1 \
  --lidar-frame raw \
  --checkpoint <ONNX_PATH> \
  --state-only
```

```bash
uv run --extra robot python deployment/deploy_real_lio_mjlab.py \
  --net eno1 \
  --lidar-frame raw \
  --checkpoint <ONNX_PATH> \
  --state-only
```

Replace `eno1` with the robot network interface and select `raw` or
`compensated` according to the Livox driver configuration. Low-level control
must be enabled explicitly with `--enable-control` after hardware-specific
validation.

## Isaac Sim 5.1: MYS FT-R

Frozen **Booster T1 foothold-tracking** policy with GT state (separate from ADAPT G1).
Use an existing Isaac Sim 5.1 / Python 3.11 environment with NumPy and SciPy; keep it separate from ADAPT's Python 3.10 environment.

```bash
conda activate env_isaaclab
bash isaac/mys_ftr/view.sh
# Headless forward walking:
bash isaac/mys_ftr/view.sh --headless --mode walk --seconds 25
```

Defaults: stand → forward → turn → stop over 25 s; red/blue markers are left/right foot targets.
The actor and robot assets are bundled; USD is generated on first launch and logs go to `isaac/mys_ftr/runs/`.
Set `ISAAC_PYTHON=/path/to/isaac/python` to select a different interpreter.
Physics, GT state logging and PD torque updates run at **1 kHz**; the frozen policy stays at **50 Hz** (20 physics steps per action).
`trace.npz` contains 1 ms state samples; `is_policy_step` selects fresh policy evaluations, and observation/action/targets are held between them. These are GT states; the policy continues to use GT when sensor recording is enabled.

Optional native sensors (IMU **1 kHz**, RTX radar **10 Hz**):

```bash
bash isaac/mys_ftr/view.sh --imu --radar
# Record without a window:
bash isaac/mys_ftr/view.sh --headless --imu --radar
```

Logs: `imu.npz` (specific force + gyro), `radar.npz` (XYZ + native Doppler, ragged frames via `offsets`), and `sensors.json` (extrinsics, units and timing).
RTX/GUI rendering stays at 50 Hz to reduce frame latency; radar output is limited to the 10 Hz simulation-time grid. Physics/IMU remain 1 kHz and the policy 50 Hz.
The generic radar sees added static steel posts outside the walking lane; `--no-radar-reflectors` keeps the original scene.
Use `--radar-translation X Y Z` / `--radar-pitch-deg DEG` to change the torso mount.
These are simulated sensor streams with no hardware noise calibration; GaRLILEO and ROS publishing are not connected.

## Citation

```bibtex
@misc{lyu2026adapt,
  title={ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion},
  author={Bofan Lyu and Jindou Jia and Kuangji Zuo and Yanshuo Lu and Shijia Han and Gen Li and Boyu Ma and Jingliang Li and Geng Li and Jianfei Yang},
  year={2026},
  eprint={2606.16542},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
