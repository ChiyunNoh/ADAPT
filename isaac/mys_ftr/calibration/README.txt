Local empirical radar calibration (not versioned)

The ti_awr1843aop surface model requires real_floor_model.npz from the existing
0916 run_02 calibration. Keep the file here or pass --radar-floor-model PATH.

From the repository root:
  mkdir -p isaac/mys_ftr/calibration
  cp /path/to/real_floor_model.npz isaac/mys_ftr/calibration/
  bash isaac/mys_ftr/view.sh --imu --radar

Known SHA256:
  054f639eba6910612dbae455904341647c1855f530939e0f90f65b50830ff8ac
Required NPZ keys: d (horizontal distance), az (degrees), intensity, counts,
height (reference radar height in metres). Loaded with allow_pickle=False.
Original model: 11,028 floor samples, 736 frame-count samples, height 1.23 m.
This is a measured sampling distribution, not a terrain map or robot trajectory.

The file is intentionally ignored by Git; do not commit source recordings.
For native-only operation without the calibration: add --no-synthetic-floor.
