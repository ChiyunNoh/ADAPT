"""Figure S3-style signed lateral drift versus forward displacement.

The common 0–6 m range retains all 30 trials in every condition. Interpolate
each trial's FIRST forward crossing; never sort a reversing path by x.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/evaluation/velocity_sensitivity_paper_metrics_2026-10-21'


def first_crossing_curve(xy, distances):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected a finite time-ordered xy path')
    out = np.empty(len(distances), dtype=np.float64)
    for i, distance in enumerate(distances):
        if distance == 0:
            out[i] = xy[0, 1]
            continue
        hits = np.flatnonzero((xy[:-1, 0] < distance) & (xy[1:, 0] >= distance))
        if not len(hits):
            raise ValueError(f'Trial never reaches {distance:g} m; do not extrapolate')
        j = int(hits[0])
        weight = (distance - xy[j, 0]) / (xy[j+1, 0] - xy[j, 0])
        out[i] = xy[j, 1] + weight * (xy[j+1, 1] - xy[j, 1])
    return out


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = json.loads((OUT / 'metadata.json').read_text())
    noises, delays = meta['noise_half_widths_mps'], meta['delays_ms']
    distances = np.linspace(0, 6, 121)
    conditions = [(a, d) for a in noises for d in delays]
    paths = {c: [] for c in conditions}
    hashes = {}
    for source in meta['source_runs']:
        source = Path(source)
        cfg = json.loads((source / 'metadata.json').read_text())
        n = cfg['trials_per_condition']
        src = source / 'trajectories.npz'
        hashes[str(src)] = hashlib.sha256(src.read_bytes()).hexdigest()
        with np.load(src) as archive:
            firsts = np.flatnonzero(np.isclose(archive['time'], meta['evaluation_window_s'][0], rtol=0, atol=1e-9))
            if len(firsts) != 1 or not archive['alive'].all():
                raise ValueError('Requires exact starting sample and complete rollouts')
            first = int(firsts[0])
            xy = archive['position'][first:, :, :2].astype(np.float64)
            xy -= xy[:1].copy()
            for ci, condition in enumerate(conditions):
                for trial in range(n):
                    paths[condition].append(first_crossing_curve(xy[:, ci*n+trial], distances))
    curves = {c: np.stack(v) for c,v in paths.items()}
    assert all(v.shape == (30, len(distances)) for v in curves.values())
    ref = curves[0, 0]
    reference_mean, reference_sd = ref.mean(axis=0), ref.std(axis=0, ddof=1)
    # Fixed shared y range covers every condition's mean +/- SD without clipping.
    extent = max(float(np.max(np.abs(v.mean(axis=0)) + v.std(axis=0, ddof=1))) for v in curves.values())
    ylimit = np.ceil(extent * 1.08 * 2) / 2
    colors = ('#4379b6', '#e65b62')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'svg.fonttype': 'none', 'axes.spines.top': True, 'axes.spines.right': True})

    def plot(selected_noises, filename):
        fig, axes = plt.subplots(len(selected_noises), len(delays),
                                 figsize=(16, 3.0 * len(selected_noises) + 1.2),
                                 sharex=True, sharey=True, squeeze=False)
        for row, amplitude in enumerate(selected_noises):
            for col, delay in enumerate(delays):
                ax = axes[row, col]
                group = curves[amplitude, delay]
                mean, sd = group.mean(axis=0), group.std(axis=0, ddof=1)
                ax.fill_between(distances, reference_mean-reference_sd, reference_mean+reference_sd,
                                color=colors[0], alpha=.15, linewidth=0)
                ax.plot(distances, reference_mean, color=colors[0], linewidth=1.9, linestyle='--')
                ax.fill_between(distances, mean-sd, mean+sd, color=colors[1], alpha=.18, linewidth=0)
                ax.plot(distances, mean, color=colors[1], linewidth=2)
                ax.axhline(0, color='#555555', linewidth=.7, linestyle=':', zorder=0)
                ax.set(xlim=(0, 6), ylim=(-ylimit, ylimit), xticks=[0,2,4,6])
                ax.grid(color='#d9dfe4', linewidth=.6, alpha=.8)
                ax.set_axisbelow(True)
                ax.tick_params(labelbottom=True)
                if row == 0:
                    ax.set_title(f'Delay = {delay} ms', pad=9)
                if col == 0:
                    ax.set_ylabel(f'Noise ±{amplitude:g} m/s\nLateral drift (m)')
                if row == len(selected_noises)-1:
                    ax.set_xlabel('Forward distance (m)')
        fig.suptitle('ADAPT | Signed lateral drift versus forward distance', fontsize=17, y=.985)
        handles = [Line2D([0],[0],color=colors[0],lw=2,ls='--'),Line2D([0],[0],color=colors[1],lw=2)]
        fig.legend(handles, ['Reference: noise 0, delay 0', 'Test condition: row noise, column delay'],
                   loc='upper center', bbox_to_anchor=(.5,.947 if len(selected_noises)==2 else .965),
                   ncol=2, frameon=False, fontsize=11)
        fig.text(.5,.02,'Command 0.8 m/s | actor + observer shared input | 30 trials in every curve | mean ± 1 SD\n'
                 'First crossing of each distance from t = 2.02 s; linear interpolation of recorded paths; all trials reach 6 m',
                 ha='center', va='bottom', fontsize=10, color='#414a54')
        fig.subplots_adjust(left=.085, right=.985, bottom=.15 if len(selected_noises)==2 else .08,
                            top=.84 if len(selected_noises)==2 else .92, wspace=.09, hspace=.18)
        fig.savefig(OUT / f'{filename}.png', dpi=220, facecolor='white')
        fig.savefig(OUT / f'{filename}.svg', facecolor='white')
        plt.close(fig)

    plot([.05, .5], 's3_style_trajectories')
    plot(noises, 's3_style_trajectories_all_conditions')
    np.savez_compressed(OUT / 's3_style_curves.npz', distance_m=distances,
                        signed_lateral_m=np.stack([curves[c] for c in conditions]),
                        noise_mps=[c[0] for c in conditions], delay_ms=[c[1] for c in conditions])
    analysis = {
        'reference_figure': 'ADAPT Figure S3, visual layout only',
        'reference_condition': {'noise_mps':0,'delay_ms':0},
        'reference_is_separate_policy': False,
        'main_rows_noise_mps': [.05,.5], 'full_rows_noise_mps': noises, 'columns_delay_ms':delays,
        'command_mps':.8, 'evaluation_start_s':2.02, 'end_of_available_data_s':30,
        'forward_distance_range_m':[0,6], 'trials_per_condition':30,
        'x_definition':'world x displacement relative to position at 2.02 s; not arc length',
        'y_definition':'signed world y displacement relative to position at 2.02 s; not absolute endpoint metric',
        'alignment':'first forward crossing of each x grid point, piecewise linear interpolation of stored ~5 Hz trajectory points; no extrapolation, sorting or trial exclusion',
        'band':'sample SD across 30 trials; not CI',
        'caution':'Signed averages can cancel left and right drift. Spatially aligned trajectories differ from fixed-time endpoint metrics. Compare spread and the endpoint metrics too.',
        'source_sha256':hashes,
    }
    (OUT / 's3_style_metadata.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(f'Plotted all {len(conditions)} conditions, 30/30 trials throughout 0–6 m; y limits ±{ylimit:g} m')
    print(OUT / 's3_style_trajectories.png')


if __name__ == '__main__':
    main()
