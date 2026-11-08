"""Regenerate figures and CSV tables from saved browser measurements."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent / 'results'
raw = json.loads((root / 'performance.json').read_text())
records = raw['records']
strategies = ['baseline', 'cached', 'cooperative', 'striped']
labels = ['Baseline', 'Cached', '64 lanes / triangle', '8 lanes / triangle']
colors = ['#46627f', '#39a69b', '#dc9646', '#9a72b2']
cases = list(dict.fromkeys((r['name'], r['grid'], r['image']) for r in records))
def record(case, strategy):
    return next(r for r in records if (r['name'], r['grid'], r['image']) == case and r['strategy'] == strategy)

with (root / 'summary.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['scene', 'lens_vertices_per_side', 'receiver_pixels_per_side', 'strategy', 'gpu_step_median_ms', 'gpu_step_p25_ms', 'gpu_step_p75_ms', 'speedup_vs_baseline'])
    for r in records:
        case = (r['name'], r['grid'], r['image'])
        stats = r['iterationSummary']['gpuMs']
        baseline = record(case, 'baseline')['iterationSummary']['gpuMs']['median']
        writer.writerow([*case, r['strategy'], stats['median'], stats['p25'], stats['p75'], baseline / stats['median']])
with (root / 'samples.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['scene', 'grid', 'image', 'strategy', 'mode', 'round', 'sample', 'metric', 'milliseconds'])
    for r in records:
        for mode in ['phases', 'iteration']:
            for sample in r[mode]:
                for metric, value in sample.items():
                    if metric not in ['round', 'sample']:
                        writer.writerow([r['name'], r['grid'], r['image'], r['strategy'], mode, sample['round'], sample['sample'], metric, value])

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
y = np.arange(len(cases))
for j, strategy in enumerate(strategies):
    stats = [record(c, strategy)['iterationSummary']['gpuMs'] for c in cases]
    values = np.array([s['median'] for s in stats])
    lower = values - np.array([s['p25'] for s in stats])
    upper = np.array([s['p75'] for s in stats]) - values
    ax.barh(y + (j - 1.5) * .18, values, height=.16, color=colors[j], label=labels[j], xerr=[lower, upper], capsize=2)
ax.set_yticks(y, [f'{c[0].title()} | lens {c[1]} x {c[1]}\nreceiver {c[2]} x {c[2]}' for c in cases])
ax.invert_yaxis()
ax.set_xlabel('GPU time per fixed-state A+B step, including final optical preview (ms)')
ax.set_title('Cooperative pixel work helps large footprints, but can slow small ones\nApple M1 Pro | 30 samples per variant | median and interquartile range', loc='left')
ax.legend(loc='lower right', frameon=False)
fig.savefig(root / 'iteration-time.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
for ax, case in zip(axes, [cases[0], cases[3]]):
    bottom = np.zeros(len(strategies))
    for phase, color in [('project', '#a8c8dd'), ('forward', '#46627f'), ('backward', '#39a69b'), ('gather', '#dc9646')]:
        values = np.array([record(case, s)['phaseSummary'].get(phase, {}).get('median', 0) for s in strategies])
        ax.bar(np.arange(len(strategies)), values, bottom=bottom, label=phase, color=color)
        bottom += values
    ax.set_xticks(np.arange(len(strategies)), ['Baseline', 'Cached', '64 lanes', '8 lanes'])
    ax.set_ylabel('Sum of isolated kernel medians (ms)')
    ax.set_title(f'{case[0].title()}: lens {case[1]} / receiver {case[2]}')
axes[0].legend(frameon=False)
fig.suptitle('Forward and backward dominate; gradient gathering is small\nIsolated submissions for diagnosis; these sums are not full-iteration timings', fontsize=12)
fig.savefig(root / 'kernel-time.png', dpi=180)
plt.close(fig)

checks = json.loads((root / 'extended.json').read_text())['correctness']
fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
for j, s in enumerate(strategies):
    points = [r for r in checks if r['strategy'] == s and r['gradientError'] is not None]
    ax.scatter(np.arange(len(points)) + (j - 1.5) * .07, [r['gradientError'] for r in points], label=labels[j], color=colors[j], s=35)
ax.axhline(.005, color='#a73f40', linestyle='--', label='Acceptance threshold')
ax.set_xticks(np.arange(len(points)), [r['name'] for r in points])
ax.set_yscale('log')
ax.set_ylim(3e-7, .02)
ax.set_ylabel('Relative L2 gradient error vs CPU finite differences')
ax.set_title('Analytic gradients remain accurate after parallelization', loc='left')
ax.legend(frameon=False, fontsize=8)
fig.savefig(root / 'gradient-check.png', dpi=180)
plt.close(fig)
print('Wrote summary.csv, samples.csv, and three figures.')
