"""
Milestone 3 Analysis + Figures
Combines:
  A) New metrics from existing dreamer.json training data (bootstrap CI)
  B) New experimental results from run_experiments.py (random policy, action scale, manipulator)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
RNG  = np.random.default_rng(0)

with open(BASE / 'scores/dreamer.json') as f:
    raw = json.load(f)
with open(BASE / 'scores/baselines.json') as f:
    baselines = json.load(f)
with open(BASE / 'scores/experiment_results.json') as f:
    exp = json.load(f)

task_data = defaultdict(list)
for r in raw:
    task_data[r['task']].append((r['seed'], np.array(r['xs']), np.array(r['ys'])))
ALL_TASKS = sorted(task_data.keys())

def pretty(task):
    return task.replace('dmc_', '').replace('_', ' ').title()

def final_scores(task):
    return np.array([ys[-1] for _, xs, ys in task_data[task]])

def bootstrap_ci(values, n_boot=10000, ci=0.95):
    values = np.asarray(values)
    boots = RNG.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100*(1-ci)/2)
    hi = np.percentile(boots, 100*(1+ci)/2)
    return values.mean(), lo, hi

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: 3-way comparison — Random Policy vs Dreamer (1M) vs D4PG (100M)
# ══════════════════════════════════════════════════════════════════════════════
COMPARE_TASKS = [
    'dmc_cartpole_swingup', 'dmc_cartpole_swingup_sparse',
    'dmc_acrobot_swingup',  'dmc_cheetah_run',
    'dmc_walker_walk',      'dmc_hopper_hop',
    'dmc_reacher_easy',     'dmc_cup_catch',
]
random_bl     = exp['random_baseline']
dreamer_means = [final_scores(t).mean() for t in COMPARE_TASKS]
random_means  = [random_bl.get(t, {}).get('return_mean', float('nan')) for t in COMPARE_TASKS]
d4pg_vals     = [baselines.get(t, {}).get('d4pg_100m', float('nan')) for t in COMPARE_TASKS]
task_labels   = [pretty(t) for t in COMPARE_TASKS]

x = np.arange(len(COMPARE_TASKS)); w = 0.25
fig1, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(x-w, random_means,  w, label='Random Policy (ours)',         color='#888888', alpha=0.8)
ax1.bar(x,   dreamer_means, w, label='Dreamer 1M steps (paper)',      color='#377eb8', alpha=0.85)
ax1.bar(x+w, d4pg_vals,     w, label='D4PG 100M steps (paper)',       color='#e41a1c', alpha=0.75)
ax1.set_xticks(x)
ax1.set_xticklabels(task_labels, rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('Episode Return', fontsize=9)
ax1.set_title('Three-Way Comparison: Random Policy (ours) vs Dreamer (1M) vs D4PG (100M)',
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig(BASE / 'Report/fig1_three_way.png', dpi=150, bbox_inches='tight')
print('Saved fig1_three_way.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Action-scale failure mode
# ══════════════════════════════════════════════════════════════════════════════
scale_data = exp['action_scale']
scales = [1.0, 0.5, 0.3, 0.1]; scale_strs = [str(s) for s in scales]

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (task_key, title) in zip(axes2, [
    ('cartpole_swingup', 'Cartpole Swingup'),
    ('acrobot_swingup',  'Acrobot Swingup'),
]):
    td    = scale_data.get(task_key, {})
    means = [td.get(s, {}).get('return_mean', 0) for s in scale_strs]
    stds  = [td.get(s, {}).get('return_std',  0) for s in scale_strs]
    ax.bar(scale_strs, means, yerr=stds, capsize=4, color='#4daf4a', alpha=0.8, width=0.5)
    dmc_key = f'dmc_{task_key}'
    dr_mean = final_scores(dmc_key).mean()
    d4pg_v  = baselines.get(dmc_key, {}).get('d4pg_100m', None)
    ax.axhline(dr_mean, color='#377eb8', ls='--', lw=1.5, label=f'Dreamer 1M ({dr_mean:.0f})')
    if d4pg_v:
        ax.axhline(d4pg_v, color='#e41a1c', ls=':', lw=1.5, label=f'D4PG 100M ({d4pg_v:.0f})')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Action Scale Factor', fontsize=8)
    ax.set_ylabel('Episode Return (random policy, 30 ep.)', fontsize=7)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
fig2.suptitle('Action-Scale Failure Mode: Episode Return vs. Control Authority\n'
              '(lower scale = less torque = harder to accumulate swing energy)',
              fontsize=9, fontweight='bold')
fig2.tight_layout()
fig2.savefig(BASE / 'Report/fig2_action_scale.png', dpi=150, bbox_inches='tight')
print('Saved fig2_action_scale.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Manipulator tasks + energy metric
# ══════════════════════════════════════════════════════════════════════════════
manip = exp['manipulator']
patch_gray   = plt.Rectangle((0,0),1,1, color='#888888', alpha=0.85)
patch_purple = plt.Rectangle((0,0),1,1, color='#984ea3', alpha=0.85)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(9, 3.8))

ref_tasks   = ['dmc_cartpole_swingup', 'dmc_walker_walk', 'dmc_cheetah_run']
ref_labels  = ['Cartpole\nSwingup', 'Walker\nWalk', 'Cheetah\nRun']
ref_returns = [random_bl.get(t, {}).get('return_mean', 0) for t in ref_tasks]
ref_stds    = [random_bl.get(t, {}).get('return_std',  0) for t in ref_tasks]

manip_keys    = ['bring_ball', 'bring_peg', 'insert_ball']
manip_labels  = ['Bring\nBall', 'Bring\nPeg', 'Insert\nBall']
manip_returns = [manip.get(k, {}).get('return_mean', 0) for k in manip_keys]
manip_stds    = [manip.get(k, {}).get('return_std',  0) for k in manip_keys]

x_ref = np.arange(len(ref_labels))
x_man = np.arange(len(manip_labels)) + len(ref_labels) + 0.6

for xi, ret, std in zip(x_ref, ref_returns, ref_stds):
    ax3a.bar(xi, ret, yerr=std, color='#888888', alpha=0.85, capsize=3, width=0.6)
for xi, ret, std in zip(x_man, manip_returns, manip_stds):
    ax3a.bar(xi, ret, yerr=std, color='#984ea3', alpha=0.85, capsize=3, width=0.6)
ax3a.set_xticks(list(x_ref)+list(x_man))
ax3a.set_xticklabels(ref_labels + manip_labels, fontsize=8)
ax3a.set_ylabel('Episode Return (random policy, 30 ep.)', fontsize=8)
ax3a.set_title('Object Interaction Tasks vs. DMC Tasks\n(Random Policy Baseline, ours)',
               fontsize=9, fontweight='bold')
ax3a.legend([patch_gray, patch_purple], ['Paper tasks', 'Manipulator (new)'], fontsize=7)
ax3a.grid(axis='y', alpha=0.3)

energy_vals = (
    [random_bl.get(t, {}).get('energy_mean', 0)/1000 for t in ref_tasks] +
    [manip.get(k, {}).get('energy_per_step_mean', 0) for k in manip_keys]
)
energy_colors = ['#888888']*3 + ['#984ea3']*3
x_energy = list(x_ref) + list(x_man)
ax3b.bar(x_energy, energy_vals, color=energy_colors, alpha=0.85, width=0.6)
ax3b.set_xticks(x_energy)
ax3b.set_xticklabels(ref_labels + manip_labels, fontsize=8)
ax3b.set_ylabel('Mean |action| per step', fontsize=8)
ax3b.set_title('Energy Metric: Action Effort per Step\n(unreported in original paper)',
               fontsize=9, fontweight='bold')
ax3b.legend([patch_gray, patch_purple], ['Paper tasks', 'Manipulator (new)'], fontsize=7)
ax3b.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig(BASE / 'Report/fig3_manipulator_energy.png', dpi=150, bbox_inches='tight')
print('Saved fig3_manipulator_energy.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Bootstrap CI (all 20 tasks)
# ══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(ALL_TASKS))
d_means, d_lo, d_hi, d4pg_pt = [], [], [], []
for task in ALL_TASKS:
    m, lo, hi = bootstrap_ci(final_scores(task))
    d_means.append(m); d_lo.append(m-lo); d_hi.append(hi-m)
    d4pg_pt.append(baselines.get(task, {}).get('d4pg_100m', np.nan))
d_means=np.array(d_means); d_lo=np.array(d_lo); d_hi=np.array(d_hi); d4pg_pt=np.array(d4pg_pt)
ax4.barh(y_pos, d_means, xerr=[d_lo, d_hi], color='#377eb8', alpha=0.75,
         capsize=3, height=0.55, label='Dreamer 1M (mean ± 95% CI, bootstrap, 5 seeds)')
valid = ~np.isnan(d4pg_pt)
ax4.scatter(d4pg_pt[valid], y_pos[valid], color='#e41a1c', zorder=5,
            marker='D', s=40, label='D4PG 100M steps (reported)')
ax4.set_yticks(y_pos)
ax4.set_yticklabels([pretty(t) for t in ALL_TASKS], fontsize=8)
ax4.set_xlabel('Final Episode Return (last checkpoint)', fontsize=9)
ax4.set_title('Bootstrap 95% CI on Final Performance: Dreamer vs. D4PG',
              fontsize=10, fontweight='bold')
ax4.legend(fontsize=8, loc='lower right'); ax4.grid(axis='x', alpha=0.3)
fig4.tight_layout()
fig4.savefig(BASE / 'Report/fig4_bootstrap_ci.png', dpi=150, bbox_inches='tight')
print('Saved fig4_bootstrap_ci.png')

# ── Key numbers ───────────────────────────────────────────────────────────────
print('\n=== Random vs Dreamer(1M) vs D4PG(100M) ===')
for t, rm, dm, dv in zip(COMPARE_TASKS, random_means, dreamer_means, d4pg_vals):
    print(f'  {pretty(t):<30} random={rm:7.1f}  dreamer={dm:7.1f}  d4pg={dv:7.1f}')
print('\n=== Action Scale (cartpole_swingup) ===')
for s in scale_strs:
    d = scale_data['cartpole_swingup'].get(s, {})
    print(f'  scale={s}: {d.get("return_mean",0):.1f} ± {d.get("return_std",0):.1f}')
print('\n=== Manipulator (random, 30 ep) ===')
for k in manip_keys:
    d = manip.get(k, {})
    print(f'  {k}: return={d.get("return_mean",0):.2f}  energy/step={d.get("energy_per_step_mean",0):.2f}')
print('\n=== Bootstrap CI (high-variance tasks) ===')
for task in ['dmc_cartpole_balance_sparse', 'dmc_hopper_hop', 'dmc_cartpole_balance']:
    m, lo, hi = bootstrap_ci(final_scores(task))
    print(f'  {pretty(task)}: {m:.1f} [{lo:.1f}, {hi:.1f}]')
