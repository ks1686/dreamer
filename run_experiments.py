"""
Milestone 3 New Experiments
Runs real environment rollouts to collect missing metrics and test failure modes.

Experiments:
  1. Random policy baseline across 8 key DMC tasks
     - Collects: episode return, duration (steps), energy (sum|action|)
     - Provides a genuine comparison point alongside Dreamer and D4PG
  2. Action-scale failure mode on cartpole_swingup and acrobot_swingup
     - Full scale [-1,1] vs reduced scale [-0.3, 0.3] and [-0.1, 0.1]
     - Demonstrates that low control authority prevents energy accumulation
  3. Manipulator bring_ball (object interaction scene missing from paper)
     - Random policy baseline to establish difficulty
"""

import json
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from dm_control import suite
from pathlib import Path

RNG = np.random.default_rng(42)
BASE = Path(__file__).parent
OUT = BASE / 'scores'

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_random_episode(env, action_scale=1.0, max_steps=1000):
    """Run one episode with uniformly random actions scaled by action_scale."""
    spec = env.action_spec()
    timestep = env.reset()
    total_reward = 0.0
    total_energy = 0.0
    steps = 0
    while not timestep.last() and steps < max_steps:
        action = RNG.uniform(spec.minimum, spec.maximum) * action_scale
        timestep = env.step(action)
        total_reward += float(timestep.reward or 0.0)
        total_energy += float(np.sum(np.abs(action)))
        steps += 1
    return {
        'return': total_reward,
        'duration': steps,
        'energy': total_energy,
        'energy_per_step': total_energy / max(steps, 1),
    }


def run_episodes(domain, task, n=30, action_scale=1.0, max_steps=1000):
    print(f'  Running {domain}/{task} (scale={action_scale}, n={n})...')
    try:
        env = suite.load(domain, task)
    except Exception as e:
        print(f'    SKIP: {e}')
        return None
    results = [run_random_episode(env, action_scale, max_steps) for _ in range(n)]
    stats = {
        'return_mean':   np.mean([r['return'] for r in results]),
        'return_std':    np.std( [r['return'] for r in results]),
        'return_min':    np.min( [r['return'] for r in results]),
        'return_max':    np.max( [r['return'] for r in results]),
        'duration_mean': np.mean([r['duration'] for r in results]),
        'energy_mean':   np.mean([r['energy'] for r in results]),
        'energy_per_step_mean': np.mean([r['energy_per_step'] for r in results]),
        'raw_returns':   [r['return'] for r in results],
        'raw_durations': [r['duration'] for r in results],
        'raw_energies':  [r['energy'] for r in results],
    }
    print(f'    return={stats["return_mean"]:.1f}±{stats["return_std"]:.1f}  '
          f'dur={stats["duration_mean"]:.0f}  energy={stats["energy_mean"]:.1f}')
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Random policy baseline on 8 key tasks
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Experiment 1: Random Policy Baseline ===')

BASELINE_TASKS = [
    ('cartpole', 'swingup'),
    ('cartpole', 'swingup_sparse'),
    ('acrobot',  'swingup'),
    ('cheetah',  'run'),
    ('walker',   'walk'),
    ('hopper',   'hop'),
    ('reacher',  'easy'),
    ('cup',      'catch'),   # ball_in_cup in dm_control
]

random_baseline = {}
for domain, task in BASELINE_TASKS:
    key = f'dmc_{domain}_{task}'
    result = run_episodes(domain, task, n=30)
    if result:
        random_baseline[key] = result

# ── ball_in_cup is 'ball_in_cup', 'catch' in dm_control ──────────────────────
# Fix the cup/catch naming
result = run_episodes('ball_in_cup', 'catch', n=30)
if result:
    random_baseline['dmc_cup_catch'] = result
# Remove the wrong-named key if it was inserted
random_baseline.pop('dmc_cup_catch_wrong', None)

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Action-scale failure mode
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Experiment 2: Action-Scale Failure Mode ===')

ACTION_TASKS = [
    ('cartpole', 'swingup'),
    ('acrobot',  'swingup'),
]
SCALES = [1.0, 0.5, 0.3, 0.1]

action_scale_results = {}
for domain, task in ACTION_TASKS:
    key = f'{domain}_{task}'
    action_scale_results[key] = {}
    for scale in SCALES:
        result = run_episodes(domain, task, n=30, action_scale=scale)
        if result:
            action_scale_results[key][str(scale)] = result

# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Manipulator task (object interaction — missing from paper)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Experiment 3: Manipulator Object Interaction ===')

manipulator_results = {}
for task_name in ['bring_ball', 'bring_peg', 'insert_ball']:
    result = run_episodes('manipulator', task_name, n=30, max_steps=1000)
    if result:
        manipulator_results[task_name] = result

# ══════════════════════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════════════════════

def convert(obj):
    if isinstance(obj, np.float64): return float(obj)
    if isinstance(obj, np.int64): return int(obj)
    if isinstance(obj, list): return [convert(x) for x in obj]
    if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
    return obj

all_results = {
    'random_baseline': convert(random_baseline),
    'action_scale': convert(action_scale_results),
    'manipulator': convert(manipulator_results),
}

out_path = OUT / 'experiment_results.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nSaved to {out_path}')

# ── Print summary ──────────────────────────────────────────────────────────────
print('\n=== Summary: Random Baseline vs Dreamer ===')
with open(BASE / 'scores/dreamer.json') as f:
    dreamer_data = json.load(f)
from collections import defaultdict
dreamer_task = defaultdict(list)
for r in dreamer_data:
    dreamer_task[r['task']].append(r['ys'][-1])

with open(BASE / 'scores/baselines.json') as f:
    d4pg_data = json.load(f)

print(f'{"Task":<35} {"Random":>10} {"Dreamer(1M)":>12} {"D4PG(100M)":>12}')
print('-' * 73)
for key, res in random_baseline.items():
    dreamer_mean = np.mean(dreamer_task.get(key, [float('nan')]))
    d4pg = d4pg_data.get(key, {}).get('d4pg_100m', float('nan'))
    print(f'{key:<35} {res["return_mean"]:>10.1f} {dreamer_mean:>12.1f} {d4pg:>12.1f}')

print('\n=== Action Scale Results ===')
for task_key, scale_dict in action_scale_results.items():
    print(f'\n{task_key}:')
    print(f'  {"Scale":>8} {"Return mean":>14} {"Return std":>12} {"Duration":>10}')
    for scale, res in scale_dict.items():
        print(f'  {scale:>8} {res["return_mean"]:>14.1f} {res["return_std"]:>12.1f} {res["duration_mean"]:>10.0f}')

print('\n=== Manipulator Results (Random Policy) ===')
for task_name, res in manipulator_results.items():
    print(f'  {task_name}: return={res["return_mean"]:.3f}±{res["return_std"]:.3f}  '
          f'energy={res["energy_mean"]:.1f}')
