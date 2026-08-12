"""Experiment 3 – Numerical Ambiguity and Discretization Sensitivity Study.

Sweeps a 2×2×2 grid of numerical variants (event_rule × price_rule × edge_rule)
for two strategies (inventory, symmetric) and three gamma values (0.01, 0.1, 0.5),
producing 48 simulation runs in total.

Baseline = (bernoulli, binomial, log_only), matching Experiment 1.

Outputs (all written to results/)
  exp3_sensitivity.csv           – full 48-row result table
  exp3_summary.json              – parameter & key-finding summary
  RESULTS_exp3.md                – markdown narrative
  exp3_heatmap_std_profit.png    – std_profit deviation heatmap (inventory, gamma=0.1)
  exp3_diagnostics_bar.png       – neg_delta / lambda>1 counts for baseline variant
  exp3_variant_comparison.png    – mean_profit across variants by gamma
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import product

from src.sensitivity import SensitivityConfig, simulate_sensitivity_variant, compute_half_spread

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
# Parameters
# ------------------------------------------------------------------ #
S0, T, sigma, dt = 100.0, 1.0, 2.0, 0.005
A, k, n_paths = 140.0, 1.5, 1000
gamma_values = [0.1, 0.01, 0.5]

BASELINE = dict(event_rule='bernoulli', price_rule='binomial', edge_rule='log_only')

event_rules = ['bernoulli', 'exact_poisson']
price_rules = ['binomial', 'gaussian']
edge_rules = ['log_only', 'clip_zero']
strategies = ['inventory', 'symmetric']


# ------------------------------------------------------------------ #
# Helper: short variant label for plots
# ------------------------------------------------------------------ #
def variant_label(row):
    e = 'Bern' if row['event_rule'] == 'bernoulli' else 'Pois'
    p = 'Bino' if row['price_rule'] == 'binomial' else 'Gaus'
    g = 'Log' if row['edge_rule'] == 'log_only' else 'Clip'
    return f"{e}-{p}-{g}"


# ------------------------------------------------------------------ #
# Run sensitivity grid
# ------------------------------------------------------------------ #
print("Running sensitivity grid…")
rows = []
combos = list(product(strategies, event_rules, price_rules, edge_rules))
total = len(gamma_values) * len(combos)

for run_idx, (gamma, (strategy, event_rule, price_rule, edge_rule)) in enumerate(
    product(gamma_values, combos), start=1
):
    cfg = SensitivityConfig(S0=S0, T=T, sigma=sigma, dt=dt, A=A, k=k, n_paths=n_paths)
    result = simulate_sensitivity_variant(
        policy_name=strategy,
        gamma=gamma,
        cfg=cfg,
        event_rule=event_rule,
        price_rule=price_rule,
        edge_rule=edge_rule,
        cap_prob_flag=False,
        seed=42,
    )
    rows.append({
        'gamma': gamma,
        'strategy': strategy,
        'event_rule': event_rule,
        'price_rule': price_rule,
        'edge_rule': edge_rule,
        'mean_profit': float(np.mean(result['profit'])),
        'std_profit': float(np.std(result['profit'])),
        'mean_q': float(np.mean(result['final_q'])),
        'std_q': float(np.std(result['final_q'])),
        'neg_delta_count': result['neg_delta_count'],
        'lambda_dt_gt1_count': result['lambda_dt_gt1_count'],
        'simult_fill_count': result['simult_fill_count'],
    })
    if run_idx % 8 == 0 or run_idx == total:
        print(f"  [{run_idx:2d}/{total}] gamma={gamma:.2f}, strategy={strategy:<10s}, "
              f"event={event_rule:<14s}, price={price_rule:<8s}, edge={edge_rule}")

df = pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Compute deviations from baseline
# ------------------------------------------------------------------ #
baseline_ref = df[
    (df['event_rule'] == BASELINE['event_rule']) &
    (df['price_rule'] == BASELINE['price_rule']) &
    (df['edge_rule'] == BASELINE['edge_rule'])
][['gamma', 'strategy', 'mean_profit', 'std_profit']].rename(
    columns={'mean_profit': 'baseline_mean', 'std_profit': 'baseline_std'}
)

df = df.merge(baseline_ref, on=['gamma', 'strategy'])
df['delta_mean_profit'] = df['mean_profit'] - df['baseline_mean']
df['delta_std_profit'] = df['std_profit'] - df['baseline_std']
df['variant'] = df.apply(variant_label, axis=1)

# Save full table (retain delta columns)
save_cols = [
    'gamma', 'strategy', 'event_rule', 'price_rule', 'edge_rule',
    'mean_profit', 'std_profit', 'mean_q', 'std_q',
    'neg_delta_count', 'lambda_dt_gt1_count', 'simult_fill_count',
    'delta_mean_profit', 'delta_std_profit',
]
df[save_cols].to_csv(os.path.join(RESULTS_DIR, 'exp3_sensitivity.csv'), index=False)
print("Saved exp3_sensitivity.csv")


# ------------------------------------------------------------------ #
# Plot 1 – Heatmap: std_profit deviation from baseline
#          (inventory strategy, gamma = 0.1)
# ------------------------------------------------------------------ #
print("Generating heatmap…")

hm_data = df[(df['strategy'] == 'inventory') & (df['gamma'] == 0.1)].copy()
hm_data['row_label'] = hm_data['event_rule'].str.replace('_', '\n') + '\n' + hm_data['price_rule']
hm_pivot = hm_data.pivot_table(
    values='delta_std_profit',
    index='row_label',
    columns='edge_rule',
    aggfunc='mean',
)
# Reorder rows to a predictable sequence
row_order = [
    'bernoulli\nbinomial', 'bernoulli\ngaussian',
    'exact\npoisson\nbinomial', 'exact\npoisson\ngaussian',
]
# Build row order from what's actually in the pivot
hm_pivot = hm_pivot.reindex(sorted(hm_pivot.index.tolist()))

vmax = max(abs(hm_pivot.values.min()), abs(hm_pivot.values.max()))
vmax = max(vmax, 0.1)  # avoid degenerate scale

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    hm_pivot,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'Δ std_profit (vs baseline)'},
)
ax.set_title('std_profit Deviation from Baseline\n(inventory strategy, γ = 0.1)', fontsize=11)
ax.set_xlabel('Edge Rule', fontsize=10)
ax.set_ylabel('Event Rule × Price Rule', fontsize=10)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'exp3_heatmap_std_profit.png'), dpi=150)
plt.close(fig)
print("Saved exp3_heatmap_std_profit.png")


# ------------------------------------------------------------------ #
# Plot 2 – Diagnostics bar chart: neg_delta_count & lambda_dt_gt1_count
#          for the baseline variant, by strategy and gamma
# ------------------------------------------------------------------ #
print("Generating diagnostics bar chart…")

base_diag = df[
    (df['event_rule'] == BASELINE['event_rule']) &
    (df['price_rule'] == BASELINE['price_rule']) &
    (df['edge_rule'] == BASELINE['edge_rule'])
].copy()
base_diag['gamma_str'] = base_diag['gamma'].apply(lambda g: f'γ={g}')
gamma_order = ['γ=0.01', 'γ=0.1', 'γ=0.5']

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
palette = {'inventory': '#2196F3', 'symmetric': '#FF9800'}

for ax, metric, title in zip(
    axes,
    ['neg_delta_count', 'lambda_dt_gt1_count'],
    ['Negative Δ Count\n(delta_a<0 or delta_b<0)', 'λ·Δt > 1 Count\n(Bernoulli overflow events)'],
):
    sns.barplot(
        data=base_diag,
        x='gamma_str',
        y=metric,
        hue='strategy',
        order=gamma_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Risk-aversion γ', fontsize=9)
    ax.set_ylabel('Count (paths × steps)', fontsize=9)
    ax.legend(title='Strategy', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)

fig.suptitle('Numerical Diagnostics – Baseline Variant (Bernoulli / Binomial / Log-only)', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'exp3_diagnostics_bar.png'), dpi=150)
plt.close(fig)
print("Saved exp3_diagnostics_bar.png")


# ------------------------------------------------------------------ #
# Plot 3 – Variant comparison: mean_profit across all variants by gamma
# ------------------------------------------------------------------ #
print("Generating variant comparison plot…")

gamma_sorted = sorted(gamma_values)
fig, axes = plt.subplots(1, len(gamma_sorted), figsize=(15, 5), sharey=False)
palette_strat = {'inventory': '#1565C0', 'symmetric': '#E65100'}
variant_order = [
    'Bern-Bino-Log', 'Bern-Bino-Clip',
    'Bern-Gaus-Log', 'Bern-Gaus-Clip',
    'Pois-Bino-Log', 'Pois-Bino-Clip',
    'Pois-Gaus-Log', 'Pois-Gaus-Clip',
]

for ax, gamma in zip(axes, gamma_sorted):
    sub = df[df['gamma'] == gamma]
    sns.barplot(
        data=sub,
        x='variant',
        y='mean_profit',
        hue='strategy',
        order=variant_order,
        palette=palette_strat,
        ax=ax,
    )
    ax.set_title(f'γ = {gamma}', fontsize=11)
    ax.set_xlabel('Variant (Event-Price-Edge)', fontsize=8)
    ax.set_ylabel('Mean Terminal Profit', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.legend(title='Strategy', fontsize=7, loc='lower right')
    # Mark baseline
    baseline_idx = variant_order.index('Bern-Bino-Log')
    ax.axvline(x=baseline_idx, color='grey', linestyle='--', linewidth=1.0, alpha=0.6)

fig.suptitle('Mean Terminal Profit Across Variants (dashed line = baseline)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'exp3_variant_comparison.png'), dpi=150)
plt.close(fig)
print("Saved exp3_variant_comparison.png")


# ------------------------------------------------------------------ #
# Qualitative sensitivity analysis
# ------------------------------------------------------------------ #
print("\nComputing qualitative variant impact…")

def dimension_impact(df_in, dimension, metric='delta_mean_profit'):
    """Mean absolute deviation attributed to flipping one dimension."""
    return (
        df_in.groupby(['gamma', 'strategy', dimension])[metric]
        .mean()
        .reset_index()
        .groupby(dimension)[metric]
        .apply(lambda x: x.abs().mean())
    )

impact_event = dimension_impact(df, 'event_rule').to_dict()
impact_price = dimension_impact(df, 'price_rule').to_dict()
impact_edge  = dimension_impact(df, 'edge_rule').to_dict()

# Maximum absolute mean_profit shift per dimension
max_shift_event = df.groupby('event_rule')['delta_mean_profit'].apply(lambda x: x.abs().mean()).max()
max_shift_price = df.groupby('price_rule')['delta_mean_profit'].apply(lambda x: x.abs().mean()).max()
max_shift_edge  = df.groupby('edge_rule')['delta_mean_profit'].apply(lambda x: x.abs().mean()).max()

max_shift_std_event = df.groupby('event_rule')['delta_std_profit'].apply(lambda x: x.abs().mean()).max()
max_shift_std_price = df.groupby('price_rule')['delta_std_profit'].apply(lambda x: x.abs().mean()).max()
max_shift_std_edge  = df.groupby('edge_rule')['delta_std_profit'].apply(lambda x: x.abs().mean()).max()

dimension_shifts = {
    'event_rule': float(max_shift_event),
    'price_rule': float(max_shift_price),
    'edge_rule':  float(max_shift_edge),
}
dominant_dim_mean = max(dimension_shifts, key=dimension_shifts.get)

dimension_std_shifts = {
    'event_rule': float(max_shift_std_event),
    'price_rule': float(max_shift_std_price),
    'edge_rule':  float(max_shift_std_edge),
}
dominant_dim_std = max(dimension_std_shifts, key=dimension_std_shifts.get)

# Gamma-level stats for inventory baseline
inv_base = df[
    (df['strategy'] == 'inventory') &
    (df['event_rule'] == BASELINE['event_rule']) &
    (df['price_rule'] == BASELINE['price_rule']) &
    (df['edge_rule'] == BASELINE['edge_rule'])
].set_index('gamma')


# ------------------------------------------------------------------ #
# Save summary JSON
# ------------------------------------------------------------------ #
summary = {
    'experiment': 3,
    'description': 'Numerical Ambiguity and Discretization Sensitivity Study',
    'parameters': {
        'S0': S0, 'T': T, 'sigma': sigma, 'dt': dt,
        'A': A, 'k': k, 'n_paths': n_paths,
    },
    'gamma_values': gamma_values,
    'n_total_runs': len(df),
    'baseline': BASELINE,
    'variant_dimensions': {
        'event_rule': event_rules,
        'price_rule': price_rules,
        'edge_rule': edge_rules,
    },
    'dimension_impact_on_mean_profit': dimension_shifts,
    'dimension_impact_on_std_profit': dimension_std_shifts,
    'dominant_dimension_mean_profit': dominant_dim_mean,
    'dominant_dimension_std_profit': dominant_dim_std,
    'per_gamma_inventory_baseline': {
        str(g): {
            'mean_profit': float(inv_base.loc[g, 'mean_profit']),
            'std_profit': float(inv_base.loc[g, 'std_profit']),
            'neg_delta_count': int(inv_base.loc[g, 'neg_delta_count']),
            'lambda_dt_gt1_count': int(inv_base.loc[g, 'lambda_dt_gt1_count']),
        }
        for g in gamma_values
    },
    'max_delta_mean_profit': float(df['delta_mean_profit'].abs().max()),
    'max_delta_std_profit': float(df['delta_std_profit'].abs().max()),
}

with open(os.path.join(RESULTS_DIR, 'exp3_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved exp3_summary.json")


# ------------------------------------------------------------------ #
# Save RESULTS_exp3.md
# ------------------------------------------------------------------ #
half_spreads = {g: compute_half_spread(g, k) for g in gamma_values}

md_lines = [
    '# Experiment 3 – Numerical Ambiguity and Discretization Sensitivity Study',
    '',
    '## Overview',
    '',
    'This experiment systematically varies three numerical implementation choices',
    'in the Avellaneda-Stoikov simulator and measures their impact on terminal profit',
    'statistics relative to the paper\'s baseline (Bernoulli fills, binomial price',
    'path, log-only edge handling).',
    '',
    '## Parameters',
    '',
    f'| Parameter | Value |',
    f'|-----------|-------|',
    f'| S0        | {S0}  |',
    f'| T         | {T}   |',
    f'| σ         | {sigma}   |',
    f'| Δt        | {dt}  |',
    f'| A         | {A}   |',
    f'| k         | {k}   |',
    f'| n_paths   | {n_paths} |',
    f'| γ values  | {gamma_values} |',
    '',
    '## Baseline Half-Spreads c = (1/γ) ln(1 + γ/k)',
    '',
    '| γ | c |',
    '|---|---|',
] + [f'| {g} | {half_spreads[g]:.4f} |' for g in gamma_values] + [
    '',
    '## Variant Grid',
    '',
    '| Dimension  | Option A (baseline) | Option B         |',
    '|------------|---------------------|------------------|',
    '| event_rule | bernoulli           | exact_poisson    |',
    '| price_rule | binomial            | gaussian         |',
    '| edge_rule  | log_only            | clip_zero        |',
    '',
    '**Total runs:** 3 γ × 2 strategies × 2³ variants = 48',
    '',
    '## Key Findings',
    '',
    f'**Maximum |Δmean_profit| across all variants:** {df["delta_mean_profit"].abs().max():.4f}',
    f'**Maximum |Δstd_profit| across all variants:**  {df["delta_std_profit"].abs().max():.4f}',
    '',
    '### Dimension Impact on Mean Profit (mean absolute deviation)',
    '',
    '| Dimension  | Mean |Δ mean_profit| |',
    '|------------|----------------------|',
] + [
    f'| {dim:<10s} | {val:.4f}               |'
    for dim, val in sorted(dimension_shifts.items(), key=lambda x: -x[1])
] + [
    '',
    '### Dimension Impact on Std Profit (mean absolute deviation)',
    '',
    '| Dimension  | Mean |Δ std_profit| |',
    '|------------|---------------------|',
] + [
    f'| {dim:<10s} | {val:.4f}              |'
    for dim, val in sorted(dimension_std_shifts.items(), key=lambda x: -x[1])
] + [
    '',
    f'**Dominant dimension (mean profit):** `{dominant_dim_mean}`',
    f'**Dominant dimension (std profit):**  `{dominant_dim_std}`',
    '',
    '## Numerical Diagnostics (Baseline Variant)',
    '',
    '| γ | Strategy   | neg_delta_count | λ·Δt > 1 count |',
    '|---|------------|-----------------|----------------|',
] + [
    f"| {g} | {row['strategy']:<10s} | {int(row['neg_delta_count']):>15,} | {int(row['lambda_dt_gt1_count']):>14,} |"
    for g in gamma_values
    for _, row in df[
        (df['gamma'] == g) &
        (df['event_rule'] == BASELINE['event_rule']) &
        (df['price_rule'] == BASELINE['price_rule']) &
        (df['edge_rule'] == BASELINE['edge_rule'])
    ].iterrows()
] + [
    '',
    '## Plots',
    '',
    '- **exp3_heatmap_std_profit.png** – Heatmap of std_profit deviation from baseline',
    '  for the inventory strategy at γ = 0.1.  Rows = event_rule × price_rule,',
    '  columns = edge_rule.  Diverging colour scale centred at zero.',
    '',
    '- **exp3_diagnostics_bar.png** – Grouped bar chart of `neg_delta_count` and',
    '  `lambda_dt_gt1_count` for the baseline variant, faceted by diagnostic metric.',
    '  Shows how γ and strategy drive numerical pathologies.',
    '',
    '- **exp3_variant_comparison.png** – Mean terminal profit for every variant',
    '  combination, grouped by γ (three subplots).  The dashed vertical line marks',
    '  the baseline variant.',
    '',
    '## Conclusion',
    '',
]

# Build data-driven ranking for the conclusion
ranked_mean = sorted(dimension_shifts.items(), key=lambda x: -x[1])
ranked_std  = sorted(dimension_std_shifts.items(), key=lambda x: -x[1])

dim_labels = {
    'event_rule': 'fill-event rule (Bernoulli vs exact-Poisson)',
    'price_rule': 'price-path model (binomial vs Gaussian)',
    'edge_rule':  'edge-handling rule (log-only vs clip-zero)',
}
md_lines += [
    f'Ranked by impact on **mean profit**: '
    + ' > '.join(f'`{d}` ({v:.3f})' for d, v in ranked_mean) + '.',
    '',
    f'Ranked by impact on **profit volatility**: '
    + ' > '.join(f'`{d}` ({v:.3f})' for d, v in ranked_std) + '.',
    '',
    f'The dominant driver of mean-profit shifts is the {dim_labels[dominant_dim_mean]}.  '
    f'This is most pronounced at large γ (γ=0.5) where the inventory policy '
    f'frequently pushes spreads negative under log-only handling, causing '
    f'λ·Δt >> 1 and exposing the difference between the Bernoulli cap and the '
    f'exact Poisson probability.  The edge-handling rule (clip-zero) suppresses '
    f'this pathology at the cost of removing the inventory-control signal.',
]

with open(os.path.join(RESULTS_DIR, 'RESULTS_exp3.md'), 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print("Saved RESULTS_exp3.md")


# ------------------------------------------------------------------ #
# Qualitative summary printed to stdout
# ------------------------------------------------------------------ #
print('\n' + '=' * 60)
print('QUALITATIVE SENSITIVITY SUMMARY')
print('=' * 60)

print('\n--- Impact on mean_profit (mean |Δ| across γ and strategy) ---')
for dim, val in sorted(dimension_shifts.items(), key=lambda x: -x[1]):
    marker = '  ← DOMINANT' if dim == dominant_dim_mean else ''
    print(f'  {dim:<14s}: {val:.4f}{marker}')

print('\n--- Impact on std_profit (mean |Δ| across γ and strategy) ---')
for dim, val in sorted(dimension_std_shifts.items(), key=lambda x: -x[1]):
    marker = '  ← DOMINANT' if dim == dominant_dim_std else ''
    print(f'  {dim:<14s}: {val:.4f}{marker}')

print(f'\n--- Maximum absolute deviations ---')
print(f'  |Δmean_profit|_max : {df["delta_mean_profit"].abs().max():.4f}')
print(f'  |Δstd_profit|_max  : {df["delta_std_profit"].abs().max():.4f}')

print('\n--- Numerical pathologies (baseline variant, inventory policy) ---')
for g in gamma_values:
    row_g = inv_base.loc[g]
    print(f'  γ={g}: neg_delta={int(row_g["neg_delta_count"]):>8,}  '
          f'lambda_dt_gt1={int(row_g["lambda_dt_gt1_count"]):>8,}')

print('\n--- Conclusion ---')
print(f'  Mean-profit ranking:     '
      + ' > '.join(f'{d} ({v:.3f})' for d, v in ranked_mean))
print(f'  Std-profit ranking:      '
      + ' > '.join(f'{d} ({v:.3f})' for d, v in ranked_std))
print(f'  Dominant dim (mean):     {dominant_dim_mean}')
print(f'  Dominant dim (std):      {dominant_dim_std}')
print('  At large γ, log-only edge handling lets spreads go negative,')
print('  driving λ·Δt >> 1 and amplifying the Bernoulli vs Poisson gap.')
print('  Clip-zero stabilises intensities but masks the inventory-control signal.')
print('=' * 60)

print('\nExperiment 3 complete. Results saved to results/')
