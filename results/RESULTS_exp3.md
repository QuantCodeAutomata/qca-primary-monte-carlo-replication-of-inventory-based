# Experiment 3 – Numerical Ambiguity and Discretization Sensitivity Study

## Overview

This experiment systematically varies three numerical implementation choices
in the Avellaneda-Stoikov simulator and measures their impact on terminal profit
statistics relative to the paper's baseline (Bernoulli fills, binomial price
path, log-only edge handling).

## Parameters

| Parameter | Value |
|-----------|-------|
| S0        | 100.0  |
| T         | 1.0   |
| σ         | 2.0   |
| Δt        | 0.005  |
| A         | 140.0   |
| k         | 1.5   |
| n_paths   | 1000 |
| γ values  | [0.1, 0.01, 0.5] |

## Baseline Half-Spreads c = (1/γ) ln(1 + γ/k)

| γ | c |
|---|---|
| 0.1 | 0.6454 |
| 0.01 | 0.6645 |
| 0.5 | 0.5754 |

## Variant Grid

| Dimension  | Option A (baseline) | Option B         |
|------------|---------------------|------------------|
| event_rule | bernoulli           | exact_poisson    |
| price_rule | binomial            | gaussian         |
| edge_rule  | log_only            | clip_zero        |

**Total runs:** 3 γ × 2 strategies × 2³ variants = 48

## Key Findings

**Maximum |Δmean_profit| across all variants:** 27.9243
**Maximum |Δstd_profit| across all variants:**  1.0576

### Dimension Impact on Mean Profit (mean absolute deviation)

| Dimension  | Mean |Δ mean_profit| |
|------------|----------------------|
| event_rule | 9.0869               |
| edge_rule  | 7.4511               |
| price_rule | 5.7645               |

### Dimension Impact on Std Profit (mean absolute deviation)

| Dimension  | Mean |Δ std_profit| |
|------------|---------------------|
| event_rule | 0.5148              |
| price_rule | 0.4870              |
| edge_rule  | 0.3402              |

**Dominant dimension (mean profit):** `event_rule`
**Dominant dimension (std profit):**  `event_rule`

## Numerical Diagnostics (Baseline Variant)

| γ | Strategy   | neg_delta_count | λ·Δt > 1 count |
|---|------------|-----------------|----------------|
| 0.1 | inventory  |           2,497 |             75 |
| 0.1 | symmetric  |               0 |              0 |
| 0.01 | inventory  |               0 |              0 |
| 0.01 | symmetric  |               0 |              0 |
| 0.5 | inventory  |          45,437 |         36,295 |
| 0.5 | symmetric  |               0 |              0 |

## Plots

- **exp3_heatmap_std_profit.png** – Heatmap of std_profit deviation from baseline
  for the inventory strategy at γ = 0.1.  Rows = event_rule × price_rule,
  columns = edge_rule.  Diverging colour scale centred at zero.

- **exp3_diagnostics_bar.png** – Grouped bar chart of `neg_delta_count` and
  `lambda_dt_gt1_count` for the baseline variant, faceted by diagnostic metric.
  Shows how γ and strategy drive numerical pathologies.

- **exp3_variant_comparison.png** – Mean terminal profit for every variant
  combination, grouped by γ (three subplots).  The dashed vertical line marks
  the baseline variant.

## Conclusion

Ranked by impact on **mean profit**: `event_rule` (9.087) > `edge_rule` (7.451) > `price_rule` (5.764).

Ranked by impact on **profit volatility**: `event_rule` (0.515) > `price_rule` (0.487) > `edge_rule` (0.340).

The dominant driver of mean-profit shifts is the fill-event rule (Bernoulli vs exact-Poisson).  This is most pronounced at large γ (γ=0.5) where the inventory policy frequently pushes spreads negative under log-only handling, causing λ·Δt >> 1 and exposing the difference between the Bernoulli cap and the exact Poisson probability.  The edge-handling rule (clip-zero) suppresses this pathology at the cost of removing the inventory-control signal.
