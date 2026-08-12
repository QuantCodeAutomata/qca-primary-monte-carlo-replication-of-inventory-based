# Project: Monte Carlo Market-Making Replication

Replication study of Avellaneda & Stoikov (2008) optimal market-making.
Pure NumPy / pandas / matplotlib stack — no backtesting frameworks.

## Directory Layout
```
src/
  config.py          SimConfig dataclass + compute_half_spread()
  simulation.py      Vectorised Monte Carlo engine (simulate_paths)
  policies.py        Exp 1 policy factories (inventory + symmetric)
  policies_exp2.py   Exp 2 theory-consistent policies (Section 3.2)
  reporting.py       Stats + DataFrame + Markdown/CSV output
  sensitivity.py     Sensitivity-analysis utilities (pre-existing)
exp/
  run_exp1.py        Experiment 1 runner (Table 1-3 replication)
  run_exp2.py        Experiment 2 runner (Section 3.2 policies)
results/             All CSV/JSON/PNG outputs (git-ignored large PNGs)
tests/               pytest suite for core modules
```

## Key Design Decisions

### Vectorisation
- `simulate_paths` loops over **time steps** (200), vectorises over **n_paths** (1000).
- Pre-generates all random draws upfront: `price_rand`, `ask_rand`, `bid_rand`,
  each shape `(n_paths, n_steps)`, using `numpy.random.default_rng(seed)`.

### Common Random Numbers (CRN)
- Both strategies are called with `seed=42` so they receive identical draw matrices.
- This enables variance reduction when comparing strategy means.

### Policy Interface
- Signature: `policy(S, q, t, tau, gamma, c) -> (delta_a, delta_b)`
- Inputs `S`, `q` are **numpy arrays** of shape `(n_paths,)`; outputs may be
  arrays or scalars (numpy broadcasts both correctly in the fill conditions).

### Half-Spread Formula
```
c = (1/gamma) * log(1 + gamma/k)
```
Constant for a given (gamma, k) pair — precomputed outside the step loop.

### Inventory Policy (Exp 1)
```
delta_a = c - q * gamma * sigma^2 * tau   # ask half-spread (skewed down when long)
delta_b = c + q * gamma * sigma^2 * tau   # bid half-spread (skewed up when long)
```

### Price Process
Symmetric Bernoulli random walk: `S ± sigma * sqrt(dt)` with p=0.5 each step.

## Running Experiments
```bash
cd /workspace/project
python exp/run_exp1.py   # Tables 1-3 + 4 PNG figures + CSV/JSON/MD
python exp/run_exp2.py   # Experiment 2
```

## Numerical Results (Exp 1, seed=42, n_paths=1000)
| Strategy  | γ    | c      | mean_profit | std_profit |
|-----------|------|--------|-------------|------------|
| Inventory | 0.10 | 0.6454 | 64.28       | 5.86       |
| Symmetric | 0.10 | 0.6454 | 68.57       | 13.43      |
| Inventory | 0.01 | 0.6645 | 68.32       | 8.72       |
| Symmetric | 0.01 | 0.6645 | 68.79       | 13.39      |
| Inventory | 0.50 | 0.5754 | 23.71       | 4.79       |
| Symmetric | 0.50 | 0.5754 | 67.89       | 13.47      |

Key insight: inventory policy dramatically reduces std_profit at the cost of
mean profit for high γ (aggressive inventory control squeezes spread to zero).

## Experiment 2 Design (Section 3.2 theory-consistent policies)

### Key difference from Exp 1
Exp 2 uses the **full time-varying spread** `gamma*sigma^2*(T-t) + 2c` instead of
the constant `2c` approximation used in Exp 1's table-matching formulas.

### Section 3.2 formulas (src/policies_exp2.py)
```
Inventory:  delta_a = ((1 - 2q) * gamma*sigma^2*tau) / 2 + c
            delta_b = ((1 + 2q) * gamma*sigma^2*tau) / 2 + c
Symmetric:  delta_a = delta_b = (gamma*sigma^2*tau) / 2 + c
```
Both strategies share identical total spread at every step (verified algebraically
and numerically via `verify_spread_property()`).

### Exp 2 self-contained fallback (exp/run_exp2.py)
run_exp2.py inserts `../src` into sys.path and wraps the import in `try/except`.
On `ImportError` it falls back to inline copies of all policy functions, ensuring
the script is fully runnable even before src/policies_exp2.py exists.

### Numerical Results (Exp 2, seed=42, n_paths=1000)
| Strategy  | γ    | mean_profit | std_profit | mean_q  |
|-----------|------|-------------|------------|---------|
| Inventory | 0.10 | 65.06       | 6.77       | 0.058   |
| Symmetric | 0.10 | 68.11       | 13.11      | -0.236  |
| Inventory | 0.01 | 68.40       | 9.25       | 0.077   |
| Symmetric | 0.01 | 68.53       | 13.54      | -0.180  |
| Inventory | 0.50 | 48.62       | 5.75       | 0.026   |
| Symmetric | 0.50 | 58.58       | 11.65      | 0.150   |

Key insight: inventory policy achieves substantially lower std_profit and near-zero
mean_q (tight inventory control) vs symmetric. Profit gap widens with γ.

### Exp 2 output files
- `results/exp2_results.csv` — flat results table (6 rows)
- `results/exp2_summary.json` — parameters + results
- `results/RESULTS_exp2.md` — markdown report
- `results/exp2_pnl_hist_gamma_{g}.png` — P&L histograms (3 files)
- `results/exp2_spread_comparison.png` — total spread over time at gamma=0.1

## Experiment 3 Design (Numerical Ambiguity and Discretization Sensitivity Study)

### Purpose
Sweeps a 2×2×2 grid of numerical implementation choices to measure their impact
on terminal profit statistics relative to the paper's baseline.

### Module: `src/sensitivity.py`
Self-contained; defines `SensitivityConfig` (same fields/defaults as SimConfig),
all variant functions, and the main entry point `simulate_sensitivity_variant`.

### Variant Dimensions

| Dimension  | Baseline (Option A) | Alternate (Option B)     |
|------------|---------------------|--------------------------|
| event_rule | `fill_bernoulli`    | `fill_exact_poisson`     |
| price_rule | `price_binomial`    | `price_gaussian`         |
| edge_rule  | `edge_log_only`     | `edge_clip_zero`         |

- **`fill_bernoulli(rand, lam_dt)`**: fill if `rand < lam_dt` (paper approximation)
- **`fill_exact_poisson(rand, lam_dt)`**: fill if `rand < 1 - exp(-lam_dt)` (exact)
- **`price_binomial(S, sigma, dt, rand)`**: `S ± sigma*sqrt(dt)` (uniform rand)
- **`price_gaussian(S, sigma, dt, rand)`**: `S + sigma*sqrt(dt)*Z` (caller passes Z~N(0,1))
- **`edge_log_only(delta)`**: identity — negative deltas pass through unchanged
- **`edge_clip_zero(delta)`**: `np.maximum(delta, 0)` — prevents λ·Δt > 1
- **`cap_prob(prob)`**: `np.minimum(prob, 1)` — optional Bernoulli overflow guard

### `simulate_sensitivity_variant` Step Ordering
1. Compute raw `delta_a`, `delta_b` → count `neg_delta_count` (pre-edge, OR logic)
2. Apply `edge_fn`
3. Compute `lam_a_dt`, `lam_b_dt` → count `lambda_dt_gt1_count` (post-edge, pre-cap, OR logic)
4. Optionally cap probabilities (if `cap_prob_flag=True`)
5. Sample fills via `fill_fn` → update `simult_fill_count` (AND logic), `total_ask_fills`, `total_bid_fills`
6. Apply cash/inventory updates: `X[ask] += S+da`, `X[bid] -= S-db`, `q[ask] -= 1`, `q[bid] += 1`
7. Update price via `price_fn`
8. Terminal: `profit = X + q * S_T`

### RNG Design
- `ask_rand`, `bid_rand`: always uniform (fill decisions use CDF inversion either way)
- `price_rand`: uniform (binomial) OR `standard_normal` (gaussian) — chosen before the loop
- All random arrays pre-generated: shape `(n_paths, n_steps)` — same as `simulation.py`

### Exp 3 Numerical Findings (seed=42, n_paths=1000)

**Baseline variant diagnostics (inventory policy, log_only edge):**
| γ    | neg_delta_count | lambda_dt_gt1_count |
|------|-----------------|---------------------|
| 0.01 | 0               | 0                   |
| 0.10 | ~2,500          | ~75                 |
| 0.50 | ~45,000         | ~36,000             |

- Symmetric strategy always has 0 neg_delta_count (c > 0 always)
- `clip_zero` edge always produces 0 `lambda_dt_gt1_count` (max λ·Δt = 0.7)
- **Dominant dimension on both mean profit and std profit: `event_rule`**
  (Bernoulli vs exact Poisson diverges when λ·Δt >> 1, i.e., large γ + log_only)
- Max |Δmean_profit| ≈ 27.9, max |Δstd_profit| ≈ 1.06

### Exp 3 Output Files
- `results/exp3_sensitivity.csv` — 48 rows × 14 cols (full grid + delta columns)
- `results/exp3_summary.json` — parameters, dominance ranking, per-gamma stats
- `results/RESULTS_exp3.md` — markdown narrative with data-driven conclusion
- `results/exp3_heatmap_std_profit.png` — 4×2 heatmap (inventory, γ=0.1)
- `results/exp3_diagnostics_bar.png` — neg_delta & λ>1 counts, baseline variant
- `results/exp3_variant_comparison.png` — mean_profit across 8 variants, 3 γ subplots

```bash
python exp/run_exp3.py   # ~5–10 s, 48 runs
```

## Known Fixes / Gotchas

- `src/simulation.py` returns diagnostics key `simult_fill_count` (was `simult_fills`).
  All consumers (`exp/run_exp1.py` line 184, `tests/test_simulation.py` line 93) must
  use `simult_fill_count`. Do not revert this key name.
- `exp/run_exp1.py` `save_diagnostics()` uses `d['simult_fill_count']` (column
  stored as `simult_fills` in the CSV — cosmetic difference, no functional impact).

## Dependencies
```
numpy>=1.26, pandas>=2.2, matplotlib>=3.8, seaborn>=0.13
```
No QuantLib, vectorbt, or backtesting libraries used anywhere.
