"""
Experiment 2: Theory-consistent Monte Carlo market-making simulation.

Implements the full approximate quoting formulas from Section 3.2:
    delta^a + delta^b = gamma * sigma^2 * (T - t) + 2c

Note: This experiment uses theory-consistent spread gamma*sigma^2*(T-t) + 2c
rather than constant 2c from tables.

Compares inventory-skewed vs symmetric-centered policies over the same
time-varying total spread.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import from src/ with inline fallback
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, 'src'))

USING_SRC = False
try:
    from policies_exp2 import (
        make_inventory_policy_exp2,
        make_symmetric_policy_exp2,
        verify_spread_property,
    )
    USING_SRC = True
    print("Loaded policies from src/policies_exp2.py")
except ImportError:
    print("src/policies_exp2.py not available — using inline fallback implementations")


# --- Inline fallback implementations (mirror src/policies_exp2.py exactly) ---

if not USING_SRC:

    def make_inventory_policy_exp2(sigma: float = 2.0):
        """
        Theory-consistent inventory policy using full Section 3.2 approximate formulas.
        delta^a = ((1 - 2*q) * gamma * sigma^2 * tau) / 2 + c
        delta^b = ((1 + 2*q) * gamma * sigma^2 * tau) / 2 + c
        """
        def policy(S, q, t, tau, gamma, c):
            # Full approximate finite-horizon formulas (Section 3.2)
            delta_a = ((1 - 2 * q) * gamma * sigma**2 * tau) / 2.0 + c
            delta_b = ((1 + 2 * q) * gamma * sigma**2 * tau) / 2.0 + c
            return delta_a, delta_b
        return policy

    def make_symmetric_policy_exp2(sigma: float = 2.0):
        """
        Theory-consistent symmetric benchmark: same time-varying total spread
        but inventory-independent, centered at mid-price.
        delta^a_sym = delta^b_sym = (gamma * sigma^2 * tau) / 2 + c
        Total spread = gamma * sigma^2 * tau + 2c  (matches Section 3.2 formula)
        """
        def policy(S, q, t, tau, gamma, c):
            delta_sym = (gamma * sigma**2 * tau) / 2.0 + c
            return delta_sym, delta_sym
        return policy

    def verify_spread_property(delta_a: np.ndarray, delta_b: np.ndarray,
                               gamma: float, sigma: float, tau: float, c: float) -> bool:
        """Verify delta^a + delta^b == gamma * sigma^2 * tau + 2c for inventory strategy."""
        expected_spread = gamma * sigma**2 * tau + 2 * c
        actual_spread = delta_a + delta_b
        return np.allclose(actual_spread, expected_spread, atol=1e-10)


# ---------------------------------------------------------------------------
# Parameters  (same as Experiment 1)
# ---------------------------------------------------------------------------

S0 = 100.0
T = 1.0
SIGMA = 2.0
dt = 0.005
A = 140.0
k = 1.5
N_PATHS = 1000
GAMMA_VALUES = [0.1, 0.01, 0.5]

N_STEPS = int(T / dt)   # 200

# ---------------------------------------------------------------------------
# Simulation engine  (vectorised over N_PATHS)
# ---------------------------------------------------------------------------

def run_simulation(policy, gamma, c,
                   price_rand, ask_rand, bid_rand,
                   verify_spread=False):
    """
    Vectorised Monte Carlo simulation using pre-generated random arrays.

    Parameters
    ----------
    policy        : callable (S, q, t, tau, gamma, c) -> (delta_a, delta_b)
    gamma, c      : model parameters
    price_rand    : (N_PATHS, N_STEPS) uniform draws for mid-price updates
    ask_rand      : (N_PATHS, N_STEPS) uniform draws for ask-side fill events
    bid_rand      : (N_PATHS, N_STEPS) uniform draws for bid-side fill events
    verify_spread : if True, spot-check delta_a + delta_b == theory on path 0

    Returns
    -------
    profit        : (N_PATHS,) terminal P&L = X + q*S
    final_q       : (N_PATHS,) terminal inventory
    diagnostics   : dict with aggregate diagnostic counts
    spread_path0  : (N_STEPS,) total spread (delta_a + delta_b) recorded on path 0
    """
    n_paths, n_steps = price_rand.shape

    S = np.full(n_paths, S0)
    q = np.zeros(n_paths)
    X = np.zeros(n_paths)

    neg_delta_count   = 0
    high_lambda_count = 0
    simul_fill_count  = 0
    spread_verif_failures = 0

    spread_path0 = np.empty(n_steps)

    for step in range(n_steps):
        t   = step * dt
        tau = T - t           # time remaining

        # (a) Quotes from policy — may be scalar or (n_paths,)
        delta_a, delta_b = policy(S, q, t, tau, gamma, c)

        # Ensure (n_paths,) arrays for unified downstream arithmetic
        delta_a = np.broadcast_to(np.asarray(delta_a, dtype=float), (n_paths,)).copy()
        delta_b = np.broadcast_to(np.asarray(delta_b, dtype=float), (n_paths,)).copy()

        # Diagnostic: count negative half-spreads
        neg_delta_count += int(np.sum(delta_a < 0) + np.sum(delta_b < 0))

        # (c) Poisson arrival intensities
        lambda_a = A * np.exp(-k * delta_a)
        lambda_b = A * np.exp(-k * delta_b)

        # Diagnostic: count steps where Bernoulli approximation breaks down
        high_lambda_count += int(np.sum(lambda_a * dt > 1.0) + np.sum(lambda_b * dt > 1.0))

        # (d) Posted prices
        p_a = S + delta_a    # ask price (MM sells)
        p_b = S - delta_b    # bid price (MM buys)

        # (e-f) Bernoulli fill events
        ask_fill = ask_rand[:, step] < lambda_a * dt
        bid_fill = bid_rand[:, step] < lambda_b * dt

        # Diagnostic: simultaneous fills on the same path at the same step
        simul_fill_count += int(np.sum(ask_fill & bid_fill))

        # (g) Update inventory and cash
        #     ask fill: MM sells → q--, X += p_a
        #     bid fill: MM buys  → q++, X -= p_b
        q = q - ask_fill.astype(float) + bid_fill.astype(float)
        X = X + ask_fill * p_a - bid_fill * p_b

        # (h) Mid-price update: symmetric binomial random walk
        price_move = np.where(
            price_rand[:, step] < 0.5,
            -SIGMA * np.sqrt(dt),
             SIGMA * np.sqrt(dt),
        )
        S = S + price_move

        # Spread verification on path 0 (inventory strategy only)
        if verify_spread:
            ok = verify_spread_property(
                delta_a[0:1], delta_b[0:1], gamma, SIGMA, tau, c
            )
            if not ok:
                spread_verif_failures += 1

        # Record spread for path 0
        spread_path0[step] = delta_a[0] + delta_b[0]

    # Terminal liquidation at mid-price
    profit = X + q * S

    diagnostics = {
        'neg_delta_count':       neg_delta_count,
        'high_lambda_count':     high_lambda_count,
        'simul_fill_count':      simul_fill_count,
        'spread_verif_failures': spread_verif_failures,
    }
    return profit, q, diagnostics, spread_path0


# ---------------------------------------------------------------------------
# Helper: DataFrame → Markdown table (no tabulate required)
# ---------------------------------------------------------------------------

def _df_to_md_table(df: pd.DataFrame) -> str:
    col_strs  = [str(c) for c in df.columns]
    header    = '| ' + ' | '.join(col_strs) + ' |'
    separator = '| ' + ' | '.join('---' for _ in col_strs) + ' |'
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join([header, separator] + rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

results_dir = os.path.join(_project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

all_rows = []     # accumulated across all gamma values

for gamma in GAMMA_VALUES:

    # c = (1/gamma) * ln(1 + gamma/k)
    c = (1.0 / gamma) * np.log(1.0 + gamma / k)

    print(f"\n{'='*62}")
    print(f"  gamma = {gamma}   c = {c:.6f}")
    print(f"  Analytical total spread at t=0 : {gamma * SIGMA**2 * T + 2*c:.6f}")
    print(f"  Analytical total spread at t=T : {2*c:.6f}")
    print(f"{'='*62}")

    # Pre-generate ALL random numbers with a fixed seed for reproducibility.
    # Same random arrays are fed to both policies so results are directly comparable
    # (common-random-numbers variance reduction).
    rng = np.random.default_rng(seed=42)
    price_rand = rng.random((N_PATHS, N_STEPS))   # shape (n_paths, n_steps)
    ask_rand   = rng.random((N_PATHS, N_STEPS))
    bid_rand   = rng.random((N_PATHS, N_STEPS))

    inv_policy = make_inventory_policy_exp2(sigma=SIGMA)
    sym_policy = make_symmetric_policy_exp2(sigma=SIGMA)

    # --- Run inventory simulation ---
    profit_inv, q_inv, diag_inv, spread_inv = run_simulation(
        inv_policy, gamma, c,
        price_rand, ask_rand, bid_rand,
        verify_spread=True,
    )

    # --- Run symmetric simulation (same random paths) ---
    profit_sym, q_sym, diag_sym, spread_sym = run_simulation(
        sym_policy, gamma, c,
        price_rand, ask_rand, bid_rand,
        verify_spread=False,
    )

    # --- Print diagnostics ---
    spread_ok_inv = (diag_inv['spread_verif_failures'] == 0)
    print(f"\nInventory policy diagnostics:")
    print(f"  Negative half-spreads (total across paths × steps): {diag_inv['neg_delta_count']}")
    print(f"  Steps where lambda*dt > 1 (total):                  {diag_inv['high_lambda_count']}")
    print(f"  Simultaneous fills (total):                         {diag_inv['simul_fill_count']}")
    print(f"  Spread property verified (all {N_STEPS} steps, path 0): {spread_ok_inv}")

    print(f"\nSymmetric policy diagnostics:")
    print(f"  Negative half-spreads (total across paths × steps): {diag_sym['neg_delta_count']}")
    print(f"  Steps where lambda*dt > 1 (total):                  {diag_sym['high_lambda_count']}")
    print(f"  Simultaneous fills (total):                         {diag_sym['simul_fill_count']}")

    # --- Accumulate results ---
    row_inv = {
        'Strategy':    'Inventory',
        'Gamma':       gamma,
        'Spread_type': 'inventory-skewed',
        'mean_profit': float(np.mean(profit_inv)),
        'std_profit':  float(np.std(profit_inv)),
        'mean_q':      float(np.mean(q_inv)),
        'std_q':       float(np.std(q_inv)),
    }
    row_sym = {
        'Strategy':    'Symmetric',
        'Gamma':       gamma,
        'Spread_type': 'symmetric-centered',
        'mean_profit': float(np.mean(profit_sym)),
        'std_profit':  float(np.std(profit_sym)),
        'mean_q':      float(np.mean(q_sym)),
        'std_q':       float(np.std(q_sym)),
    }
    all_rows.extend([row_inv, row_sym])

    df_gamma = pd.DataFrame([row_inv, row_sym])
    print(f"\nResults for gamma = {gamma}:")
    print(df_gamma.to_string(index=False))

    # -----------------------------------------------------------------------
    # Plot 1: Terminal P&L histograms
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(profit_inv, bins=50, alpha=0.60, density=True,
            color='steelblue', label='Inventory')
    ax.hist(profit_sym, bins=50, alpha=0.60, density=True,
            color='darkorange', label='Symmetric')
    ax.axvline(np.mean(profit_inv), color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Inv mean = {np.mean(profit_inv):.2f}')
    ax.axvline(np.mean(profit_sym), color='darkorange', linestyle='--', linewidth=1.5,
               label=f'Sym mean = {np.mean(profit_sym):.2f}')
    ax.set_xlabel('Terminal P&L')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Exp 2: Terminal P&L (gamma={gamma})\n'
        f'Spread: $\\gamma\\sigma^2(T-t) + 2c$  [Section 3.2]'
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    pnl_path = os.path.join(results_dir, f'exp2_pnl_hist_gamma_{gamma}.png')
    fig.savefig(pnl_path, dpi=120)
    plt.close(fig)
    print(f"Saved {pnl_path}")

    # -----------------------------------------------------------------------
    # Plot 2: Spread over time — inventory vs symmetric at gamma = 0.1
    # -----------------------------------------------------------------------
    if abs(gamma - 0.1) < 1e-9:
        time_axis  = np.arange(N_STEPS) * dt
        tau_axis   = T - time_axis
        analytical = gamma * SIGMA**2 * tau_axis + 2 * c

        fig, ax = plt.subplots(figsize=(9, 5))

        # Both simulated spreads should lie on the analytical curve
        ax.plot(time_axis, spread_inv,
                color='steelblue', linewidth=2.0, label='Inventory (path 0)')
        ax.plot(time_axis, spread_sym,
                color='darkorange', linewidth=2.0, linestyle='--',
                label='Symmetric (path 0)')
        ax.plot(time_axis, analytical,
                color='green', linewidth=1.2, linestyle=':',
                label=r'Analytical: $\gamma\sigma^2\tau + 2c$')

        ax.set_xlabel('Time $t$')
        ax.set_ylabel(r'Total spread $\delta^a + \delta^b$')
        ax.set_title(
            f'Exp 2: Bid-Ask Spread Over Time (gamma={gamma}, path 0)\n'
            r'Both strategies share total spread $\gamma\sigma^2(T-t) + 2c$'
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        spread_path = os.path.join(results_dir, 'exp2_spread_comparison.png')
        fig.savefig(spread_path, dpi=120)
        plt.close(fig)
        print(f"Saved {spread_path}")


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_rows)
csv_path = os.path.join(results_dir, 'exp2_results.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

print("\n--- Full results table ---")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

md_lines = [
    '# Experiment 2 Results',
    '',
    '> **Note:** This experiment uses theory-consistent spread `gamma*sigma^2*(T-t) + 2c`',
    '> rather than constant `2c` from tables (Section 3.2, Avellaneda-Stoikov).',
    '',
    f'**Parameters:** S0={S0}, T={T}, sigma={SIGMA}, dt={dt}, A={A}, k={k}, '
    f'n_paths={N_PATHS}',
    '',
    f'**Gamma values:** {GAMMA_VALUES}',
    '',
    '**Half-spread constant:** `c = (1/gamma) * ln(1 + gamma/k)`',
    '',
]

for gamma in GAMMA_VALUES:
    c_val = (1.0 / gamma) * np.log(1.0 + gamma / k)
    md_lines.append(f'## gamma = {gamma}  (c = {c_val:.6f})')
    md_lines.append('')
    sub = df[df['Gamma'] == gamma].reset_index(drop=True)
    md_lines.append(_df_to_md_table(sub))
    md_lines.append('')

md_path = os.path.join(results_dir, 'RESULTS_exp2.md')
with open(md_path, 'w') as fh:
    fh.write('\n'.join(md_lines) + '\n')
print(f"Saved {md_path}")

# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

summary = {
    'experiment': 2,
    'parameters': {
        'S0':           S0,
        'T':            T,
        'sigma':        SIGMA,
        'dt':           dt,
        'A':            A,
        'k':            k,
        'n_paths':      N_PATHS,
        'n_steps':      N_STEPS,
        'gamma_values': GAMMA_VALUES,
        'c_formula':    'c = (1/gamma) * ln(1 + gamma/k)',
        'spread_formula': 'gamma * sigma^2 * (T - t) + 2c',
    },
    'using_src': USING_SRC,
    'results': all_rows,
}

json_path = os.path.join(results_dir, 'exp2_summary.json')
with open(json_path, 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f"Saved {json_path}")

print('\nExperiment 2 complete. Results saved to results/')
