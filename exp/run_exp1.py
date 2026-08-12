#!/usr/bin/env python3
"""Experiment 1: Primary Monte Carlo replication – table-matching constant spread.

Both strategies are run with the SAME seed (42) so that common random numbers
(CRN) are exploited: price paths and fill draws are identical, isolating the
pure effect of inventory skewing from sampling noise.

Run from the project root:
    python exp/run_exp1.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import SimConfig, compute_half_spread
from src.simulation import simulate_paths
from src.policies import make_inventory_policy, make_symmetric_policy
from src.reporting import (
    build_results_table,
    print_results_table,
    save_results_markdown,
    compute_stats,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _res(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_experiment(cfg: SimConfig) -> tuple[list, dict]:
    """Run both strategies for all gamma values.

    Returns
    -------
    records : list of result-record dicts (consumed by build_results_table)
    raw     : nested dict keyed by (strategy, gamma) with raw simulation output
    """
    records = []
    raw: dict = {}

    for gamma in cfg.gamma_values:
        c = compute_half_spread(gamma, cfg.k)

        # ---- Inventory strategy ----------------------------------------
        inv_policy = make_inventory_policy(cfg.sigma)
        inv_out    = simulate_paths(inv_policy, gamma, cfg, seed=42, record_path=True)
        records.append({
            'Strategy': 'Inventory',
            'Gamma':    gamma,
            'Spread':   c,
            'profit':   inv_out['profit'],
            'final_q':  inv_out['final_q'],
        })
        raw[('Inventory', gamma)] = inv_out

        # ---- Symmetric (baseline) strategy -----------------------------
        # Same seed → same price paths and fill draws (CRN variance reduction)
        sym_policy = make_symmetric_policy()
        sym_out    = simulate_paths(sym_policy, gamma, cfg, seed=42, record_path=False)
        records.append({
            'Strategy': 'Symmetric',
            'Gamma':    gamma,
            'Spread':   c,
            'profit':   sym_out['profit'],
            'final_q':  sym_out['final_q'],
        })
        raw[('Symmetric', gamma)] = sym_out

    return records, raw


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_per_gamma_tables(df: pd.DataFrame, cfg: SimConfig) -> None:
    """Print one sub-table per gamma value (Table 1 / 2 / 3)."""
    for table_num, gamma in enumerate(cfg.gamma_values, start=1):
        sub = df[df['Gamma'] == gamma].copy()
        print(f'\nTable {table_num}: gamma = {gamma}')
        print('-' * 60)
        print_results_table(sub)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_illustrative_path(path0: dict, gamma: float) -> None:
    """Figure 1 – single-path price, reservation price, and quotes."""
    t   = path0['t']
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, path0['S_t'],   label='S (mid price)',       color='steelblue',  linewidth=1.4)
    ax.plot(t, path0['r_t'],   label='r (reservation)',     color='darkorange', linewidth=1.2, linestyle='--')
    ax.plot(t, path0['p_a_t'], label='$p^a$ (ask quote)',   color='seagreen',   linewidth=1.0, linestyle=':')
    ax.plot(t, path0['p_b_t'], label='$p^b$ (bid quote)',   color='crimson',    linewidth=1.0, linestyle=':')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(
        f'Inventory Strategy: Price, Reservation Price, and Quotes (gamma={gamma})',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_res('exp1_path_illustration.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved: {_res("exp1_path_illustration.png")}')


def plot_pnl_histograms(raw: dict, cfg: SimConfig) -> None:
    """Figures 2-4 – terminal P&L histograms for each gamma."""
    for gamma in cfg.gamma_values:
        inv_profit = raw[('Inventory', gamma)]['profit']
        sym_profit = raw[('Symmetric', gamma)]['profit']

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(inv_profit, bins=50, alpha=0.6, label='Inventory', color='steelblue')
        ax.hist(sym_profit, bins=50, alpha=0.6, label='Symmetric', color='darkorange')

        ax.set_xlabel('Terminal P&L', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Terminal P&L Distribution (gamma={gamma})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fname = f'exp1_pnl_hist_gamma_{gamma}.png'
        fig.savefig(_res(fname), dpi=150)
        plt.close(fig)
        print(f'  Saved: {_res(fname)}')


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_json_summary(df: pd.DataFrame) -> None:
    summary: dict = {}
    for _, row in df.iterrows():
        key = f"{row['Strategy']}_gamma_{row['Gamma']}"
        summary[key] = {
            'gamma':       float(row['Gamma']),
            'spread':      float(row['Spread']),
            'mean_profit': float(row['mean_profit']),
            'std_profit':  float(row['std_profit']),
            'mean_q':      float(row['mean_q']),
            'std_q':       float(row['std_q']),
        }
    json_path = _res('exp1_summary.json')
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'  Saved: {json_path}')


def save_diagnostics(raw: dict, cfg: SimConfig) -> None:
    diag_rows = []
    for gamma in cfg.gamma_values:
        for strategy in ('Inventory', 'Symmetric'):
            d = raw[(strategy, gamma)]['diagnostics']
            diag_rows.append({
                'Strategy':           strategy,
                'Gamma':              gamma,
                'neg_delta_count':    d['neg_delta_count'],
                'lambda_dt_gt1':      d['lambda_dt_gt1_count'],
                'simult_fills':       d['simult_fill_count'],
            })
    diag_df = pd.DataFrame(diag_rows)
    diag_path = _res('exp1_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print(f'  Saved: {diag_path}')
    print('\nDiagnostics summary:')
    print(diag_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = SimConfig()

    print('=' * 60)
    print('Experiment 1: Monte Carlo Market-Making Replication')
    print(f'n_paths={cfg.n_paths}, n_steps={int(cfg.T/cfg.dt)}, seed=42 (CRN)')
    print('=' * 60)

    # Run simulations
    print('\nRunning simulations …')
    records, raw = run_experiment(cfg)

    # Build summary table
    df = build_results_table(records)

    # Print per-gamma tables
    print_per_gamma_tables(df, cfg)

    # Save CSV
    csv_path = _res('exp1_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # Save Markdown
    md_path = _res('RESULTS_exp1.md')
    save_results_markdown(df, md_path)
    print(f'Saved: {md_path}')

    # Save JSON
    save_json_summary(df)

    # Diagnostics
    save_diagnostics(raw, cfg)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print('\nGenerating plots …')

    # Figure 1 – illustrative single path (inventory, gamma=0.1)
    path0 = raw[('Inventory', 0.1)]['path0']
    plot_illustrative_path(path0, gamma=0.1)

    # Figures 2-4 – P&L histograms per gamma
    plot_pnl_histograms(raw, cfg)

    print('\nExperiment 1 complete. Results saved to results/')


if __name__ == '__main__':
    main()
