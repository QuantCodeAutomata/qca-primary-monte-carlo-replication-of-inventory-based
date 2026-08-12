"""Pluggable variant functions and simulator for Experiment 3.

Provides three categories of interchangeable building blocks:
  1. Execution sampling  – Bernoulli vs exact-Poisson fill rules
  2. Price-path model    – Binomial vs Gaussian increments
  3. Edge-case handling  – log-only vs clip-zero for negative spreads

The main entry point is ``simulate_sensitivity_variant``, which accepts
string keys selecting each building block and returns terminal statistics
together with numerical-diagnostics counters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Any


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

@dataclass
class SensitivityConfig:
    S0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: float = 0.0
    X0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    n_paths: int = 1000


def compute_half_spread(gamma: float, k: float) -> float:
    """Optimal symmetric half-spread c = (1/gamma) * ln(1 + gamma/k)."""
    return (1.0 / gamma) * np.log(1.0 + gamma / k)


# ------------------------------------------------------------------ #
# Category 1 – Execution sampling
# ------------------------------------------------------------------ #

def fill_bernoulli(rand: np.ndarray, lam_dt: np.ndarray) -> np.ndarray:
    """Paper's Bernoulli approximation: fill if rand < lambda*dt."""
    return rand < lam_dt


def fill_exact_poisson(rand: np.ndarray, lam_dt: np.ndarray) -> np.ndarray:
    """Exact no-event probability: fill if rand < 1 - exp(-lambda*dt)."""
    return rand < (1.0 - np.exp(-lam_dt))


# ------------------------------------------------------------------ #
# Category 2 – Price path
# ------------------------------------------------------------------ #

def price_binomial(S: np.ndarray, sigma: float, dt: float, rand: np.ndarray) -> np.ndarray:
    """Paper's +/- sigma*sqrt(dt) binomial increment."""
    delta = sigma * np.sqrt(dt)
    return np.where(rand < 0.5, S + delta, S - delta)


def price_gaussian(S: np.ndarray, sigma: float, dt: float, rand: np.ndarray) -> np.ndarray:
    """Gaussian increment: S + sigma*sqrt(dt)*Z where Z~N(0,1)."""
    # rand is uniform, transform to normal
    # Use pre-generated normal variates instead
    return S + sigma * np.sqrt(dt) * rand  # caller passes normal variates


# ------------------------------------------------------------------ #
# Category 3 – Edge-case handling
# ------------------------------------------------------------------ #

def edge_log_only(delta: np.ndarray) -> np.ndarray:
    """Log negative deltas but leave unchanged."""
    return delta


def edge_clip_zero(delta: np.ndarray) -> np.ndarray:
    """Clip deltas below zero to zero."""
    return np.maximum(delta, 0.0)


def cap_prob(prob: np.ndarray) -> np.ndarray:
    """Cap Bernoulli probabilities at 1."""
    return np.minimum(prob, 1.0)


# ------------------------------------------------------------------ #
# Main simulation entry point
# ------------------------------------------------------------------ #

def simulate_sensitivity_variant(
    policy_name: str,
    gamma: float,
    cfg: SensitivityConfig,
    event_rule: str,
    price_rule: str,
    edge_rule: str,
    cap_prob_flag: bool,
    seed: int = 42,
) -> dict:
    """Simulate one strategy variant and return terminal stats + diagnostics.

    Parameters
    ----------
    policy_name : {'inventory', 'symmetric'}
        Quoting strategy.  Inventory uses delta_a = c - q*gamma*sigma^2*tau,
        delta_b = c + q*gamma*sigma^2*tau.  Symmetric uses delta_a = delta_b = c.
    gamma : float
        Risk-aversion coefficient.
    cfg : SensitivityConfig
        Simulation hyper-parameters.
    event_rule : {'bernoulli', 'exact_poisson'}
        Fill-event sampling rule.
    price_rule : {'binomial', 'gaussian'}
        Mid-price path model.
    edge_rule : {'log_only', 'clip_zero'}
        How negative spreads are handled before computing intensities.
    cap_prob_flag : bool
        If True, Bernoulli probabilities are capped at 1.0 before comparison.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys
        profit              – np.ndarray (n_paths,) terminal P&L
        final_q             – np.ndarray (n_paths,) terminal inventory
        neg_delta_count     – int, (path, step) pairs where delta_a<0 or delta_b<0
        lambda_dt_gt1_count – int, (path, step) pairs where lambda*dt>1 (pre-cap)
        simult_fill_count   – int, (path, step) pairs with simultaneous ask+bid fill
        total_ask_fills     – int, total ask-side executions
        total_bid_fills     – int, total bid-side executions
    """
    n_paths = cfg.n_paths
    n_steps = int(cfg.T / cfg.dt)
    sig2 = cfg.sigma ** 2
    sqrt_dt = np.sqrt(cfg.dt)
    c = compute_half_spread(gamma, cfg.k)

    # Select building-block functions
    fill_fn = fill_bernoulli if event_rule == 'bernoulli' else fill_exact_poisson
    edge_fn = edge_log_only if edge_rule == 'log_only' else edge_clip_zero

    # ------------------------------------------------------------------ #
    # Pre-generate all random numbers (common-random-number design)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)
    ask_rand = rng.random((n_paths, n_steps))
    bid_rand = rng.random((n_paths, n_steps))
    if price_rule == 'binomial':
        price_rand = rng.random((n_paths, n_steps))
    else:
        price_rand = rng.standard_normal((n_paths, n_steps))

    # ------------------------------------------------------------------ #
    # State vectors – shape (n_paths,)
    # ------------------------------------------------------------------ #
    S = np.full(n_paths, cfg.S0, dtype=float)
    q = np.full(n_paths, cfg.q0, dtype=float)
    X = np.full(n_paths, cfg.X0, dtype=float)

    # ------------------------------------------------------------------ #
    # Diagnostic accumulators
    # ------------------------------------------------------------------ #
    neg_delta_count = 0
    lambda_dt_gt1_count = 0
    simult_fill_count = 0
    total_ask_fills = 0
    total_bid_fills = 0

    # ------------------------------------------------------------------ #
    # Main time-step loop (vectorised over paths)
    # ------------------------------------------------------------------ #
    for step in range(n_steps):
        tau = cfg.T - step * cfg.dt

        # --- Compute raw spreads from current state ---
        if policy_name == 'inventory':
            skew = q * gamma * sig2 * tau
            delta_a = c - skew
            delta_b = c + skew
        else:  # symmetric
            delta_a = np.full(n_paths, c)
            delta_b = np.full(n_paths, c)

        # --- Count negative spreads BEFORE edge handling ---
        neg_delta_count += int(np.sum((delta_a < 0) | (delta_b < 0)))

        # --- Apply edge rule ---
        delta_a = edge_fn(delta_a)
        delta_b = edge_fn(delta_b)

        # --- Arrival intensities ---
        lam_a = cfg.A * np.exp(-cfg.k * delta_a)
        lam_b = cfg.A * np.exp(-cfg.k * delta_b)
        lam_a_dt = lam_a * cfg.dt
        lam_b_dt = lam_b * cfg.dt

        # --- Count lambda*dt > 1 BEFORE capping ---
        lambda_dt_gt1_count += int(np.sum((lam_a_dt > 1.0) | (lam_b_dt > 1.0)))

        # --- Optional probability cap ---
        if cap_prob_flag:
            lam_a_dt = cap_prob(lam_a_dt)
            lam_b_dt = cap_prob(lam_b_dt)

        # --- Sample fill events ---
        ask_fill = fill_fn(ask_rand[:, step], lam_a_dt)
        bid_fill = fill_fn(bid_rand[:, step], lam_b_dt)

        # --- Accumulate diagnostics ---
        simult_fill_count += int(np.sum(ask_fill & bid_fill))
        total_ask_fills += int(np.sum(ask_fill))
        total_bid_fills += int(np.sum(bid_fill))

        # --- Apply fills to cash and inventory ---
        ask_price = S + delta_a
        bid_price = S - delta_b
        X[ask_fill] += ask_price[ask_fill]
        X[bid_fill] -= bid_price[bid_fill]
        q[ask_fill] -= 1.0
        q[bid_fill] += 1.0

        # --- Price move ---
        if price_rule == 'binomial':
            S = price_binomial(S, cfg.sigma, cfg.dt, price_rand[:, step])
        else:
            S = price_gaussian(S, cfg.sigma, cfg.dt, price_rand[:, step])

    # ------------------------------------------------------------------ #
    # Terminal liquidation at mid-price
    # ------------------------------------------------------------------ #
    profit = X + q * S

    return {
        'profit': profit,
        'final_q': q,
        'neg_delta_count': neg_delta_count,
        'lambda_dt_gt1_count': lambda_dt_gt1_count,
        'simult_fill_count': simult_fill_count,
        'total_ask_fills': total_ask_fills,
        'total_bid_fills': total_bid_fills,
    }
