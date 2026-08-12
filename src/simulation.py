"""Vectorised discrete-time Monte Carlo simulator for the
Avellaneda-Stoikov (2008) market-making model.

Random numbers are generated once upfront and shared across strategies
(common random numbers / variance reduction).  All paths are processed
simultaneously using NumPy array operations; the outer loop iterates only
over time steps (n_steps = 200).
"""
import numpy as np
from src.config import SimConfig, compute_half_spread


def simulate_paths(
    policy_fn,
    gamma: float,
    cfg: SimConfig,
    seed: int = 42,
    record_path: bool = True,
) -> dict:
    """Run a full Monte Carlo simulation for one (policy, gamma) combination.

    Parameters
    ----------
    policy_fn : callable
        Signature: (S, q, t, tau, gamma, c) -> (delta_a, delta_b).
        Inputs S and q are numpy arrays of shape (n_paths,);
        outputs may be arrays or scalars (broadcast-safe).
    gamma : float
        Risk-aversion coefficient used by the policy and half-spread.
    cfg : SimConfig
        Simulation hyper-parameters.
    seed : int
        Seed for numpy.random.default_rng.  Using the same seed across
        strategies enables common-random-number variance reduction.
    record_path : bool
        If True, record the full state time-series for path index 0.

    Returns
    -------
    dict with keys:
        'profit'      : np.ndarray, shape (n_paths,) – terminal P&L
        'final_q'     : np.ndarray, shape (n_paths,) – terminal inventory
        'diagnostics' : dict – aggregate counters across all (path, step) pairs
        'path0'       : dict or None – illustrative time-series for path 0
    """
    n_paths = cfg.n_paths
    n_steps = int(cfg.T / cfg.dt)          # 200
    c = compute_half_spread(gamma, cfg.k)  # constant for this gamma

    # ------------------------------------------------------------------ #
    # Pre-generate ALL random numbers (common-random-number design)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)
    price_rand = rng.random((n_paths, n_steps))  # price-move draws
    ask_rand   = rng.random((n_paths, n_steps))  # ask-fill draws
    bid_rand   = rng.random((n_paths, n_steps))  # bid-fill draws

    # ------------------------------------------------------------------ #
    # State arrays – shape (n_paths,)
    # ------------------------------------------------------------------ #
    S = np.full(n_paths, cfg.S0, dtype=float)
    q = np.zeros(n_paths, dtype=float)
    X = np.zeros(n_paths, dtype=float)

    # ------------------------------------------------------------------ #
    # Illustrative-path storage (path index 0, length n_steps + 1)
    # ------------------------------------------------------------------ #
    if record_path:
        S_t   = np.empty(n_steps + 1)
        r_t   = np.empty(n_steps + 1)
        p_a_t = np.empty(n_steps + 1)
        p_b_t = np.empty(n_steps + 1)
        q_t   = np.empty(n_steps + 1)
        t_arr = np.empty(n_steps + 1)

    # ------------------------------------------------------------------ #
    # Diagnostic accumulators
    # ------------------------------------------------------------------ #
    neg_delta_count    = 0   # (path, step) pairs where delta_a<0 or delta_b<0
    lambda_dt_gt1_count = 0  # (path, step) pairs where lambda*dt > 1
    simult_fills       = 0   # (path, step) pairs with both ask & bid fill

    sqrt_dt = np.sqrt(cfg.dt)

    # ------------------------------------------------------------------ #
    # Main time-step loop (vectorised over paths)
    # ------------------------------------------------------------------ #
    for step in range(n_steps):
        t   = step * cfg.dt
        tau = cfg.T - t

        # Policy: returns scalars or arrays (n_paths,)
        delta_a, delta_b = policy_fn(S, q, t, tau, gamma, c)

        # Arrival intensities
        lambda_a = cfg.A * np.exp(-cfg.k * delta_a)
        lambda_b = cfg.A * np.exp(-cfg.k * delta_b)

        # Quote prices
        p_a = S + delta_a
        p_b = S - delta_b

        # Fill events
        ask_fill = ask_rand[:, step] < lambda_a * cfg.dt
        bid_fill = bid_rand[:, step] < lambda_b * cfg.dt

        # ---- Record illustrative path (path 0) BEFORE state updates ----
        if record_path:
            S_t[step]   = S[0]
            q_t[step]   = q[0]
            # Evaluate delta_a/delta_b for path 0 (handle scalar or array)
            da0 = float(delta_a[0]) if hasattr(delta_a, '__len__') else float(delta_a)
            db0 = float(delta_b[0]) if hasattr(delta_b, '__len__') else float(delta_b)
            p_a_t[step] = S[0] + da0
            p_b_t[step] = S[0] - db0
            r_t[step]   = (p_a_t[step] + p_b_t[step]) / 2.0
            t_arr[step] = t

        # ---- Apply fills -----------------------------------------------
        q[ask_fill] -= 1.0
        X[ask_fill] += p_a[ask_fill] if hasattr(p_a, '__len__') else p_a

        q[bid_fill] += 1.0
        X[bid_fill] -= p_b[bid_fill] if hasattr(p_b, '__len__') else p_b

        # ---- Price move (symmetric random walk) ------------------------
        price_up = price_rand[:, step] < 0.5
        S[ price_up] += cfg.sigma * sqrt_dt
        S[~price_up] -= cfg.sigma * sqrt_dt

        # ---- Diagnostics -----------------------------------------------
        da_arr = np.asarray(delta_a)
        db_arr = np.asarray(delta_b)
        neg_delta_count    += int(np.sum((da_arr < 0) | (db_arr < 0)))
        la_arr = np.asarray(lambda_a)
        lb_arr = np.asarray(lambda_b)
        lambda_dt_gt1_count += int(np.sum((la_arr * cfg.dt > 1) | (lb_arr * cfg.dt > 1)))
        simult_fills       += int(np.sum(ask_fill & bid_fill))

    # ------------------------------------------------------------------ #
    # Terminal state (t = T, tau = 0)
    # ------------------------------------------------------------------ #
    terminal_profit = X + q * S

    # ---- Record terminal point for illustrative path -------------------
    if record_path:
        S_t[n_steps]   = S[0]
        q_t[n_steps]   = q[0]
        # At tau=0 the inventory skew vanishes; quotes straddle S by c
        p_a_t[n_steps] = S[0] + c
        p_b_t[n_steps] = S[0] - c
        r_t[n_steps]   = S[0]
        t_arr[n_steps] = cfg.T

        path0 = {
            'S_t':   S_t,
            'r_t':   r_t,
            'p_a_t': p_a_t,
            'p_b_t': p_b_t,
            'q_t':   q_t,
            't':     t_arr,
        }
    else:
        path0 = None

    return {
        'profit':      terminal_profit,
        'final_q':     q,
        'diagnostics': {
            'neg_delta_count':     neg_delta_count,
            'lambda_dt_gt1_count': lambda_dt_gt1_count,
            'simult_fill_count':   simult_fills,
        },
        'path0': path0,
    }
