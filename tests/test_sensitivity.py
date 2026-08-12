import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.sensitivity import (
    SensitivityConfig,
    compute_half_spread,
    fill_bernoulli,
    fill_exact_poisson,
    price_binomial,
    edge_log_only,
    edge_clip_zero,
    cap_prob,
    simulate_sensitivity_variant,
)


def test_fill_bernoulli_below_threshold():
    """Bernoulli fill: all fills when rand < lam_dt."""
    rand = np.array([0.01, 0.05, 0.1])
    lam_dt = np.array([0.5, 0.5, 0.5])
    result = fill_bernoulli(rand, lam_dt)
    assert result.all()


def test_fill_bernoulli_above_threshold():
    """Bernoulli fill: no fills when rand > lam_dt."""
    rand = np.array([0.9, 0.95, 0.99])
    lam_dt = np.array([0.1, 0.1, 0.1])
    result = fill_bernoulli(rand, lam_dt)
    assert not result.any()


def test_fill_exact_poisson_vs_bernoulli_small_lam():
    """For small lambda*dt, exact Poisson and Bernoulli should give similar results."""
    np.random.seed(42)
    rand = np.random.uniform(0, 1, 10000)
    lam_dt = np.full(10000, 0.01)  # Small intensity
    bern = fill_bernoulli(rand, lam_dt).mean()
    exact = fill_exact_poisson(rand, lam_dt).mean()
    # Should be close since lambda*dt << 1
    assert abs(bern - exact) < 0.005, f'Bernoulli={bern:.4f} vs Exact={exact:.4f}'


def test_exact_poisson_bounded():
    """Exact Poisson probability must be in [0, 1]."""
    lam_dt = np.array([0.0, 0.1, 1.0, 5.0, 100.0])
    prob = 1.0 - np.exp(-lam_dt)
    assert (prob >= 0).all() and (prob <= 1.0 + 1e-10).all()


def test_edge_clip_zero_removes_negatives():
    """edge_clip_zero should eliminate all negative values."""
    delta = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    result = edge_clip_zero(delta)
    assert (result >= 0).all()
    assert result[3] == 1.0
    assert result[4] == 3.0


def test_edge_log_only_unchanged():
    """edge_log_only should return delta unchanged."""
    delta = np.array([-1.0, 0.5, 2.0])
    result = edge_log_only(delta)
    np.testing.assert_array_equal(result, delta)


def test_cap_prob_clamps_at_one():
    """cap_prob should clamp values > 1 to 1 and leave others unchanged."""
    prob = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    result = cap_prob(prob)
    assert (result <= 1.0).all()
    assert result[0] == 0.0
    assert result[1] == 0.5


def test_simulate_sensitivity_baseline_positive_profit():
    """Baseline variant should produce positive mean profit."""
    cfg = SensitivityConfig(n_paths=200)
    result = simulate_sensitivity_variant(
        policy_name='inventory',
        gamma=0.1,
        cfg=cfg,
        event_rule='bernoulli',
        price_rule='binomial',
        edge_rule='log_only',
        cap_prob_flag=False,
        seed=42,
    )
    assert np.mean(result['profit']) > 0


def test_simulate_sensitivity_diagnostics_non_negative():
    """Diagnostic counts must be non-negative integers."""
    cfg = SensitivityConfig(n_paths=100)
    result = simulate_sensitivity_variant(
        policy_name='inventory',
        gamma=0.5,
        cfg=cfg,
        event_rule='bernoulli',
        price_rule='binomial',
        edge_rule='log_only',
        cap_prob_flag=False,
        seed=0,
    )
    assert result['neg_delta_count'] >= 0
    assert result['lambda_dt_gt1_count'] >= 0
    assert result['simult_fill_count'] >= 0


def test_simulate_sensitivity_exact_poisson_returns_profit():
    """Exact Poisson variant should return valid profit array."""
    cfg = SensitivityConfig(n_paths=100)
    result = simulate_sensitivity_variant(
        policy_name='symmetric',
        gamma=0.1,
        cfg=cfg,
        event_rule='exact_poisson',
        price_rule='binomial',
        edge_rule='log_only',
        cap_prob_flag=False,
        seed=1,
    )
    assert len(result['profit']) == 100
    assert np.all(np.isfinite(result['profit']))
