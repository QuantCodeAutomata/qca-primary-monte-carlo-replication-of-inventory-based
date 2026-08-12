import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import SimConfig, compute_half_spread
from src.simulation import simulate_paths
from src.policies import make_inventory_policy, make_symmetric_policy


def test_simulate_paths_returns_correct_shape():
    """simulate_paths must return profit and final_q arrays of length n_paths."""
    cfg = SimConfig(n_paths=50)
    gamma = 0.1
    policy = make_symmetric_policy()
    result = simulate_paths(policy, gamma, cfg, seed=0)
    assert 'profit' in result
    assert 'final_q' in result
    assert len(result['profit']) == 50
    assert len(result['final_q']) == 50


def test_simulate_paths_reproducible():
    """Same seed must give identical results."""
    cfg = SimConfig(n_paths=20)
    policy = make_symmetric_policy()
    r1 = simulate_paths(policy, 0.1, cfg, seed=42)
    r2 = simulate_paths(policy, 0.1, cfg, seed=42)
    np.testing.assert_array_equal(r1['profit'], r2['profit'])
    np.testing.assert_array_equal(r1['final_q'], r2['final_q'])


def test_simulate_paths_different_seeds_differ():
    """Different seeds should generally give different results."""
    cfg = SimConfig(n_paths=100)
    policy = make_symmetric_policy()
    r1 = simulate_paths(policy, 0.1, cfg, seed=1)
    r2 = simulate_paths(policy, 0.1, cfg, seed=99)
    assert not np.allclose(r1['profit'], r2['profit'])


def test_inventory_strategy_lower_inventory_std():
    """
    Core paper hypothesis: inventory strategy should have lower std(final_q)
    than symmetric strategy. Test with gamma=0.5 where effect is strongest.
    """
    cfg = SimConfig(n_paths=1000)
    gamma = 0.5
    inv_policy = make_inventory_policy(cfg.sigma)
    sym_policy = make_symmetric_policy()
    inv_result = simulate_paths(inv_policy, gamma, cfg, seed=42)
    sym_result = simulate_paths(sym_policy, gamma, cfg, seed=42)
    inv_std_q = np.std(inv_result['final_q'])
    sym_std_q = np.std(sym_result['final_q'])
    assert inv_std_q < sym_std_q, (
        f'Inventory std_q={inv_std_q:.3f} should be < symmetric std_q={sym_std_q:.3f}'
    )


def test_inventory_strategy_lower_profit_std():
    """Paper hypothesis: inventory strategy has lower std(profit) than symmetric."""
    cfg = SimConfig(n_paths=1000)
    gamma = 0.5
    inv_result = simulate_paths(make_inventory_policy(cfg.sigma), gamma, cfg, seed=42)
    sym_result = simulate_paths(make_symmetric_policy(), gamma, cfg, seed=42)
    assert np.std(inv_result['profit']) < np.std(sym_result['profit'])


def test_symmetric_strategy_zero_inventory_bias():
    """Symmetric strategy should have mean final inventory near zero (no systematic skew)."""
    cfg = SimConfig(n_paths=1000)
    sym_result = simulate_paths(make_symmetric_policy(), 0.1, cfg, seed=42)
    mean_q = np.mean(sym_result['final_q'])
    # Should be close to zero but may have noise
    assert abs(mean_q) < 2.0, f'Expected near-zero mean inventory, got {mean_q:.3f}'


def test_positive_mean_profit():
    """Both strategies should produce positive mean terminal profit (market makers earn spread)."""
    cfg = SimConfig(n_paths=500)
    for policy in [make_inventory_policy(2.0), make_symmetric_policy()]:
        result = simulate_paths(policy, 0.1, cfg, seed=42)
        assert np.mean(result['profit']) > 0, 'Expected positive mean profit'


def test_diagnostics_returned():
    """Diagnostics dict should contain expected keys."""
    cfg = SimConfig(n_paths=50)
    result = simulate_paths(make_inventory_policy(2.0), 0.1, cfg, seed=0)
    assert 'diagnostics' in result
    diag = result['diagnostics']
    assert 'neg_delta_count' in diag
    assert 'lambda_dt_gt1_count' in diag
    assert 'simult_fill_count' in diag
