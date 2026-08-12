import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import SimConfig, compute_half_spread


def test_simconfig_defaults():
    """Test SimConfig has correct default parameters matching paper."""
    cfg = SimConfig()
    assert cfg.S0 == 100.0
    assert cfg.T == 1.0
    assert cfg.sigma == 2.0
    assert cfg.dt == 0.005
    assert cfg.q0 == 0
    assert cfg.X0 == 0.0
    assert cfg.A == 140.0
    assert cfg.k == 1.5
    assert cfg.n_paths == 1000
    # n_steps should be 200
    assert int(cfg.T / cfg.dt) == 200


def test_half_spread_gamma_01():
    """Half-spread for gamma=0.1 should yield total spread ~1.29."""
    c = compute_half_spread(0.1, 1.5)
    total_spread = 2 * c
    # Paper states ~1.29 for gamma=0.1
    assert 1.2 < total_spread < 1.4, f'Expected ~1.29, got {total_spread:.4f}'


def test_half_spread_gamma_001():
    """Half-spread for gamma=0.01 should yield total spread ~1.33."""
    c = compute_half_spread(0.01, 1.5)
    total_spread = 2 * c
    assert 1.25 < total_spread < 1.4, f'Expected ~1.33, got {total_spread:.4f}'


def test_half_spread_gamma_05():
    """Half-spread for gamma=0.5 should yield total spread ~1.15."""
    c = compute_half_spread(0.5, 1.5)
    total_spread = 2 * c
    assert 1.0 < total_spread < 1.3, f'Expected ~1.15, got {total_spread:.4f}'


def test_half_spread_formula():
    """Verify formula: c = (1/gamma)*ln(1+gamma/k)."""
    gamma, k = 0.1, 1.5
    expected = (1 / gamma) * np.log(1 + gamma / k)
    result = compute_half_spread(gamma, k)
    assert abs(result - expected) < 1e-12


def test_half_spread_decreases_with_gamma():
    """Larger risk aversion yields smaller half-spread (due to formula)."""
    c_01 = compute_half_spread(0.1, 1.5)
    c_05 = compute_half_spread(0.5, 1.5)
    # Higher gamma -> smaller spread
    assert c_05 < c_01
