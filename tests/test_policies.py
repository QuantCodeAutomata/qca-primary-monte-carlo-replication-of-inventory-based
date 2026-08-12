import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import compute_half_spread
from src.policies import make_inventory_policy, make_symmetric_policy
from src.policies_exp2 import make_inventory_policy_exp2, make_symmetric_policy_exp2, verify_spread_property


def test_symmetric_policy_constant_spread():
    """Symmetric policy always returns delta_a = delta_b = c."""
    gamma = 0.1
    c = compute_half_spread(gamma, 1.5)
    policy = make_symmetric_policy()
    # Test with various inventory levels
    for q in [-5, -1, 0, 1, 5]:
        da, db = policy(S=100.0, q=q, t=0.5, tau=0.5, gamma=gamma, c=c)
        assert abs(da - c) < 1e-12, f'Expected da={c}, got {da}'
        assert abs(db - c) < 1e-12, f'Expected db={c}, got {db}'


def test_inventory_policy_zero_inventory_equals_spread():
    """With q=0, inventory policy gives delta_a = delta_b = c."""
    gamma = 0.1
    c = compute_half_spread(gamma, 1.5)
    policy = make_inventory_policy(sigma=2.0)
    da, db = policy(S=100.0, q=0, t=0.0, tau=1.0, gamma=gamma, c=c)
    assert abs(da - c) < 1e-12
    assert abs(db - c) < 1e-12


def test_inventory_policy_positive_q_skews_ask():
    """With positive inventory (q>0), ask quote is tighter (smaller delta_a), bid is wider."""
    gamma = 0.1
    c = compute_half_spread(gamma, 1.5)
    policy = make_inventory_policy(sigma=2.0)
    da_pos, db_pos = policy(S=100.0, q=5, t=0.0, tau=1.0, gamma=gamma, c=c)
    da_zero, db_zero = policy(S=100.0, q=0, t=0.0, tau=1.0, gamma=gamma, c=c)
    # With positive q, delta_a decreases (more aggressive ask) and delta_b increases
    assert da_pos < da_zero
    assert db_pos > db_zero


def test_inventory_policy_negative_q_skews_bid():
    """With negative inventory (q<0), bid quote is tighter."""
    gamma = 0.1
    c = compute_half_spread(gamma, 1.5)
    policy = make_inventory_policy(sigma=2.0)
    da_neg, db_neg = policy(S=100.0, q=-5, t=0.0, tau=1.0, gamma=gamma, c=c)
    da_zero, db_zero = policy(S=100.0, q=0, t=0.0, tau=1.0, gamma=gamma, c=c)
    # With negative q, delta_b decreases and delta_a increases
    assert db_neg < db_zero
    assert da_neg > da_zero


def test_inventory_policy_approaches_symmetric_at_maturity():
    """At tau=0, inventory and symmetric policies should be identical (no time-to-maturity effect)."""
    gamma = 0.1
    c = compute_half_spread(gamma, 1.5)
    inv_policy = make_inventory_policy(sigma=2.0)
    sym_policy = make_symmetric_policy()
    # At tau=0
    da_inv, db_inv = inv_policy(S=100.0, q=5, t=1.0, tau=0.0, gamma=gamma, c=c)
    da_sym, db_sym = sym_policy(S=100.0, q=5, t=1.0, tau=0.0, gamma=gamma, c=c)
    assert abs(da_inv - da_sym) < 1e-12
    assert abs(db_inv - db_sym) < 1e-12


def test_exp2_inventory_spread_property_vectorized():
    """For exp2 inventory policy: delta_a + delta_b = gamma*sigma^2*tau + 2c."""
    gamma = 0.1
    sigma = 2.0
    c = compute_half_spread(gamma, 1.5)
    policy = make_inventory_policy_exp2(sigma=sigma)
    q_arr = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    tau = 0.5
    da_arr, db_arr = policy(S=100.0, q=q_arr, t=0.5, tau=tau, gamma=gamma, c=c)
    expected_spread = gamma * sigma**2 * tau + 2 * c
    actual_spread = da_arr + db_arr
    np.testing.assert_allclose(actual_spread, expected_spread, atol=1e-12)


def test_exp2_symmetric_spread_property():
    """For exp2 symmetric policy: delta_a + delta_b = gamma*sigma^2*tau + 2c."""
    gamma = 0.1
    sigma = 2.0
    c = compute_half_spread(gamma, 1.5)
    policy = make_symmetric_policy_exp2(sigma=sigma)
    tau = 0.5
    da, db = policy(S=100.0, q=np.array([0.0, 3.0]), t=0.5, tau=tau, gamma=gamma, c=c)
    expected_spread = gamma * sigma**2 * tau + 2 * c
    np.testing.assert_allclose(da + db, expected_spread, atol=1e-12)


def test_exp2_verify_spread_property_passes():
    """verify_spread_property should return True for valid input."""
    gamma, sigma, tau, k = 0.1, 2.0, 0.5, 1.5
    c = compute_half_spread(gamma, k)
    q = np.array([0.0, 1.0, -2.0])
    policy = make_inventory_policy_exp2(sigma=sigma)
    da, db = policy(S=100.0, q=q, t=0.5, tau=tau, gamma=gamma, c=c)
    assert verify_spread_property(da, db, gamma, sigma, tau, c)
