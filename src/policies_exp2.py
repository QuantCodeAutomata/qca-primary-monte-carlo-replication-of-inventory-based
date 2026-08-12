"""
Section 3.2 theory-consistent quoting policies for Experiment 2.

Both policies maintain the same total spread:
    delta^a + delta^b = gamma * sigma^2 * tau + 2c

The inventory policy skews the individual half-spreads based on q;
the symmetric policy keeps them equal.
"""

import numpy as np


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
