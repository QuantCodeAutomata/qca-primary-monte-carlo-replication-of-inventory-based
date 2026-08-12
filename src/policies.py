"""Policy factories for the Avellaneda-Stoikov market-making model.

Each factory returns a callable with signature:
    policy(S, q, t, tau, gamma, c) -> (delta_a, delta_b)

where S and q may be numpy arrays (vectorised over paths) or scalars.
delta_a is the half-spread added to S for the ask quote,
delta_b is the half-spread subtracted from S for the bid quote.
"""


def make_inventory_policy(sigma: float = 2.0):
    """Optimal inventory-skewing policy (Avellaneda-Stoikov 2008).

    The reservation price is r = S - q * gamma * sigma^2 * tau.
    Quotes are centred on r with half-spread c:
        delta_a = r + c - S = c - q * gamma * sigma^2 * tau
        delta_b = S - (r - c) = c + q * gamma * sigma^2 * tau
    """
    sig2 = sigma ** 2

    def policy(S, q, t, tau, gamma, c):
        skew = q * gamma * sig2 * tau
        delta_a = c - skew
        delta_b = c + skew
        return delta_a, delta_b

    return policy


def make_symmetric_policy():
    """Constant symmetric spread policy (no inventory adjustment).

    Posts at the theoretical half-spread c regardless of inventory,
    used as the baseline comparator for variance-reduction analysis.
    """
    def policy(S, q, t, tau, gamma, c):
        return c, c

    return policy
