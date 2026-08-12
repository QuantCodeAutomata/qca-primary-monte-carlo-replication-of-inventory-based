from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    S0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: int = 0
    X0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    n_paths: int = 1000
    gamma_values: tuple = (0.1, 0.01, 0.5)


def compute_half_spread(gamma: float, k: float) -> float:
    """Optimal symmetric half-spread from Avellaneda-Stoikov (2008).

    Derived as the reservation-price half-spread that maximises expected
    utility: c = (1/gamma) * log(1 + gamma/k).
    """
    return (1.0 / gamma) * np.log(1.0 + gamma / k)
