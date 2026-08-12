# Monte Carlo Replication: Inventory-Based vs Symmetric Market-Making

This repository replicates and extends the Monte Carlo simulation study from the
Avellaneda-Stoikov (2008) market-making paper, comparing inventory-dependent and
symmetric quoting strategies in a synthetic limit-order-book environment.

## Project Structure

```
qca-primary-monte-carlo-replication-of-inventory-based/
├── data/                  # Placeholder (no real market data used)
├── src/                   # Core simulation modules
│   ├── config.py          # Global parameters (SimConfig dataclass)
│   ├── simulation.py      # Vectorized Monte Carlo engine
│   ├── policies.py        # Quoting policy functions (exp1 table-matching)
│   ├── policies_exp2.py   # Theory-consistent policies (exp2 Section 3.2)
│   └── reporting.py       # Stats, tables, markdown output
├── exp/                   # Experiment scripts
│   ├── run_exp1.py        # Exp 1: Primary table-matching replication
│   ├── run_exp2.py        # Exp 2: Theory-consistent full formulas
│   └── run_exp3.py        # Exp 3: Discretization sensitivity analysis
├── tests/                 # Pytest unit and integration tests
│   ├── test_config.py
│   ├── test_simulation.py
│   ├── test_policies.py
│   └── test_sensitivity.py
├── results/               # Output: tables, plots, RESULTS.md
└── README.md
```

## Experiments

### Experiment 1: Primary Monte Carlo Replication
Replicates published Monte Carlo comparison using table-matching constant half-spread
c = (1/γ) ln(1 + γ/k). Tests γ ∈ {0.01, 0.1, 0.5} with 1000 paths each.

**Run:** `python exp/run_exp1.py`

### Experiment 2: Theory-Consistent Robustness Check
Implements the full Section 3.2 approximate quoting formulas with time-varying
spread γσ²(T-t) + 2c. Verifies qualitative conclusions hold under literal formulas.

**Run:** `python exp/run_exp2.py`

### Experiment 3: Discretization Sensitivity Study
Tests sensitivity to: Bernoulli vs exact-Poisson arrivals, binomial vs Gaussian
price increments, edge-case handling (negative deltas, lambda·dt > 1).

**Run:** `python exp/run_exp3.py`

## Parameters
| Parameter | Value |
|-----------|-------|
| S₀        | 100   |
| T         | 1     |
| σ         | 2     |
| dt        | 0.005 |
| A         | 140   |
| k         | 1.5   |
| Paths     | 1000  |
| γ values  | 0.01, 0.1, 0.5 |

## Results
See [results/RESULTS.md](results/RESULTS.md) for full tables and findings.

## Setup
```bash
pip install -r requirements.txt
python exp/run_exp1.py
python exp/run_exp2.py
python exp/run_exp3.py
pytest tests/
```

## References
Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
*Quantitative Finance*, 8(3), 217-224.
