# Experiment 2 Results

> **Note:** This experiment uses theory-consistent spread `gamma*sigma^2*(T-t) + 2c`
> rather than constant `2c` from tables (Section 3.2, Avellaneda-Stoikov).

**Parameters:** S0=100.0, T=1.0, sigma=2.0, dt=0.005, A=140.0, k=1.5, n_paths=1000

**Gamma values:** [0.1, 0.01, 0.5]

**Half-spread constant:** `c = (1/gamma) * ln(1 + gamma/k)`

## gamma = 0.1  (c = 0.645385)

| Strategy | Gamma | Spread_type | mean_profit | std_profit | mean_q | std_q |
| --- | --- | --- | --- | --- | --- | --- |
| Inventory | 0.1000 | inventory-skewed | 65.0555 | 6.7690 | 0.0580 | 2.9693 |
| Symmetric | 0.1000 | symmetric-centered | 68.1100 | 13.1062 | -0.2360 | 8.0504 |

## gamma = 0.01  (c = 0.664454)

| Strategy | Gamma | Spread_type | mean_profit | std_profit | mean_q | std_q |
| --- | --- | --- | --- | --- | --- | --- |
| Inventory | 0.0100 | inventory-skewed | 68.4037 | 9.2530 | 0.0770 | 5.1471 |
| Symmetric | 0.0100 | symmetric-centered | 68.5295 | 13.5397 | -0.1800 | 8.5314 |

## gamma = 0.5  (c = 0.575364)

| Strategy | Gamma | Spread_type | mean_profit | std_profit | mean_q | std_q |
| --- | --- | --- | --- | --- | --- | --- |
| Inventory | 0.5000 | inventory-skewed | 48.6190 | 5.7456 | 0.0260 | 2.0242 |
| Symmetric | 0.5000 | symmetric-centered | 58.5763 | 11.6470 | 0.1500 | 6.8472 |

