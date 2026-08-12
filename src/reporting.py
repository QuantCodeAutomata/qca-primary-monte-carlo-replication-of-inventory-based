"""Reporting utilities: statistics, table construction, and file output."""
import numpy as np
import pandas as pd


def compute_stats(profits: np.ndarray, final_qs: np.ndarray) -> dict:
    """Return mean and std of terminal P&L and final inventory."""
    return {
        'mean_profit': float(np.mean(profits)),
        'std_profit':  float(np.std(profits)),
        'mean_q':      float(np.mean(final_qs)),
        'std_q':       float(np.std(final_qs)),
    }


def build_results_table(results: list) -> pd.DataFrame:
    """Build summary DataFrame from a list of result records.

    Each record is a dict with keys:
        Strategy, Gamma, Spread, profit (array), final_q (array).

    Returns a DataFrame with columns:
        Strategy, Gamma, Spread, mean_profit, std_profit, mean_q, std_q.
    """
    rows = []
    for rec in results:
        stats = compute_stats(rec['profit'], rec['final_q'])
        rows.append({
            'Strategy':    rec['Strategy'],
            'Gamma':       rec['Gamma'],
            'Spread':      rec['Spread'],
            'mean_profit': stats['mean_profit'],
            'std_profit':  stats['std_profit'],
            'mean_q':      stats['mean_q'],
            'std_q':       stats['std_q'],
        })
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame) -> None:
    """Pretty-print the results DataFrame to stdout."""
    float_cols = ['Spread', 'mean_profit', 'std_profit', 'mean_q', 'std_q']
    fmt = {col: '{:.4f}'.format for col in float_cols}
    print(df.to_string(index=False, formatters=fmt))


def save_results_markdown(df: pd.DataFrame, path: str) -> None:
    """Write the results DataFrame as a Markdown table to *path*."""
    cols = df.columns.tolist()
    float_cols = {'Spread', 'mean_profit', 'std_profit', 'mean_q', 'std_q'}

    def _fmt(col, val):
        return f'{val:.4f}' if col in float_cols else str(val)

    header = '| ' + ' | '.join(cols) + ' |'
    sep    = '|' + '|'.join([' :---: ' for _ in cols]) + '|'
    lines  = [header, sep]
    for _, row in df.iterrows():
        line = '| ' + ' | '.join(_fmt(c, row[c]) for c in cols) + ' |'
        lines.append(line)

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
