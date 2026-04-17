# ML-for-Finance: Table 2 Replication + SPY Extension

This project reproduces key results from:

- Fecamp, Mikael, Warin, *Deep learning for discrete-time hedging in incomplete markets* (arXiv:1902.05287v4)

and extends the setup to real SPY + listed options data without full retraining.

## What Is In This Repo

- `notebooks/02_table2_replication.ipynb`  
  Paper-style Table 2 setup (Classical LSTM vs Augmented LSTM, BS call + 2-market spread).
- `results_table2/table2_summary.json`  
  Saved Table 2 summary metrics (used as baseline; avoids retraining).
- `notebooks/03_spy_options_extension.ipynb`  
  Real-data extension using SPY prices and SPY options chain, with no full retrain by default.
- `src/`  
  Core simulation/training code (`Config`, `run_experiment`, architectures, losses).

## Table 2 Replication (Paper Baseline)

The notebook `02_table2_replication.ipynb` matches the paper's Table 2 experimental design:

- Horizon/grid: `T=3/12`, `dt=1/360`, `N=90`
- Training setup: Adam `1e-3`, batch `50`, up to `20,000` iterations
- Architectures: Classical LSTM and Augmented LSTM
- Criterion: MSE of hedging error `Y` (paper Eq. 4)

Saved baseline values in `results_table2/table2_summary.json`:

- BS call
  - Classical LSTM: `5.495e-05` (paper `5.73e-05`)
  - Augmented LSTM: `4.617e-05` (paper `3.97e-05`)
- 2-markets spread
  - Classical LSTM: `8.168e-05` (paper `3.64e-04`)
  - Augmented LSTM: `8.419e-05` (paper `1.11e-04`)

These are the reference numbers reused in the extension notebook.

## SPY + Options Extension (No Full Retraining)

The notebook `03_spy_options_extension.ipynb` does:

1. Loads saved Table 2 baseline (`results_table2/table2_summary.json`)
2. Pulls real SPY market data (price history + option chain)
3. Builds ATM implied-vol proxy from option mid price
4. Simulates short-horizon paths via block-bootstrap of SPY returns
5. Evaluates discrete delta-hedging error MSE
6. Adds transaction-cost stress (0/1/5/10 bps)

### Data Source Logic

The notebook tries sources in this order:

1. `yfinance` (preferred)
2. Stooq prices + Yahoo options API
3. Yahoo chart + Yahoo options API

If all fail, it raises a combined error and suggests installing notebook extras.

## Does It Work?

Yes, based on current executed outputs in `03_spy_options_extension.ipynb`:

- Data loaded from `yfinance`
- SPY history rows: `8361`
- Spot: `710.32`
- Chosen expiry: `2026-05-08`
- ATM IV proxy: `0.1783`
- Real extension MSE (no transaction costs): `6.618e+01`
- Transaction-cost stress:
  - `0 bps`: `6.618e+01`
  - `1 bps`: `6.720e+01`
  - `5 bps`: `7.171e+01`
  - `10 bps`: `7.839e+01`

### Important Interpretation Note

Do **not** directly compare `6.618e+01` to Table 2's `~1e-5` as if they were same-scale metrics.
They are not apples-to-apples because:

- Table 2 uses normalized synthetic settings (`S0=K=1`) and neural strategies.
- Extension uses real SPY spot/strike levels (hundreds), shorter real maturity, and a classical delta-hedge benchmark.

So the extension should be interpreted as:

- a practical real-data stress test,
- not a direct numeric replication target.

## Quick Start

From repo root:

```bash
uv sync --extra notebook
```

Open and run in order:

1. `notebooks/02_table2_replication.ipynb` (or reuse saved `results_table2`)
2. `notebooks/03_spy_options_extension.ipynb`

## If Data Download Fails

- Re-run the notebook once (transient API/network failures are common).
- Ensure dependencies are installed (`uv sync --extra notebook`).
- If Yahoo endpoints return `401`, rely on `yfinance` path (primary).

## Next Recommended Improvement

For tighter paper alignment, add a scaled metric to the extension (for example `MSE(Y / spot)` or variance ratio) so comparisons across synthetic and real regimes are easier to interpret.

