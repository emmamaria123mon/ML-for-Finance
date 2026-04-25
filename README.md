# Deep Learning for Discrete-Time Hedging in Incomplete Markets

This repository implements a PyTorch reproduction of the deep hedging framework from the paper `Deep Learning for Discrete-Time Hedging in Incomplete Markets` (arXiv: `1902.05287`), and extends it to a market-calibrated SPY hedging study under a Bates stochastic volatility jump-diffusion model.

The repo combines:

- reproduction experiments for the main deep hedging architectures in the paper,
- notebook-based analysis for the reported tables and loss-function comparisons,
- a reusable experiment/training codebase under `src/`,
- an extension that calibrates a Bates model to SPY option data and evaluates hedging performance on simulated and realized paths.

The paper PDF followed in this project is included in the repository root as [1902.05287v4 (1).pdf](/c:/Users/Michael/Code/ML-for-Finance/1902.05287v4%20%281%29.pdf).

## What This Repo Covers

The project reproduces the paper's global deep hedging setup built around:

- discrete-time self-financing hedging,
- neural-network policies for hedge ratios,
- training by minimizing terminal hedging error,
- comparison across feedforward and recurrent architectures,
- alternative loss functions and Pareto-style risk trade-offs.

In addition to the baseline reproduction, the repo extends the framework to:

- calibrate Bates parameters from SPY option snapshots,
- simulate SPY dynamics under stochastic volatility with jumps,
- train deep hedging models on those calibrated dynamics,
- compare learned strategies with Black-Scholes and Bates delta benchmarks,
- run a realized backtest with optional transaction costs and liquidity limits.

More precisely, the reproduction in this repo focuses on the paper's global algorithm across the GBM settings behind Tables 1 and 2, the alternative-loss study from Section 6.2, and the Pareto-frontier experiment from Section 7. The local dynamic-programming comparison associated with the paper's Table 3 is not part of this repo.

## Repository Layout

```text
.
|- notebooks/
|  |- 01_full_reproduction.ipynb
|  |- 02_table2_improved.ipynb
|  `- 04_heston_spy_calibrated_extension.ipynb
|- src/
|  |- architectures.py
|  |- benchmarks.py
|  |- experiment.py
|  |- losses.py
|  |- payoffs.py
|  |- simulators.py
|  |- training.py
|  `- bates/
|- data/
|  `- option_snapshots/
|- results_table1/
|- results_table2/
|- results_bates/
|- pyproject.toml
|- requirements.txt
|- uv.lock
`- 1902.05287v4 (1).pdf
```

## Notebooks

### `notebooks/01_full_reproduction.ipynb`

Main reproduction notebook for the baseline paper experiments. It uses the shared `Config + run_experiment(...)` workflow from `src/experiment.py` and covers:

- Table 1 style Black-Scholes call experiments,
- comparisons across feedforward and recurrent architectures,
- loss-function variants,
- Pareto frontier style experiments.

Outputs are written primarily to `results_table1/` and `results_table2/`.

The baseline implementation is a PyTorch port of the paper's original TensorFlow-style workflow.

### `notebooks/02_table2_improved.ipynb`

Focused follow-up analysis for the Table 2 style experiments, using saved results from the baseline workflow.

### `notebooks/04_heston_spy_calibrated_extension.ipynb`

Extension notebook for the SPY study. It:

- loads SPY option snapshot data,
- calibrates a Bates model,
- generates Monte Carlo paths,
- trains hedging models under the calibrated dynamics,
- evaluates benchmark deltas,
- writes summary outputs and backtest files to `results_bates/`.

## Code Structure

### Core modules

- `src/architectures.py`: feedforward and LSTM hedging architectures used in the paper-style experiments.
- `src/training.py`: implementation of the global training loop with checkpointing on evaluation loss.
- `src/experiment.py`: experiment harness built around a `Config` dataclass and `run_experiment(...)`.
- `src/simulators.py`: GBM, correlated multi-asset GBM, and other simulation utilities plus normalization helpers.
- `src/payoffs.py`: option payoff definitions.
- `src/losses.py`: training losses for hedging error minimization.
- `src/benchmarks.py`: benchmark pricing and delta utilities.

### Bates extension

- `src/bates/simulation.py`: Bates path simulation.
- `src/bates/pricing.py`: Bates option pricing utilities.
- `src/bates/deltas.py`: benchmark delta calculations.
- `src/bates/workflow.py`: calibration, path-factory, plotting, and experiment helpers.
- `src/bates/execution.py`: transaction-cost and liquidity execution helper for realized hedging.

## Data and Results

### Input data

The repository currently includes SPY option snapshot data under `data/option_snapshots/`, including quote tables and metadata used by the extension notebook.

### Saved outputs

- `results_table1/`: baseline reproduction summaries, hedging PnL arrays, and generated figures.
- `results_table2/`: Table 2 style summaries and PnL arrays.
- `results_bates/`: calibrated Bates parameters, SPY hedging summaries, Pareto outputs, and realized backtest CSV files.

Representative saved artifacts include:

- `results_table1/table1_summary.json`
- `results_table2/table2_summary.json`
- `results_bates/bates_spy_summary.json`
- `results_bates/bates_calibrated_params.json`

## Environment Setup

Python `>=3.11` is required.

### Using `uv`

```bash
uv sync
```

For notebook support:

```bash
uv sync --extra notebook
```

### Using `pip`

```bash
pip install -r requirements.txt
```

Core dependencies include `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`, and `yfinance`.

## Reproducibility Notes

The baseline experiments follow the report configuration closely:

- optimizer: Adam,
- initial learning rate: `1e-3`,
- training iterations: `20,000`,
- batch size: `50`,
- gradient clipping: unit norm,
- training pool: `50,000` simulated paths,
- evaluation pool: `100,000` paths,
- normalization pool: `100,000` paths.

Input normalization is computed once on a separate reference pool and then frozen for training and evaluation.

## How To Run

Launch Jupyter from the project root so `src/` resolves cleanly:

```bash
jupyter lab
```

Then open the notebooks in `notebooks/`.

Recommended order:

1. `01_full_reproduction.ipynb`
2. `02_table2_improved.ipynb`
3. `04_heston_spy_calibrated_extension.ipynb`

The extension notebook is more self-contained than the reproduction notebooks, but running from the project root is still the safest setup.

## Methodology Notes

- The implementation follows the paper's global deep hedging idea: learn the full hedging policy directly over discrete trading times.
- Models output hedge positions step by step and optimize terminal hedging error.
- Price normalization is applied to network inputs only; PnL is always computed on raw prices.
- The training loop uses a fixed simulated training pool with mini-batch sampling and periodic evaluation checkpointing.
- The repo includes both baseline paper-style benchmarks and a calibrated real-market extension.

## Headline Results

From the saved summaries currently in the repo:

- Table 1 reproduction: the Black-Scholes delta benchmark achieves MSE `1.2897e-05`, and the augmented LSTM reaches `1.4573e-05`.
- Table 2 reproduction: the classical LSTM reaches MSE `5.6557e-05` on the Black-Scholes call setup and `7.6167e-05` on the two-markets spread setup.
- SPY Bates extension: the saved calibration uses 231 quotes, and the summary compares Black-Scholes delta, Bates delta, Classical LSTM, and Augmented LSTM on common evaluation paths.
- Realized SPY backtest with transaction costs and liquidity constraints: the Augmented LSTM attains the lowest one-step residual MSE at `2.2853`, ahead of Black-Scholes delta at `2.3984`, Classical LSTM at `2.4255`, and Bates delta at `8.4567`.

For full metrics and saved parameters, see the JSON files in `results_table1/`, `results_table2/`, and `results_bates/`.

At a high level, the report's main conclusion is not that deep hedging dominates everywhere. Under Bates-simulated dynamics, delta-style benchmarks remain very strong. Under the realized frictional SPY backtest, the learned hedgers, especially the augmented LSTM, appear more robust to model misspecification and trading frictions.

## Extending The Project

The cleanest extension point is `src/experiment.py`. New experiments can usually be added by changing:

- the path generator,
- the payoff function,
- the architecture dictionary,
- the loss function,
- liquidity or transaction cost settings.

This design keeps the training loop reusable across both the paper reproduction and new market-model experiments.

## Scope

This repository focuses on reproducing the neural deep hedging components and on extending them to a calibrated SPY setting. It is not a full reproduction of every experiment or external dependency used in the paper's broader numerical study.

## Reference

Fecamp, S., Mikael, J., and Warin, X. `Deep Learning for Discrete-Time Hedging in Incomplete Markets`. arXiv:1902.05287.
