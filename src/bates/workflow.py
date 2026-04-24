from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution

from ..simulators import compute_norm_stats
from .deltas import bates_delta_mc_fd_vec_spots, bs_call_price, bs_delta_np
from .pricing import bates_call_price_mc, bates_call_prices
from .simulation import simulate_bates_paths_torch


BatesParams = tuple[float, float, float, float, float, float, float, float]


def simulate_bates_tensor(
    n_paths: int,
    *,
    spot: float,
    horizon: float,
    n_steps: int,
    params: BatesParams,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = params
    return simulate_bates_paths_torch(
        n_paths=n_paths,
        S0=spot,
        T=horizon,
        N=n_steps,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
        lambdaJ=lambda_j,
        muJ=mu_j,
        sigmaJ=sigma_j,
        seed=seed,
        device=device,
    ).unsqueeze(-1)


def build_bates_paths_factory(
    *,
    n_train: int,
    n_eval: int,
    n_norm: int,
    spot: float,
    horizon: float,
    n_steps: int,
    params: BatesParams,
    train_seed: int = 101,
    eval_seed: int = 102,
    norm_seed: int = 999,
    device: str = "cpu",
) -> Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    def factory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_raw = simulate_bates_tensor(
            n_train,
            spot=spot,
            horizon=horizon,
            n_steps=n_steps,
            params=params,
            seed=train_seed,
            device=device,
        )
        eval_raw = simulate_bates_tensor(
            n_eval,
            spot=spot,
            horizon=horizon,
            n_steps=n_steps,
            params=params,
            seed=eval_seed,
            device=device,
        )
        norm_raw = simulate_bates_tensor(
            n_norm,
            spot=spot,
            horizon=horizon,
            n_steps=n_steps,
            params=params,
            seed=norm_seed,
            device=device,
        )
        mean, std = compute_norm_stats(norm_raw)
        return train_raw, eval_raw, mean, std

    return factory


def run_bates_preflight(
    *,
    spot: float,
    horizon: float,
    n_steps: int,
    params: BatesParams,
    device: str = "cpu",
    train_size: int = 256,
    eval_size: int = 128,
    norm_size: int = 128,
    train_seed: int = 11,
    eval_seed: int = 12,
    norm_seed: int = 13,
) -> dict[str, object]:
    train_raw = simulate_bates_tensor(
        train_size,
        spot=spot,
        horizon=horizon,
        n_steps=n_steps,
        params=params,
        seed=train_seed,
        device=device,
    )
    eval_raw = simulate_bates_tensor(
        eval_size,
        spot=spot,
        horizon=horizon,
        n_steps=n_steps,
        params=params,
        seed=eval_seed,
        device=device,
    )
    norm_raw = simulate_bates_tensor(
        norm_size,
        spot=spot,
        horizon=horizon,
        n_steps=n_steps,
        params=params,
        seed=norm_seed,
        device=device,
    )
    mean, std = compute_norm_stats(norm_raw)

    assert train_raw.shape[1] == n_steps + 1 and train_raw.shape[-1] == 1
    assert eval_raw.shape[1] == n_steps + 1 and eval_raw.shape[-1] == 1
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert float(std.min()) > 0.0

    return {
        "train_shape": tuple(train_raw.shape),
        "eval_shape": tuple(eval_raw.shape),
        "norm_shape": tuple(norm_raw.shape),
    }


def plot_training_histories(histories: dict[str, dict[str, Sequence[float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for model_name, hist in histories.items():
        axes[0].plot(hist["train"], label=model_name, alpha=0.9)
        axes[1].plot(hist["test_iters"], hist["test"], marker="o", ms=3, label=model_name, alpha=0.9)

    axes[0].set_title("LSTM training loss (per iteration)")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("LSTM eval MSE (checkpointed)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("eval MSE of Y")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def price_surface_by_tau(
    quotes_df: pd.DataFrame,
    *,
    spot: float,
    params: BatesParams,
    hedge_horizon: float,
    method: str = "mc",
    n_steps: int = 40,
    n_paths: int = 12_000,
    seed: int = 123,
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> np.ndarray:
    out = np.zeros(len(quotes_df), dtype=float)
    for group_index, (tau, group) in enumerate(quotes_df.groupby("tau", sort=True)):
        strikes = group["strike"].to_numpy(dtype=float)
        model_prices = bates_call_prices(
            method=method,
            S0=float(spot),
            K_array=strikes,
            T=float(tau),
            params=params,
            r=0.0,
            n_steps=max(6, int(round(n_steps * float(tau) / max(hedge_horizon, 1e-6)))),
            n_paths=n_paths,
            seed=seed + group_index,
            alpha=alpha,
            v_max=v_max,
            n_v=n_v,
        )
        out[group.index.to_numpy()] = model_prices
    return out


def attach_surface_prices(
    quotes_df: pd.DataFrame,
    *,
    spot: float,
    call_prices: Sequence[float],
    call_column: str,
    price_column: str,
) -> pd.DataFrame:
    out = quotes_df.copy()
    out[call_column] = np.asarray(call_prices, dtype=float)
    out[price_column] = np.where(
        out["option_type"].eq("call"),
        out[call_column],
        out[call_column] - float(spot) + out["strike"].to_numpy(dtype=float),
    )
    return out


def surface_mse(observed: Sequence[float], modeled: Sequence[float]) -> float:
    obs = np.asarray(observed, dtype=float)
    fit = np.asarray(modeled, dtype=float)
    return float(np.mean((fit - obs) ** 2))


def weighted_surface_objective(
    params: Sequence[float],
    *,
    quotes_df: pd.DataFrame,
    spot: float,
    hedge_horizon: float,
    method: str = "mc",
    n_steps: int = 40,
    n_paths: int = 7000,
    seed: int = 777,
    penalty_scale: float = 10.0,
    invalid_penalty: float = 1e9,
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> float:
    kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = [float(v) for v in params]
    if theta <= 0 or xi <= 0 or v0 <= 0 or kappa <= 0 or abs(rho) >= 0.999:
        return invalid_penalty
    if lambda_j < 0 or sigma_j <= 0:
        return invalid_penalty

    feller_penalty = max(0.0, xi * xi - 2.0 * kappa * theta)
    model_prices = price_surface_by_tau(
        quotes_df,
        spot=spot,
        params=(kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j),
        hedge_horizon=hedge_horizon,
        method=method,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        alpha=alpha,
        v_max=v_max,
        n_v=n_v,
    )
    target = quotes_df["target_price"].to_numpy(dtype=float)
    weight = quotes_df["weight"].to_numpy(dtype=float)
    err = weight * (model_prices - target) ** 2
    return float(err.mean() + penalty_scale * feller_penalty)


def save_bates_params(
    path: str | Path,
    *,
    method: str,
    params: BatesParams,
    objective: float,
    alpha: float,
    v_max: float,
    n_v: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "kappa": float(params[0]),
        "theta": float(params[1]),
        "xi": float(params[2]),
        "rho": float(params[3]),
        "v0": float(params[4]),
        "lambdaJ": float(params[5]),
        "muJ": float(params[6]),
        "sigmaJ": float(params[7]),
        "objective": float(objective),
        "cm_alpha": float(alpha),
        "cm_v_max": float(v_max),
        "cm_n_v": int(n_v),
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_bates_params(path: str | Path) -> tuple[BatesParams, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = (
        float(payload["kappa"]),
        float(payload["theta"]),
        float(payload["xi"]),
        float(payload["rho"]),
        float(payload["v0"]),
        float(payload["lambdaJ"]),
        float(payload["muJ"]),
        float(payload["sigmaJ"]),
    )
    objective = float(payload.get("objective", float("nan")))
    return params, objective


def calibrate_bates_model(
    *,
    quotes_df: pd.DataFrame,
    spot: float,
    hedge_horizon: float,
    bounds: Sequence[tuple[float, float]],
    method: str,
    params_path: str | Path | None = None,
    use_saved: bool = False,
    save_result: bool = True,
    objective_n_steps: int = 40,
    objective_n_paths: int = 7000,
    objective_seed: int = 777,
    maxiter: int = 40,
    popsize: int = 12,
    seed: int = 42,
    disp: bool = False,
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> tuple[BatesParams, float]:
    if use_saved and params_path is not None and Path(params_path).is_file():
        return load_bates_params(params_path)

    result = differential_evolution(
        lambda x: weighted_surface_objective(
            x,
            quotes_df=quotes_df,
            spot=spot,
            hedge_horizon=hedge_horizon,
            method=method,
            n_steps=objective_n_steps,
            n_paths=objective_n_paths,
            seed=objective_seed,
            alpha=alpha,
            v_max=v_max,
            n_v=n_v,
        ),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=False,
        disp=disp,
    )
    params = tuple(float(v) for v in result.x)
    objective = float(result.fun)

    if save_result and params_path is not None:
        save_bates_params(
            params_path,
            method=method,
            params=params,
            objective=objective,
            alpha=alpha,
            v_max=v_max,
            n_v=n_v,
        )

    return params, objective


def to_call_equivalent_prices(
    prices: Sequence[float],
    strikes: Sequence[float],
    option_types: Sequence[str],
    *,
    spot: float,
) -> np.ndarray:
    prices_arr = np.asarray(prices, dtype=float)
    strikes_arr = np.asarray(strikes, dtype=float)
    option_types_arr = np.asarray(option_types)
    return np.where(option_types_arr == "call", prices_arr, prices_arr + float(spot) - strikes_arr)


def bs_call_price_r(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)


def implied_vol_from_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    max_iter: int = 100,
) -> float:
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    clipped_price = min(max(float(price), intrinsic + 1e-10), S)
    low, high = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if bs_call_price_r(S, K, T, mid, r=r) > clipped_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def run_delta_hedge(
    paths: np.ndarray,
    *,
    strike: float,
    horizon: float,
    premium: float,
    delta_fn: Callable[[np.ndarray, float, int], np.ndarray],
) -> dict[str, object]:
    path_array = np.asarray(paths, dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1).")

    n_paths, n_cols = path_array.shape
    n_steps = n_cols - 1
    dt = float(horizon) / max(n_steps, 1)

    current_spot = path_array[:, 0]
    delta_prev = np.asarray(delta_fn(current_spot, float(horizon), 0), dtype=float)
    cash = np.full(n_paths, float(premium), dtype=float) - delta_prev * current_spot

    for step in range(1, n_steps + 1):
        tau = max(float(horizon) - step * dt, 0.0)
        spot_t = path_array[:, step]
        if step < n_steps:
            delta_new = np.asarray(delta_fn(spot_t, tau, step), dtype=float)
        else:
            delta_new = np.zeros_like(spot_t)
        cash -= (delta_new - delta_prev) * spot_t
        delta_prev = delta_new

    payoff = np.maximum(path_array[:, -1] - float(strike), 0.0)
    portfolio = cash + delta_prev * path_array[:, -1]
    error = portfolio - payoff
    return {
        "premium": float(premium),
        "portfolio": portfolio,
        "payoff": payoff,
        "error": error,
        "mse": float(np.mean(error ** 2)),
        "mean": float(np.mean(error)),
        "std": float(np.std(error)),
    }


def evaluate_bs_hedge(
    paths: np.ndarray,
    *,
    spot: float,
    strike: float,
    horizon: float,
    sigma: float,
) -> dict[str, object]:
    premium = bs_call_price(float(spot), float(strike), float(horizon), float(sigma))
    return run_delta_hedge(
        paths,
        strike=strike,
        horizon=horizon,
        premium=premium,
        delta_fn=lambda spots, tau, step: bs_delta_np(spots, strike, tau, sigma),
    )


def evaluate_bates_fd_hedge(
    paths: np.ndarray,
    *,
    spot: float,
    strike: float,
    horizon: float,
    params: BatesParams,
    price_n_paths: int = 3500,
    initial_delta_n_paths: int = 3500,
    rebalance_delta_n_paths: int = 3000,
    seed: int = 301,
    r: float = 0.0,
) -> dict[str, object]:
    kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = params
    premium = bates_call_price_mc(
        S0=float(spot),
        K=float(strike),
        T=float(horizon),
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
        lambdaJ=lambda_j,
        muJ=mu_j,
        sigmaJ=sigma_j,
        n_steps=max(6, int(round(365 * horizon))),
        n_paths=price_n_paths,
        r=r,
        seed=seed,
    )

    def delta_fn(spots: np.ndarray, tau: float, step: int) -> np.ndarray:
        n_paths_mc = initial_delta_n_paths if step == 0 else rebalance_delta_n_paths
        return bates_delta_mc_fd_vec_spots(
            spots,
            strike,
            tau,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            lambdaJ=lambda_j,
            muJ=mu_j,
            sigmaJ=sigma_j,
            r=r,
            n_paths_mc=n_paths_mc,
            seed=seed + step,
        )

    return run_delta_hedge(
        paths,
        strike=strike,
        horizon=horizon,
        premium=premium,
        delta_fn=delta_fn,
    )


def bates_fd_delta_series(
    spots: Sequence[float],
    taus: Sequence[float],
    *,
    strike: float,
    params: BatesParams,
    n_paths_mc: int = 12_000,
    seed: int = 1234,
    r: float = 0.0,
) -> np.ndarray:
    spot_arr = np.asarray(spots, dtype=float)
    tau_arr = np.asarray(taus, dtype=float)
    if spot_arr.shape != tau_arr.shape:
        raise ValueError("spots and taus must have the same shape.")

    kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j = params
    deltas = np.empty_like(spot_arr)
    for index, (spot_i, tau_i) in enumerate(zip(spot_arr, tau_arr)):
        deltas[index] = bates_delta_mc_fd_vec_spots(
            np.array([spot_i], dtype=float),
            strike,
            float(tau_i),
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            lambdaJ=lambda_j,
            muJ=mu_j,
            sigmaJ=sigma_j,
            r=r,
            n_paths_mc=n_paths_mc,
            seed=seed + index,
        )[0]
    return deltas


def evaluate_frictional_delta_hedge(
    paths: np.ndarray,
    *,
    strike: float,
    horizon: float,
    premium: float,
    delta_fn: Callable[[np.ndarray, float, int], np.ndarray],
    tc_per_unit: float,
    liq_step: float,
) -> dict[str, object]:
    path_array = np.asarray(paths, dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1).")

    n_paths, n_cols = path_array.shape
    n_steps = n_cols - 1
    dt = float(horizon) / max(n_steps, 1)

    cash = np.full(n_paths, float(premium), dtype=float)
    delta_prev = np.zeros(n_paths, dtype=float)
    tc_paid = np.zeros(n_paths, dtype=float)

    for step in range(n_steps):
        tau = max(float(horizon) - step * dt, 1e-8)
        spot_t = path_array[:, step]
        target_delta = np.asarray(delta_fn(spot_t, tau, step), dtype=float)
        delta_new = delta_prev + np.clip(target_delta - delta_prev, -liq_step, liq_step)
        trade = delta_new - delta_prev
        tc_paid += float(tc_per_unit) * np.abs(trade)
        cash += delta_new * (path_array[:, step + 1] - spot_t)
        delta_prev = delta_new

    payoff = np.maximum(path_array[:, -1] - float(strike), 0.0)
    error_before_cost = cash - payoff
    error_after_cost = error_before_cost - tc_paid
    return {
        "error": error_before_cost,
        "error_after_cost": error_after_cost,
        "tc_paid": tc_paid,
        "mean_tc": float(np.mean(tc_paid)),
        "rmse_err": float(np.sqrt(np.mean(error_before_cost ** 2))),
        "mse": float(np.mean(error_after_cost ** 2)),
    }
