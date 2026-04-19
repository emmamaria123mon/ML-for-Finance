from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .simulation import simulate_bates_terminal_multipliers


def bates_call_prices_mc(
    *,
    S0: float,
    K_array: np.ndarray,
    T: float,
    params: Tuple[float, float, float, float, float, float, float, float],
    n_steps: int = 40,
    n_paths: int = 25_000,
    r: float = 0.04,
    seed: int = 123,
) -> np.ndarray:
    """
    Price a strip of call options under Bates by Monte Carlo.

    Returns an array of prices with same length as K_array.
    """
    kappa, theta, xi, rho, v0, lambdaJ, muJ, sigmaJ = params
    G = simulate_bates_terminal_multipliers(
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
        lambdaJ=lambdaJ,
        muJ=muJ,
        sigmaJ=sigmaJ,
        r=r,
        seed=seed,
    )
    ST = float(S0) * G
    payoff = np.maximum(ST[:, None] - K_array[None, :], 0.0)
    return payoff.mean(axis=0)


def bates_call_price_mc(
    *,
    S0: float,
    K: float,
    T: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    lambdaJ: float,
    muJ: float,
    sigmaJ: float,
    n_steps: int = 40,
    n_paths: int = 25_000,
    r: float = 0.04,
    seed: int = 123,
) -> float:
    """Scalar Bates MC call price."""
    K_arr = np.array([float(K)], dtype=float)
    out = bates_call_prices_mc(
        S0=float(S0),
        K_array=K_arr,
        T=float(T),
        params=(kappa, theta, xi, rho, v0, lambdaJ, muJ, sigmaJ),
        n_steps=int(n_steps),
        n_paths=int(n_paths),
        r=float(r),
        seed=int(seed),
    )
    return float(out[0])

