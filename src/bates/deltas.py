from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .simulation import simulate_bates_terminal_multipliers


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta_np(S, K: float, T, sigma):
    """
    Vectorized Black–Scholes delta for calls.

    S and T can be scalars or arrays; sigma can be scalar or broadcastable array.
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    S, T = np.broadcast_arrays(S, T)

    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full_like(S, float(sigma_arr))
    else:
        sigma_arr = np.broadcast_to(sigma_arr, S.shape)

    out = np.empty_like(S, dtype=float)
    bad = (T <= 0.0) | (sigma_arr <= 0.0)
    out[bad] = np.where(S[bad] > K, 1.0, 0.0)

    good = ~bad
    if np.any(good):
        Tg = T[good]
        Sg = S[good]
        sg = sigma_arr[good]
        d1 = (np.log(Sg / K) + 0.5 * sg * sg * Tg) / (sg * np.sqrt(Tg))
        out[good] = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / np.sqrt(2.0)))
    return out


def bates_delta_mc_fd_vec_spots(
    S_vec,
    K: float,
    T: float,
    *,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    lambdaJ: float,
    muJ: float,
    sigmaJ: float,
    r: float = 0.04,
    eps_rel: float = 1e-3,
    n_paths_mc: int = 3500,
    batch_size: int = 600,
    seed: int = 2027,
) -> np.ndarray:
    """
    Bates delta via MC finite-difference, variance-reduced by reusing terminal multipliers G.

    Uses multiplicative dynamics: S_T(S0) = S0 * G, so the same G can price for many S0.
    """
    S_vec = np.asarray(S_vec, dtype=float)
    if S_vec.size == 0:
        return np.array([], dtype=float)
    if T <= 1e-10:
        return (S_vec > K).astype(float)

    n_steps_fd = max(6, int(round(365 * T)))
    G = simulate_bates_terminal_multipliers(
        T=float(T),
        n_steps=int(n_steps_fd),
        n_paths=int(n_paths_mc),
        kappa=float(kappa),
        theta=float(theta),
        xi=float(xi),
        rho=float(rho),
        v0=float(v0),
        lambdaJ=float(lambdaJ),
        muJ=float(muJ),
        sigmaJ=float(sigmaJ),
        r=float(r),
        seed=int(seed),
    )

    out = np.empty_like(S_vec)
    for i0 in range(0, len(S_vec), batch_size):
        i1 = min(i0 + batch_size, len(S_vec))
        Sb = S_vec[i0:i1]
        eps = np.maximum(eps_rel * Sb, 1e-4)
        Cup = np.maximum((Sb[:, None] + eps[:, None]) * G[None, :] - K, 0.0).mean(axis=1)
        Cdn = np.maximum((Sb[:, None] - eps[:, None]) * G[None, :] - K, 0.0).mean(axis=1)
        out[i0:i1] = (Cup - Cdn) / (2.0 * eps)

    return np.clip(out, 0.0, 1.0)

