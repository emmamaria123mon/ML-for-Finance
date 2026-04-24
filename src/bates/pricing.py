from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np

from .simulation import simulate_bates_terminal_multipliers


def _bates_char_func(
    u: np.ndarray,
    *,
    S0: float,
    T: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    lambdaJ: float,
    muJ: float,
    sigmaJ: float,
    r: float,
) -> np.ndarray:
    """
    Characteristic function φ(u) = E[e^{i u log(S_T)}] for Bates model.

    Uses Heston CF with risk-neutral drift adjusted by jump compensator and
    multiplies by Merton jump CF term.
    """
    iu = 1j * u
    x0 = math.log(float(S0))

    # Jump compensator k_J = E[e^J - 1], J ~ N(muJ, sigmaJ^2)
    k_j = math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
    r_eff = float(r) - float(lambdaJ) * k_j

    a = float(kappa) * float(theta)
    b = float(kappa)
    sigma = float(xi)
    rho_ = float(rho)

    # Heston CF core terms
    d = np.sqrt((rho_ * sigma * iu - b) ** 2 + (sigma ** 2) * (u * u + iu))
    g = (b - rho_ * sigma * iu - d) / (b - rho_ * sigma * iu + d)

    exp_neg_dT = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g

    c = (
        iu * (x0 + r_eff * T)
        + (a / (sigma ** 2))
        * ((b - rho_ * sigma * iu - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )
    d_term = (
        (b - rho_ * sigma * iu - d) / (sigma ** 2)
        * ((1.0 - exp_neg_dT) / one_minus_g_exp)
    )
    phi_heston = np.exp(c + d_term * float(v0))

    # Jump CF factor for compound Poisson with lognormal jump size
    jump_cf = np.exp(
        float(lambdaJ)
        * T
        * (np.exp(iu * float(muJ) - 0.5 * (float(sigmaJ) ** 2) * (u * u)) - 1.0)
    )
    return phi_heston * jump_cf


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


def bates_call_prices_carr_madan(
    *,
    S0: float,
    K_array: np.ndarray,
    T: float,
    params: Tuple[float, float, float, float, float, float, float, float],
    r: float = 0.04,
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> np.ndarray:
    """
    Price a strip of call options under Bates using Carr-Madan damped inversion.

    Notes:
    - This is a Fourier inversion implementation (Carr-Madan style) using a
      fixed v-grid and trapezoid integration, which is typically smoother/faster
      than MC during calibration loops.
    - n_v is forced to an even integer >= 128 for stable integration.
    """
    if T <= 0:
        return np.maximum(float(S0) - np.asarray(K_array, dtype=float), 0.0)

    kappa, theta, xi, rho, v0, lambdaJ, muJ, sigmaJ = params
    K = np.asarray(K_array, dtype=float)
    if np.any(K <= 0):
        raise ValueError("K_array must contain strictly positive strikes.")

    n_v = int(max(128, n_v))
    if n_v % 2 == 1:
        n_v += 1

    v = np.linspace(0.0, float(v_max), n_v, dtype=float)
    dv = v[1] - v[0]

    u = v - 1j * (float(alpha) + 1.0)
    phi = _bates_char_func(
        u,
        S0=float(S0),
        T=float(T),
        kappa=float(kappa),
        theta=float(theta),
        xi=float(xi),
        rho=float(rho),
        v0=float(v0),
        lambdaJ=float(lambdaJ),
        muJ=float(muJ),
        sigmaJ=float(sigmaJ),
        r=float(r),
    )

    den = (alpha * alpha + alpha - v * v) + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-float(r) * float(T)) * phi / den

    # Trapezoidal weights (half on edges)
    w = np.ones_like(v)
    w[0] = 0.5
    w[-1] = 0.5

    k_log = np.log(K)
    phase = np.exp(-1j * np.outer(v, k_log))  # (n_v, n_k)
    integ = np.real((w[:, None] * psi[:, None]) * phase).sum(axis=0) * dv

    calls = np.exp(-alpha * k_log) * integ / math.pi
    # Numerical guardrails
    intrinsic = np.maximum(float(S0) - K * math.exp(-float(r) * float(T)), 0.0)
    return np.maximum(calls, intrinsic)


def bates_call_prices(
    *,
    method: Literal["mc", "carr_madan"] = "mc",
    S0: float,
    K_array: np.ndarray,
    T: float,
    params: Tuple[float, float, float, float, float, float, float, float],
    r: float = 0.04,
    # MC kwargs
    n_steps: int = 40,
    n_paths: int = 25_000,
    seed: int = 123,
    # Carr-Madan kwargs
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> np.ndarray:
    """
    Unified Bates strip pricer with selectable backend.
    """
    if method == "mc":
        return bates_call_prices_mc(
            S0=S0,
            K_array=K_array,
            T=T,
            params=params,
            n_steps=n_steps,
            n_paths=n_paths,
            r=r,
            seed=seed,
        )
    if method == "carr_madan":
        return bates_call_prices_carr_madan(
            S0=S0,
            K_array=K_array,
            T=T,
            params=params,
            r=r,
            alpha=alpha,
            v_max=v_max,
            n_v=n_v,
        )
    raise ValueError(f"Unknown Bates pricing method: {method!r}")


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


def bates_call_price_carr_madan(
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
    r: float = 0.04,
    alpha: float = 1.5,
    v_max: float = 200.0,
    n_v: int = 4096,
) -> float:
    """Scalar Bates call price via Carr-Madan inversion."""
    K_arr = np.array([float(K)], dtype=float)
    out = bates_call_prices_carr_madan(
        S0=float(S0),
        K_array=K_arr,
        T=float(T),
        params=(kappa, theta, xi, rho, v0, lambdaJ, muJ, sigmaJ),
        r=float(r),
        alpha=float(alpha),
        v_max=float(v_max),
        n_v=int(n_v),
    )
    return float(out[0])

