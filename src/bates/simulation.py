from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def simulate_bates_terminal_multipliers(
    *,
    T: float,
    n_steps: int,
    n_paths: int,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    lambdaJ: float,
    muJ: float,
    sigmaJ: float,
    r: float = 0.04,
    seed: int = 123,
) -> np.ndarray:
    """
    Simulate Bates terminal multipliers G under the risk-neutral measure such that:
        S_T(S0) = S0 * G

    This is the same as simulating the Bates process starting at S0=1.0 and returning S_T.
    We expose it as a multiplier because it enables variance reduction / reuse for many spots.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Start at 1 so output is a multiplicative factor.
    S = np.full(n_paths, 1.0, dtype=np.float64)
    v = np.full(n_paths, max(v0, 1e-8), dtype=np.float64)

    sqdt = math.sqrt(dt)
    rho2 = math.sqrt(max(1.0 - rho * rho, 1e-12))
    kJ = math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW1 = sqdt * z1
        dW2 = sqdt * (rho * z1 + rho2 * z2)

        v_pos = np.maximum(v, 0.0)
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos + 1e-12) * dW2
        v = np.maximum(v, 1e-10)

        log_ret_diff = (r - lambdaJ * kJ - 0.5 * v_pos) * dt + np.sqrt(v_pos + 1e-12) * dW1

        Np = rng.poisson(lambdaJ * dt, size=n_paths)
        jump_log = np.zeros(n_paths, dtype=np.float64)
        idx = np.where(Np > 0)[0]
        for ii in idx:
            jump_log[ii] = rng.normal(loc=muJ * Np[ii], scale=sigmaJ * math.sqrt(Np[ii]))

        S = S * np.exp(log_ret_diff + jump_log)

    return S


def simulate_bates_paths_torch(
    n_paths: int,
    S0: float,
    T: float,
    N: int,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    lambdaJ: float,
    muJ: float,
    sigmaJ: float,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Torch simulation of Bates paths for LSTM experiments.

    Returns tensor of shape (n_paths, N+1) with prices.
    """
    dt = T / N
    sqdt = math.sqrt(dt)
    gen = torch.Generator(device=device).manual_seed(seed)

    S = torch.full((n_paths,), float(S0), dtype=torch.float32, device=device)
    v = torch.full((n_paths,), float(max(v0, 1e-8)), dtype=torch.float32, device=device)

    out = torch.zeros((n_paths, N + 1), dtype=torch.float32, device=device)
    out[:, 0] = S

    rho2 = math.sqrt(max(1.0 - rho * rho, 1e-12))
    kJ = math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0

    for j in range(N):
        z1 = torch.randn((n_paths,), generator=gen, device=device)
        z2 = torch.randn((n_paths,), generator=gen, device=device)
        dW1 = sqdt * z1
        dW2 = sqdt * (rho * z1 + rho2 * z2)

        v_pos = torch.clamp(v, min=0.0)
        v = v + kappa * (theta - v_pos) * dt + xi * torch.sqrt(v_pos + 1e-10) * dW2
        v = torch.clamp(v, min=1e-10)

        Np = torch.poisson(torch.full((n_paths,), lambdaJ * dt, device=device))
        jump = torch.zeros(n_paths, device=device)
        idx = Np > 0
        if torch.any(idx):
            jump[idx] = (
                muJ * Np[idx]
                + sigmaJ * torch.sqrt(Np[idx]) * torch.randn((int(idx.sum().item()),), generator=gen, device=device)
            )

        S = S * torch.exp((-(lambdaJ * kJ) - 0.5 * v_pos) * dt + torch.sqrt(v_pos + 1e-10) * dW1 + jump)
        out[:, j + 1] = S

    return out

