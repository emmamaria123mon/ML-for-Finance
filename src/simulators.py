"""Market simulators.

Covers the three price processes that appear in the paper:

1. Geometric Brownian motion (used for the Table 1 Black-Scholes call).
2. d-dimensional correlated GBM (used for the Table 2 two-markets spread).
3. Mean-reverting forward with stochastic volume risk (used in Section 5 and
   Table 3 for the electricity-style variance cases).

All simulators return *raw* prices — i.e. the quantities that appear in the
paper's P&L equation (3). A separate helper computes fixed batch-normalisation
statistics (paper §4.3) that are used only at the network input, never inside
the P&L.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1.  Single-asset GBM   (Black-Scholes setup, Table 1)
# ---------------------------------------------------------------------------
def simulate_gbm(
    n_paths: int,
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    N: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simulate GBM with exact log-Euler discretisation.

    Returns a tensor of shape (n_paths, N+1). Column 0 = S0, column -1 = S_T.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        Z = torch.randn(n_paths, N, device=device, generator=gen)
    else:
        Z = torch.randn(n_paths, N, device=device)
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S = torch.cat(
        [torch.zeros(n_paths, 1, device=device), torch.cumsum(log_inc, dim=1)],
        dim=1,
    )
    return S0 * torch.exp(log_S)


# ---------------------------------------------------------------------------
# 2.  Multi-asset correlated GBM   (two-markets spread, Table 2)
# ---------------------------------------------------------------------------
def simulate_multi_gbm(
    n_paths: int,
    S0: np.ndarray,        # shape (d,)
    mu: np.ndarray,        # shape (d,)
    sigma: np.ndarray,     # shape (d,)
    corr: np.ndarray,      # shape (d, d) — correlation matrix
    dt: float,
    N: int,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Correlated GBM for d assets.

    Returns shape (n_paths, N+1, d). S_{paths, j, i} is price of asset i at
    step j.
    """
    d = len(S0)
    L = np.linalg.cholesky(corr).astype(np.float32)
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        Z = torch.randn(n_paths, N, d, device=device, generator=gen)
    else:
        Z = torch.randn(n_paths, N, d, device=device)
    # Correlate shocks: Z_corr = Z @ L.T
    L_t = torch.from_numpy(L).to(device)
    Z_corr = Z @ L_t.T

    sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    mu_t    = torch.as_tensor(mu,    device=device, dtype=torch.float32)
    S0_t    = torch.as_tensor(S0,    device=device, dtype=torch.float32)

    log_inc = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * Z_corr
    log_S = torch.cat(
        [torch.zeros(n_paths, 1, d, device=device), torch.cumsum(log_inc, dim=1)],
        dim=1,
    )
    return S0_t * torch.exp(log_S)


# ---------------------------------------------------------------------------
# 3.  Mean-reverting forward with stochastic volume   (Section 5, Table 3)
# ---------------------------------------------------------------------------
@dataclass
class MRForwardParams:
    """Parameters for one forward contract (paper eq. 2, §2.1).

    F_t = F0 * exp( -sigma^2 * (e^{-2a(T-t)} - e^{-2aT}) / (4 a)
                    + e^{-a(T-t)} * W_hat_t )
    where W_hat_t is an OU process driven by dW with mean-reversion a and
    vol sigma, all measured *per day* in the paper's convention.
    """
    F0: float
    sigma: float
    a: float        # mean-reversion (per day in paper units)


def simulate_mr_forwards(
    n_paths: int,
    params: list,           # list[MRForwardParams], length d
    corr: np.ndarray,       # (d, d) tradable-asset correlation
    dt: float,
    N: int,
    T: float,               # maturity in same units as dt
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return shape (n_paths, N+1, d). Closed-form discretisation of the
    paper's eq. (2) for each tradable forward."""
    d = len(params)
    L = np.linalg.cholesky(corr).astype(np.float32)
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        Z = torch.randn(n_paths, N, d, device=device, generator=gen)
    else:
        Z = torch.randn(n_paths, N, d, device=device)
    L_t = torch.from_numpy(L).to(device)
    Z_corr = Z @ L_t.T

    # Build W_hat_t = σ ∫₀^t e^{-a(t-s)} dW_s  via exact AR(1) recursion.
    W_hat = torch.zeros(n_paths, N + 1, d, device=device)
    for j in range(N):
        for i in range(d):
            p = params[i]
            decay = np.exp(-p.a * dt)
            var   = (p.sigma ** 2) * (1.0 - np.exp(-2 * p.a * dt)) / (2 * p.a)
            W_hat[:, j + 1, i] = (
                decay * W_hat[:, j, i]
                + np.sqrt(var) * Z_corr[:, j, i]
            )

    # Price path F^i_t
    F = torch.zeros(n_paths, N + 1, d, device=device)
    t_grid = torch.arange(N + 1, device=device, dtype=torch.float32) * dt
    for i in range(d):
        p = params[i]
        expo_det = (
            -(p.sigma ** 2)
            * (torch.exp(-2 * p.a * (T - t_grid)) - np.exp(-2 * p.a * T))
            / (4 * p.a)
        )
        decay_factor = torch.exp(-p.a * (T - t_grid))
        # broadcast decay along paths
        exponent = expo_det[None, :] + decay_factor[None, :] * W_hat[:, :, i]
        F[:, :, i] = p.F0 * torch.exp(exponent)
    return F


@dataclass
class VolumeOUParams:
    """Non-tradable volume process (paper eq. 1, §2.1)."""
    V0: float
    V_hat: float      # long-run mean V̂
    sigma: float
    a: float


def simulate_volume(
    n_paths: int,
    p: VolumeOUParams,
    dt: float,
    N: int,
    correlated_noise: Optional[torch.Tensor] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """OU process for the volume risk V_t. Returns (n_paths, N+1).

    If `correlated_noise` (shape n_paths × N) is supplied, it is used as
    the driving Gaussian increment (unit variance per step) — useful when V
    is correlated with the tradable forwards.
    """
    if correlated_noise is None:
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
            Z = torch.randn(n_paths, N, device=device, generator=gen)
        else:
            Z = torch.randn(n_paths, N, device=device)
    else:
        Z = correlated_noise

    decay = np.exp(-p.a * dt)
    var   = (p.sigma ** 2) * (1.0 - np.exp(-2 * p.a * dt)) / (2 * p.a) if p.a > 0 else (p.sigma ** 2) * dt

    V = torch.zeros(n_paths, N + 1, device=device)
    V[:, 0] = p.V0
    for j in range(N):
        V[:, j + 1] = (
            p.V_hat
            + decay * (V[:, j] - p.V_hat)
            + np.sqrt(var) * Z[:, j]
        )
    return V


# ---------------------------------------------------------------------------
# Batch-normalisation stats (paper §4.3: fixed stats on 100k reference paths)
# ---------------------------------------------------------------------------
def compute_norm_stats(paths: torch.Tensor, eps: float = 1e-8
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-time-step mean and std. Works for 2-D (single asset) or 3-D
    (multi-asset) tensors — reduces over the first (paths) dimension."""
    mean = paths.mean(dim=0, keepdim=True)
    std  = paths.std(dim=0, keepdim=True).clamp(min=eps)
    return mean, std


def normalize_with(paths: torch.Tensor,
                   mean: torch.Tensor,
                   std: torch.Tensor) -> torch.Tensor:
    """Apply the fixed normalisation  S̃_t = (S_t - μ_t) / σ_t."""
    return (paths - mean) / std
