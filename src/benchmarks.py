"""Analytical benchmarks used to validate the neural networks."""

from __future__ import annotations
import numpy as np
import torch
from scipy.stats import norm as _N


# ---------------------------------------------------------------------------
# Black-Scholes analytics
# ---------------------------------------------------------------------------
def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * _N.cdf(d1) - K * np.exp(-r * T) * _N.cdf(d2))


def bs_delta(S: np.ndarray, K: float, T_rem: np.ndarray, sigma: float,
             r: float = 0.0) -> np.ndarray:
    """BS delta N(d1). Vectorises over S and T_rem."""
    eps = 1e-10
    d1 = (np.log(S / K + eps) + (r + 0.5 * sigma ** 2) * T_rem) \
         / (sigma * np.sqrt(T_rem) + eps)
    return _N.cdf(d1)


def bs_hedge_mse(
    paths: torch.Tensor,
    K: float, sigma: float, T: float, dt: float, r: float = 0.0,
):
    """Monte-Carlo MSE of the Black-Scholes delta hedge for a call option.

    IMPORTANT: collects the BS premium at t=0 so E[Y] ≈ 0.  Without it the
    MSE would be ≈ price² instead of the variance-of-PnL target (~1.6e-5).
    """
    N_steps = paths.shape[1] - 1
    paths_np = paths.cpu().numpy()
    S0 = float(paths_np[0, 0])
    prem = bs_call_price(S0, K, T, sigma, r)

    # BS delta evaluated in one vectorised pass
    t_rem = np.arange(N_steps, 0, -1) * dt                # (N_steps,)
    t_rem = np.clip(t_rem, 1e-10, None)
    deltas = bs_delta(
        paths_np[:, :-1],                                 # (n, N)
        K,
        t_rem[None, :],                                   # broadcast
        sigma,
        r,
    )
    dS = paths_np[:, 1:] - paths_np[:, :-1]               # (n, N) real increments
    pnl = prem + (deltas * dS).sum(axis=1)
    g_ST = np.maximum(paths_np[:, -1] - K, 0.0)
    err = pnl - g_ST
    return float((err ** 2).mean()), err, prem


def unhedged_variance(
    paths: torch.Tensor, K: float, V_T: torch.Tensor | None = None,
    payoff_fn=None,
) -> float:
    """Variance of g(S_T) − E[g(S_T)]. Used as a sanity / upper-bound check
    (Table 3's "Unhedged Portfolio" row in the paper)."""
    if payoff_fn is None:
        from .payoffs import call_payoff
        payoff_fn = lambda F: call_payoff(F[..., -1] if F.dim() == 3 else F[..., -1], K)
    # Caller will typically just compute var themselves; this helper is
    # convenience only.
    return float(payoff_fn(paths).var(unbiased=False).item())
