"""Payoffs used in the paper's numerical experiments."""

from __future__ import annotations
import torch


def call_payoff(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """European call — g(S_T) = max(S_T − K, 0).  (Table 1 test case.)"""
    return torch.clamp(S_T - K, min=0.0)


def spread_payoff(
    F_T: torch.Tensor,             # (batch, d)   terminal prices
    K: float,
    V_T: torch.Tensor | None = None,   # (batch,)  volume factor, or None ⇒ 1
) -> torch.Tensor:
    """Spread option from paper eq. (21):

        g(S_T) = V_T · ( F^1_T − (1/(d−1)) Σ_{i=2}^d F^i_T − K )^+

    For two markets (Table 2 / Case 1) this reduces to
        V_T · max(F^1_T − F^2_T − K, 0).

    If V_T is None the volume factor is taken as 1 (i.e. no volume risk).
    """
    d = F_T.shape[-1]
    if d < 2:
        raise ValueError("spread payoff needs at least 2 assets")
    short_leg = F_T[..., 1:].mean(dim=-1)     # (batch,)
    intrinsic = torch.clamp(F_T[..., 0] - short_leg - K, min=0.0)
    if V_T is None:
        return intrinsic
    return V_T * intrinsic
