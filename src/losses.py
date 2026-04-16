"""Loss functions from the paper.

All losses act on the *hedging error*  Y = X_T − g(S_T).
(For the Pareto loss with transaction costs Y also includes the TC term.)
"""

from __future__ import annotations
import torch


def mse(Y: torch.Tensor) -> torch.Tensor:
    """Eq. (4): E[Y²]."""
    return Y.pow(2).mean()


def asymmetric(Y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Eq. (5): E[(1+α) Y²·1_{Y≤0} + Y²·1_{Y≥0}].

    α > 0 penalises losses more than gains; α = 0 reduces to MSE.
    """
    sq = Y.pow(2)
    loss_side = (sq * (Y <= 0).float()) * (1.0 + alpha)
    gain_side =  sq * (Y >= 0).float()
    return (loss_side + gain_side).mean()


def moment_2_4(Y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Eq. (6): E[Y²·1_{Y≥0} + α · Y⁴·1_{Y≤0}], α ≥ 1.

    Squared gains + quartic penalty on losses — strongly suppresses the left
    tail.
    """
    sq = Y.pow(2)
    q4 = Y.pow(4)
    gain_side = sq * (Y >= 0).float()
    loss_side = alpha * q4 * (Y <= 0).float()
    return (loss_side + gain_side).mean()


def pareto_mean_variance(Y: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Eq. (22) of the paper:

        d^α(X, g) = (1 − α) · E[Y]  +  α · √E[Y²]

    Used for the Pareto-frontier training in §7.  `alpha` is a scalar or
    a per-sample tensor in [0, 1].  At α=1 it reduces to std(Y); at α=0
    it only minimises the expected TC.
    """
    # We use per-batch scalar α (the paper samples one α per batch).
    a = alpha.mean() if alpha.dim() else alpha
    return (1.0 - a) * Y.mean() + a * Y.pow(2).mean().sqrt()
