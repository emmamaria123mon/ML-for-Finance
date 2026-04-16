"""Network architectures from Section 4 of the paper.

All four classes share the same forward-pass contract:

    out = model(paths_raw, paths_norm, liq=..., alpha=None)

    paths_raw  : (batch, N+1, d) raw prices — used ONLY for the P&L sum
                  (batch, N+1) is also accepted for d = 1
    paths_norm : (batch, N+1, d) pre-normalised prices — used ONLY as input
                 features to the network
    liq        : scalar liquidity per step (paper's l^i).  For uncapped
                 training use liq=float('inf'); for liquidity-constrained
                 cases pass the actual bound.
    alpha      : optional scalar or (batch,) tensor appended as an extra
                 input feature (used for the Pareto-frontier network in §7).

Returned dict:
    pnl      : (batch,) terminal portfolio value  p + Σ Δ·ΔF
    delta    : (batch, N, d) positions held over [t_j, t_{j+1}]
    premium  : the scalar trainable p
    tc       : (batch,) transaction-cost penalty (0 if costs disabled)

Key fix vs the earlier notebook: ΔF uses `paths_raw` — NOT the normalised
paths — so the P&L is in the same units as g(S_T) (paper eq. 3).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def make_ff_block(in_dim: int, width: int, n_layers: int,
                  out_dim: int = 1, activation: str = "relu") -> nn.Sequential:
    """FF block: n_layers hidden ReLU/ELU layers of given width, then linear."""
    act_cls = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation]
    layers = [nn.Linear(in_dim, width), act_cls()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(width, width), act_cls()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Accept (batch, N+1) or (batch, N+1, d); return (batch, N+1, d)."""
    return x if x.dim() == 3 else x.unsqueeze(-1)


def _liquidity_gate(delta_prev: torch.Tensor,
                    C_hat: torch.Tensor,
                    liq: float) -> torch.Tensor:
    """Paper eq. (13): Δ_t = Δ_{t-1} + l · tanh(Ĉ).
    If liq is +inf the gate is identity (unbounded position changes)."""
    if liq == float("inf"):
        return delta_prev + C_hat
    return delta_prev + liq * torch.tanh(C_hat)


# ---------------------------------------------------------------------------
# Architecture A — Feedforward basic (§4.1.1, Fig. 1)
# ---------------------------------------------------------------------------
class FeedforwardBasic(nn.Module):
    """N separate feedforward networks, one per step. d assets supported."""

    def __init__(self, N: int, d: int = 1,
                 width: int = 10, n_layers: int = 3,
                 extra_in: int = 0):
        super().__init__()
        self.N, self.d = N, d
        # each per-step net sees d normalised prices (+ optional α) → d deltas
        in_dim = d + extra_in
        self.nets = nn.ModuleList([
            make_ff_block(in_dim, width, n_layers, out_dim=d) for _ in range(N)
        ])
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, paths_raw, paths_norm, liq: float = float("inf"),
                alpha: Optional[torch.Tensor] = None,
                tc_cost: Optional[torch.Tensor] = None) -> dict:
        paths_raw  = _ensure_3d(paths_raw)
        paths_norm = _ensure_3d(paths_norm)
        batch, _, d = paths_raw.shape

        pnl = torch.zeros(batch, device=paths_raw.device)
        tc  = torch.zeros(batch, device=paths_raw.device)
        deltas = []
        delta_prev = torch.zeros(batch, d, device=paths_raw.device)

        for j, net in enumerate(self.nets):
            s_j = paths_norm[:, j, :]
            if alpha is not None:
                a = alpha if alpha.dim() else alpha.expand(batch)
                s_j = torch.cat([s_j, a.unsqueeze(-1)], dim=-1)
            raw_delta = net(s_j)
            # liquidity gate applied as a change vs previous position
            delta_j = _liquidity_gate(delta_prev, raw_delta - delta_prev, liq)
            deltas.append(delta_j.unsqueeze(1))

            dF = paths_raw[:, j + 1, :] - paths_raw[:, j, :]
            pnl = pnl + (delta_j * dF).sum(dim=-1)
            if tc_cost is not None:
                tc = tc + (tc_cost * (delta_j - delta_prev).abs()).sum(dim=-1)
            delta_prev = delta_j

        return {"pnl":     pnl + self.premium.expand(batch),
                "delta":   torch.cat(deltas, dim=1),
                "premium": self.premium,
                "tc":      tc}


# ---------------------------------------------------------------------------
# Architecture B — Feedforward merged (§4.1.2, Fig. 2)
# ---------------------------------------------------------------------------
class FeedforwardMerged(nn.Module):
    """Single shared network fed with (S̃_t, t/T [, α])."""

    def __init__(self, N: int, d: int = 1,
                 width: int = 10, n_layers: int = 3,
                 extra_in: int = 0):
        super().__init__()
        self.N, self.d = N, d
        in_dim = d + 1 + extra_in
        self.net = make_ff_block(in_dim, width, n_layers, out_dim=d)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, paths_raw, paths_norm, liq: float = float("inf"),
                alpha: Optional[torch.Tensor] = None,
                tc_cost: Optional[torch.Tensor] = None) -> dict:
        paths_raw  = _ensure_3d(paths_raw)
        paths_norm = _ensure_3d(paths_norm)
        batch, _, d = paths_raw.shape

        pnl = torch.zeros(batch, device=paths_raw.device)
        tc  = torch.zeros(batch, device=paths_raw.device)
        deltas = []
        delta_prev = torch.zeros(batch, d, device=paths_raw.device)

        for j in range(self.N):
            t_feat = torch.full((batch, 1), j / self.N,
                                device=paths_raw.device)
            feats = torch.cat([paths_norm[:, j, :], t_feat], dim=-1)
            if alpha is not None:
                a = alpha if alpha.dim() else alpha.expand(batch)
                feats = torch.cat([feats, a.unsqueeze(-1)], dim=-1)
            raw_delta = self.net(feats)
            delta_j = _liquidity_gate(delta_prev, raw_delta - delta_prev, liq)
            deltas.append(delta_j.unsqueeze(1))

            dF = paths_raw[:, j + 1, :] - paths_raw[:, j, :]
            pnl = pnl + (delta_j * dF).sum(dim=-1)
            if tc_cost is not None:
                tc = tc + (tc_cost * (delta_j - delta_prev).abs()).sum(dim=-1)
            delta_prev = delta_j

        return {"pnl":     pnl + self.premium.expand(batch),
                "delta":   torch.cat(deltas, dim=1),
                "premium": self.premium,
                "tc":      tc}


# ---------------------------------------------------------------------------
# Architecture C — Classical LSTM (§4.2, Fig. 3)
# ---------------------------------------------------------------------------
class ClassicalLSTM(nn.Module):
    """LSTM cell only — no feedforward head. Output is the raw cell output."""

    def __init__(self, N: int, d: int = 1,
                 hidden: int = 50,
                 extra_in: int = 0):
        super().__init__()
        self.N, self.d, self.hidden = N, d, hidden
        in_dim = d + 1 + extra_in  # (S̃, t/T [, α])
        self.cell = nn.LSTMCell(in_dim, hidden)
        self.out  = nn.Linear(hidden, d)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, paths_raw, paths_norm, liq: float = float("inf"),
                alpha: Optional[torch.Tensor] = None,
                tc_cost: Optional[torch.Tensor] = None) -> dict:
        paths_raw  = _ensure_3d(paths_raw)
        paths_norm = _ensure_3d(paths_norm)
        batch, _, d = paths_raw.shape
        dev = paths_raw.device

        h = torch.zeros(batch, self.hidden, device=dev)
        c = torch.zeros(batch, self.hidden, device=dev)
        pnl = torch.zeros(batch, device=dev)
        tc  = torch.zeros(batch, device=dev)
        deltas = []
        delta_prev = torch.zeros(batch, d, device=dev)

        for j in range(self.N):
            t_feat = torch.full((batch, 1), j / self.N, device=dev)
            feats = torch.cat([paths_norm[:, j, :], t_feat], dim=-1)
            if alpha is not None:
                a = alpha if alpha.dim() else alpha.expand(batch)
                feats = torch.cat([feats, a.unsqueeze(-1)], dim=-1)
            h, c = self.cell(feats, (h, c))
            C_hat = self.out(h)
            delta_j = _liquidity_gate(delta_prev, C_hat, liq)
            deltas.append(delta_j.unsqueeze(1))

            dF = paths_raw[:, j + 1, :] - paths_raw[:, j, :]
            pnl = pnl + (delta_j * dF).sum(dim=-1)
            if tc_cost is not None:
                tc = tc + (tc_cost * (delta_j - delta_prev).abs()).sum(dim=-1)
            delta_prev = delta_j

        return {"pnl":     pnl + self.premium.expand(batch),
                "delta":   torch.cat(deltas, dim=1),
                "premium": self.premium,
                "tc":      tc}


# ---------------------------------------------------------------------------
# Architecture D — Augmented LSTM (§4.2, Fig. 4)  ← paper's best
# ---------------------------------------------------------------------------
class AugmentedLSTM(nn.Module):
    """LSTM cell + FF head (ReLU). This is the paper's best architecture."""

    def __init__(self, N: int, d: int = 1,
                 hidden: int = 50,
                 ff_width: int = 10, ff_layers: int = 3,
                 extra_in: int = 0):
        super().__init__()
        self.N, self.d, self.hidden = N, d, hidden
        in_dim = d + 1 + extra_in     # (S̃, t/T [, α])
        self.cell = nn.LSTMCell(in_dim, hidden)
        self.head = make_ff_block(hidden, ff_width, ff_layers, out_dim=d)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, paths_raw, paths_norm, liq: float = float("inf"),
                alpha: Optional[torch.Tensor] = None,
                tc_cost: Optional[torch.Tensor] = None) -> dict:
        paths_raw  = _ensure_3d(paths_raw)
        paths_norm = _ensure_3d(paths_norm)
        batch, _, d = paths_raw.shape
        dev = paths_raw.device

        h = torch.zeros(batch, self.hidden, device=dev)
        c = torch.zeros(batch, self.hidden, device=dev)
        pnl = torch.zeros(batch, device=dev)
        tc  = torch.zeros(batch, device=dev)
        deltas = []
        delta_prev = torch.zeros(batch, d, device=dev)

        for j in range(self.N):
            t_feat = torch.full((batch, 1), j / self.N, device=dev)
            feats = torch.cat([paths_norm[:, j, :], t_feat], dim=-1)
            if alpha is not None:
                a = alpha if alpha.dim() else alpha.expand(batch)
                feats = torch.cat([feats, a.unsqueeze(-1)], dim=-1)
            h, c = self.cell(feats, (h, c))
            C_hat = self.head(h)
            delta_j = _liquidity_gate(delta_prev, C_hat, liq)
            deltas.append(delta_j.unsqueeze(1))

            dF = paths_raw[:, j + 1, :] - paths_raw[:, j, :]
            pnl = pnl + (delta_j * dF).sum(dim=-1)
            if tc_cost is not None:
                tc = tc + (tc_cost * (delta_j - delta_prev).abs()).sum(dim=-1)
            delta_prev = delta_j

        return {"pnl":     pnl + self.premium.expand(batch),
                "delta":   torch.cat(deltas, dim=1),
                "premium": self.premium,
                "tc":      tc}
