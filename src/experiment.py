"""Experiment harness — one Config to rule them all.

A `Config` captures every "thing that might change" across experiments:
  * dataset           → `paths_factory`
  * payoff            → `payoff_fn`
  * architectures     → `arch_factory` (dict of factories)
  * loss/risk measure → `loss_fn`
  * constraints       → `liq`, `tc_cost`
  * optimisation      → `n_iter`, `batch_size`, `lr`, ...
  * paper reference   → `paper_values` (used only by the reporter)

`run_experiment(cfg)` trains every model in `cfg.arch_factory` on the same
paths with the same loss and returns a uniform results dict that the
reporting helpers and the notebook's save block both consume.

This separation lets the replication notebook and the extension notebooks
share one runner:  swap a factory, swap a loss, keep everything else
constant.  Anything overridden is visible as a single config-diff.

Defaults mirror the paper (§4.3):  Adam lr = 1e-3,  batch 50,  20 000
iterations,  grad clip 1.0,  test every 1 000 iterations,  no liquidity
constraint,  no transaction costs,  MSE loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .losses import mse
from .simulators import normalize_with
from .training import train_model


# ---------------------------------------------------------------------------
# Type aliases — spelled out so the Config fields are self-documenting
# ---------------------------------------------------------------------------
# A paths factory is a zero-arg callable that returns
#   (train_raw, eval_raw, norm_mean, norm_std)
# so that a swap of dataset (simulated, real, bootstrapped, ...) only
# changes this one function.
PathsFactory = Callable[
    [], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]

# An arch factory is a zero-arg callable returning a fresh nn.Module.
# Using zero-arg callables (not instances) ensures every experiment gets a
# freshly-initialised model — otherwise re-running a cell would warm-start
# from the previous training run.
ArchFactory = Callable[[], nn.Module]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Single source of truth for one experiment.

    Mandatory:
      name           identifier used in logs and save-file names
      d              number of tradable assets (1 = single-asset, d >= 2 multi)
      N              number of hedging steps (paper's N = T/Δt)
      paths_factory  see PathsFactory alias above
      payoff_fn      receives raw paths (batch, N+1[, d]) → payoff (batch,)
      arch_factory   {name: ArchFactory}

    Optional (paper defaults):
      loss_fn        default mse (paper eq. 4)
      liq            default +inf (no liquidity constraint)
      tc_cost        default None (no transaction costs)
      n_iter         default 20 000
      batch_size     default 50
      lr             default 1e-3
      test_every     default 1 000
      grad_clip      default 1.0  (None disables)
      paper_values   {arch_name: reference MSE} for the reporter; may be {}
    """

    name: str
    d: int
    N: int
    paths_factory: PathsFactory
    payoff_fn: Callable[[torch.Tensor], torch.Tensor]
    arch_factory: Dict[str, ArchFactory]

    # paper-default optimisation & loss
    loss_fn: Callable = mse
    liq: float = float("inf")
    tc_cost: Optional[torch.Tensor] = None
    n_iter: int = 20_000
    batch_size: int = 50
    lr: float = 1e-3
    test_every: int = 1_000
    grad_clip: Optional[float] = 1.0

    # for the reporter only — empty is fine (extension experiments have no
    # paper reference)
    paper_values: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_experiment(
    cfg: Config,
    device: str = "cpu",
    progress: bool = True,
) -> Dict[str, Any]:
    """Train every model in `cfg.arch_factory` and return a results dict.

    Returned dict layout:
      {
        'cfg'      : cfg,
        'paths'    : {'train_raw', 'eval_raw', 'train_norm', 'eval_norm',
                      'norm_mean', 'norm_std'},       # tensors on `device`
        'models'   : {name: nn.Module},               # with best-test weights loaded
        'histories': {name: history dict from train_model},
        'pnl'      : {name: np.ndarray of shape (N_eval,)},
        'mse'      : {name: float},                   # final eval MSE of Y
      }

    The training loop itself already handles best-test-loss checkpointing,
    so `models[name]` is the best-test version, not the last-iter version.
    """
    train_raw, eval_raw, norm_mean, norm_std = cfg.paths_factory()
    train_raw = train_raw.to(device)
    eval_raw = eval_raw.to(device)
    norm_mean = norm_mean.to(device)
    norm_std = norm_std.to(device)

    train_norm = normalize_with(train_raw, norm_mean, norm_std)
    eval_norm = normalize_with(eval_raw, norm_mean, norm_std)

    # Move TC cost tensor to the device if provided.  A scalar float is
    # also fine (it will broadcast); we only need .to() for tensors.
    tc_cost = cfg.tc_cost
    if isinstance(tc_cost, torch.Tensor):
        tc_cost = tc_cost.to(device)

    models: Dict[str, nn.Module] = {}
    histories: Dict[str, Dict] = {}
    pnls: Dict[str, np.ndarray] = {}
    mses: Dict[str, float] = {}

    for name, make in cfg.arch_factory.items():
        model = make().to(device)

        histories[name] = train_model(
            model,
            train_paths_raw=train_raw,
            train_paths_norm=train_norm,
            eval_paths_raw=eval_raw,
            eval_paths_norm=eval_norm,
            payoff_fn=cfg.payoff_fn,
            loss_fn=cfg.loss_fn,
            liq=cfg.liq,
            tc_cost=tc_cost,
            n_iter=cfg.n_iter,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            test_every=cfg.test_every,
            grad_clip=cfg.grad_clip,
            label=f"{cfg.name}:{name}",
            progress=progress,
        )
        models[name] = model

        # final PnL & MSE on the eval set — paper metric is MSE of
        # Y = pnl - g(S_T) - tc, matching Eq. (4).
        model.eval()
        with torch.no_grad():
            out = model(eval_raw, eval_norm, liq=cfg.liq, tc_cost=tc_cost)
            g = cfg.payoff_fn(eval_raw)
            Y = out["pnl"] - g - out["tc"]
        pnls[name] = Y.detach().cpu().numpy()
        mses[name] = float(Y.pow(2).mean().item())

    return {
        "cfg": cfg,
        "paths": {
            "train_raw": train_raw,
            "eval_raw": eval_raw,
            "train_norm": train_norm,
            "eval_norm": eval_norm,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
        },
        "models": models,
        "histories": histories,
        "pnl": pnls,
        "mse": mses,
    }


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
def report_table(
    all_results: Dict[str, Dict[str, Any]],
    title: str = "Results",
    ratio_ok: Tuple[float, float] = (0.3, 3.0),
) -> None:
    """Print a cross-experiment comparison table.

    `all_results` maps experiment-name → the dict returned by
    `run_experiment`.  For each (experiment, model) we print our MSE,
    the paper's MSE (from `cfg.paper_values`, if present), and the ratio.

    If no paper value is available for an entry, the Paper / Ratio columns
    show '—' — so extension notebooks get a degraded-but-readable table
    without us needing to special-case them.
    """
    line = "=" * 86
    print(line)
    print(f"  {title}")
    print(line)
    print(
        f"  {'Experiment':<22}  {'Model':<22}  "
        f"{'Our MSE':>12}  {'Paper MSE':>12}  {'Ratio':>10}"
    )
    print("-" * 86)
    for exp_name, res in all_results.items():
        cfg: Config = res["cfg"]
        for model_name, our in res["mse"].items():
            paper = cfg.paper_values.get(model_name)
            if paper is not None and paper > 0:
                ratio = our / paper
                tag = "OK" if ratio_ok[0] < ratio < ratio_ok[1] else "!!"
                paper_s = f"{paper:.3e}"
                ratio_s = f"{ratio:6.2f}x {tag}"
            else:
                paper_s = "—"
                ratio_s = "—"
            print(
                f"  {exp_name:<22}  {model_name:<22}  "
                f"{our:>12.3e}  {paper_s:>12}  {ratio_s:>10}"
            )
    print(line)
