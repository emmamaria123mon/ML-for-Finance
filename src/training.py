"""Training loop — paper's Algorithm 1 (global forward resolution).

The loop is deliberately loss-agnostic: pass any `loss_fn(Y_or_error)` that
returns a scalar tensor. The model's forward pass must return the dict
contract documented in architectures.py.

The one subtlety vs the paper: we pre-simulate a finite training set and
sample mini-batches from it (rather than drawing new Monte-Carlo samples
every iteration). This matches what is done in the authors' released code
and is consistent with the paper's §4.3 description of running over a fixed
training pool with periodic test-loss checkpointing.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional
import copy

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_model(
    model: nn.Module,
    *,
    train_paths_raw: torch.Tensor,
    train_paths_norm: torch.Tensor,
    eval_paths_raw: torch.Tensor,
    eval_paths_norm: torch.Tensor,
    payoff_fn: Callable[[torch.Tensor], torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    liq: float = float("inf"),
    tc_cost: Optional[torch.Tensor] = None,
    alpha_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
    loss_needs_alpha: bool = False,
    n_iter: int = 20_000,
    batch_size: int = 50,
    lr: float = 1e-3,
    lr_step_frac: float = 1 / 3,
    lr_gamma: float = 0.5,
    test_every: int = 1000,
    grad_clip: Optional[float] = 1.0,
    label: str = "",
    progress: bool = True,
) -> Dict:
    """Algorithm 1 training loop with best-test-loss checkpointing.

    Returns history dict {train, test, test_iters, best_metric, best_state}.
    """
    device = train_paths_raw.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(n_iter * lr_step_frac)), gamma=lr_gamma,
    )

    history = {"train": [], "test": [], "test_iters": [],
               "best_metric": float("inf"), "best_state": None}

    iterator = range(n_iter)
    if progress:
        iterator = tqdm(iterator, desc=label or "train")

    n_train = train_paths_raw.shape[0]

    for it in iterator:
        model.train()
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        br  = train_paths_raw[idx]
        bn  = train_paths_norm[idx]

        alpha = alpha_sampler(batch_size, device) if alpha_sampler else None

        out  = model(br, bn, liq=liq, alpha=alpha, tc_cost=tc_cost)
        g_ST = payoff_fn(br)
        Y = out["pnl"] - g_ST - out["tc"]
        if loss_needs_alpha:
            loss = loss_fn(Y, alpha)
        else:
            loss = loss_fn(Y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        history["train"].append(loss.item())

        if (it + 1) % test_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on the full eval set in chunks to limit memory
                eval_Y = []
                chunk = max(1, min(4096, eval_paths_raw.shape[0]))
                for start in range(0, eval_paths_raw.shape[0], chunk):
                    ebr = eval_paths_raw[start:start + chunk]
                    ebn = eval_paths_norm[start:start + chunk]
                    a = alpha_sampler(ebr.shape[0], device) if alpha_sampler else None
                    o = model(ebr, ebn, liq=liq, alpha=a, tc_cost=tc_cost)
                    Y_e = o["pnl"] - payoff_fn(ebr) - o["tc"]
                    eval_Y.append(Y_e)
                eval_Y = torch.cat(eval_Y)
                # For test metric we use plain MSE of Y regardless of training
                # loss, so curves are comparable across runs (paper does the
                # same — "MSE" column in Table 1).
                test_metric = eval_Y.pow(2).mean().item()

            history["test"].append(test_metric)
            history["test_iters"].append(it + 1)

            if test_metric < history["best_metric"]:
                history["best_metric"] = test_metric
                history["best_state"] = copy.deepcopy(model.state_dict())

    if history["best_state"] is not None:
        model.load_state_dict(history["best_state"])
    return history
