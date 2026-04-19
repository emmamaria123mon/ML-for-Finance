from __future__ import annotations

import numpy as np


def apply_friction_to_deltas(delta_target, dS, tc_per_unit: float, liq_step: float):
    """
    Execute target deltas with a per-step liquidity cap and proportional transaction costs.

    Returns:
      - delta_exec: executed delta per step
      - hedge_inc: per-step hedge PnL increment: delta_exec_t * dS_t - tc_per_unit * |trade_t|
      - tc_paid: per-step transaction cost paid
    """
    delta_target = np.asarray(delta_target, dtype=float)
    dS = np.asarray(dS, dtype=float)

    n = min(len(delta_target), len(dS))
    delta_target = delta_target[:n]
    dS = dS[:n]

    delta_exec = np.zeros(n, dtype=float)
    tc_paid = np.zeros(n, dtype=float)

    prev = 0.0
    for t in range(n):
        trade_desired = delta_target[t] - prev
        trade_exec = np.clip(trade_desired, -liq_step, liq_step)
        curr = prev + trade_exec
        delta_exec[t] = curr
        tc_paid[t] = tc_per_unit * abs(trade_exec)
        prev = curr

    hedge_inc = delta_exec * dS - tc_paid
    return delta_exec, hedge_inc, tc_paid

