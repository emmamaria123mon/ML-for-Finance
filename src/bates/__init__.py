"""
Bates (Heston + jumps) utilities used by notebooks.

This module is intentionally lightweight and notebook-friendly:
- numpy MC terminal simulation + pricing helpers
- torch path simulation for training/eval datasets
- delta helpers (BS analytic, Bates MC finite-difference via terminal multipliers)
"""

from .simulation import simulate_bates_paths_torch, simulate_bates_terminal_multipliers
from .pricing import (
    bates_call_prices_mc,
    bates_call_price_mc,
    bates_call_prices_carr_madan,
    bates_call_price_carr_madan,
    bates_call_prices,
)
from .deltas import bs_call_price, bs_delta_np, bates_delta_mc_fd_vec_spots
from .execution import apply_friction_to_deltas

__all__ = [
    "simulate_bates_paths_torch",
    "simulate_bates_terminal_multipliers",
    "bates_call_prices_mc",
    "bates_call_price_mc",
    "bates_call_prices_carr_madan",
    "bates_call_price_carr_madan",
    "bates_call_prices",
    "bs_call_price",
    "bs_delta_np",
    "bates_delta_mc_fd_vec_spots",
    "apply_friction_to_deltas",
]

