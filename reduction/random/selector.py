"""Final Random comparator: independent selections for predefined seeds 1--30."""

from __future__ import annotations

import numpy as np

SEEDS = tuple(range(1, 31))


def select_random(n_windows: int, n_keep: int, seed: int) -> np.ndarray:
    """Return sorted, distinct zero-based observed-window indices."""
    if seed not in SEEDS:
        raise ValueError("The published Random protocol uses seeds 1--30 only")
    if not (0 < n_keep <= n_windows):
        raise ValueError("Invalid fixed retention size")
    return np.sort(np.random.default_rng(seed).choice(n_windows, n_keep, replace=False))
