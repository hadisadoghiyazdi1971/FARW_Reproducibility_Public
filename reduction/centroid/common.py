"""Shared numerical utilities for the two conceptually separate baselines."""

from __future__ import annotations

from math import floor, isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment


DEFAULT_BUDGETS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


def nominal_target_count(budget: float, budget_reference_n: int = 350) -> int:
    """MATLAB-compatible half-up target based on a fixed nominal reference."""
    budget = float(budget)
    budget_reference_n = int(budget_reference_n)
    if budget_reference_n < 1:
        raise ValueError("budget_reference_n must be at least 1.")
    if not isfinite(budget) or not (0.0 < budget <= 1.0):
        raise ValueError("budget must be finite and in the interval (0, 1].")
    return max(1, floor(budget * budget_reference_n + 0.5))


def validate_and_standardize(
    X: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate X and Z-score columns for calculations without mutating X."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional [N_actual, N_features] matrix.")
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("X must contain at least one window and one feature.")
    if not np.all(np.isfinite(data)):
        row, feature = np.argwhere(~np.isfinite(data))[0]
        raise ValueError(
            "X contains NaN or Inf at zero-based row "
            f"{int(row)}, feature {int(feature)}; rows are never removed silently."
        )

    means = np.mean(data, axis=0, dtype=np.float64)
    if data.shape[0] == 1:
        standard_deviations = np.zeros(data.shape[1], dtype=np.float64)
    else:
        standard_deviations = np.std(data, axis=0, ddof=1, dtype=np.float64)
    constant_mask = standard_deviations == 0.0
    safe_scale = standard_deviations.copy()
    safe_scale[constant_mask] = 1.0
    standardized = (data - means) / safe_scale
    standardized[:, constant_mask] = 0.0
    if not np.all(np.isfinite(standardized)):
        raise FloatingPointError("Z-score preprocessing produced a non-finite value.")
    return standardized, constant_mask


def resolve_target(
    X: ArrayLike, budget: float, budget_reference_n: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_], int, int, int]:
    standardized, constant_mask = validate_and_standardize(X)
    n_actual, n_features = standardized.shape
    k_target = nominal_target_count(budget, budget_reference_n)
    if k_target > n_actual:
        raise ValueError(
            "Requested K_target exceeds the available observed windows: "
            f"K_target={k_target}, N_actual={n_actual}, "
            f"N_budget_reference={int(budget_reference_n)}, budget={float(budget):g}. "
            "No clamping, padding, duplication, or truncation is allowed."
        )
    return standardized, constant_mask, n_actual, n_features, k_target


def squared_distances(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def global_unique_assignment(
    observations: NDArray[np.float64], representatives: NDArray[np.float64]
) -> tuple[NDArray[np.int64], float]:
    """Map continuous representatives to distinct observations at minimum total cost."""
    costs = squared_distances(representatives, observations)
    representative_rows, observation_columns = linear_sum_assignment(costs)
    if representative_rows.size != representatives.shape[0]:
        raise RuntimeError("The assignment did not map every representative.")
    selected = np.empty(representatives.shape[0], dtype=np.int64)
    selected[representative_rows] = observation_columns.astype(np.int64, copy=False)
    return selected, float(costs[representative_rows, observation_columns].sum())


def base_metadata(
    *,
    method: str,
    n_actual: int,
    n_features: int,
    budget_reference_n: int,
    budget: float,
    k_target: int,
    random_state: int,
    constant_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    return {
        "method": method,
        "method_version": "1.0.0",
        "index_base": 0,
        "N_actual": int(n_actual),
        "N_budget_reference": int(budget_reference_n),
        "budget": float(budget),
        "K_target": int(k_target),
        "n_features": int(n_features),
        "random_state": int(random_state),
        "distance": "euclidean_in_per_recording_zscore_space",
        "preprocessing": {
            "method": "per_recording_zscore",
            "std_ddof": 1,
            "constant_columns": "retained_and_set_to_zero",
            "constant_feature_indices": np.flatnonzero(constant_mask).astype(int).tolist(),
            "selection_space_only": True,
        },
        "uses_labels": False,
        "uses_classifier_information": False,
        "uses_temporal_stratification": False,
        "uses_all_actual_windows": True,
    }


def verify_selected_indices(
    selected: NDArray[np.int64], *, k_target: int, n_actual: int
) -> None:
    if selected.ndim != 1 or selected.size != k_target:
        raise RuntimeError("Selection did not return exactly K_target indices.")
    if np.unique(selected).size != k_target:
        raise RuntimeError("Selection returned duplicate indices.")
    if int(selected.min()) < 0 or int(selected.max()) >= n_actual:
        raise RuntimeError("Selection returned an out-of-range original-row index.")

