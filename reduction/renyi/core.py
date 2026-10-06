"""Faithful implementation of the professor-original RDS/RDS-IS method.

The optimizer follows the published synchronous fixed-point update for
minimizing D_a(p || q_Y).  It deliberately contains no Wasserstein,
Sinkhorn, repulsion, Adam, temporal, label, or classifier component.
"""

from __future__ import annotations

from math import floor, isfinite, log, pi
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


DEFAULT_BUDGETS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_MAX_ITER = 2000
_MIN_POSITIVE_BANDWIDTH = np.sqrt(np.finfo(np.float64).tiny)


def matlab_half_up_count(budget: float, budget_reference_n: int = 350) -> int:
    """Return max(1, floor(budget * N_budget_reference + 0.5))."""
    budget = float(budget)
    budget_reference_n = int(budget_reference_n)
    if budget_reference_n < 1:
        raise ValueError("budget_reference_n must be at least 1.")
    if not isfinite(budget) or not (0.0 < budget <= 1.0):
        raise ValueError("budget must be finite and in the interval (0, 1].")
    return max(1, floor(budget * budget_reference_n + 0.5))


def _validate_matrix(X: ArrayLike) -> NDArray[np.float64]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional [N_windows, N_features] matrix.")
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("X must contain at least one window and one feature.")
    if not np.all(np.isfinite(data)):
        bad = np.argwhere(~np.isfinite(data))[0]
        raise ValueError(
            "X contains NaN or Inf at zero-based row "
            f"{int(bad[0])}, feature {int(bad[1])}; rows are never deleted silently."
        )
    return data


def zscore_for_selection(
    X: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Per-recording Z-score using sample std (ddof=1).

    Constant columns are retained and mapped exactly to zero.  The input is
    never modified.
    """
    data = _validate_matrix(X)
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
    return standardized, means, safe_scale, constant_mask


def scott_isotropic_bandwidth(n_windows: int, n_features: int) -> float:
    """Scott factor for isotropic KDE after per-feature standardization.

    h = N**(-1 / (d + 4)).  After Z-scoring, each nonconstant coordinate has
    unit scale; constant coordinates remain zero but remain part of d.
    """
    n_windows = int(n_windows)
    n_features = int(n_features)
    if n_windows < 1 or n_features < 1:
        raise ValueError("Scott bandwidth requires positive N and d.")
    return float(n_windows ** (-1.0 / (n_features + 4.0)))


def _resolve_bandwidth(
    bandwidth: str | float,
    n_windows: int,
    n_features: int,
) -> tuple[float, str]:
    if isinstance(bandwidth, str):
        if bandwidth.lower() != "scott":
            raise ValueError("The only named bandwidth strategy currently supported is 'scott'.")
        h = scott_isotropic_bandwidth(n_windows, n_features)
        method = "scott_isotropic_after_zscore"
    else:
        h = float(bandwidth)
        method = "fixed_numeric"
    if not isfinite(h) or h < _MIN_POSITIVE_BANDWIDTH:
        raise ValueError(f"bandwidth must be finite and >= {_MIN_POSITIVE_BANDWIDTH:.3e}.")
    return h, method


def _squared_distances(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def _full_reference_log_kde(
    X: NDArray[np.float64], h: float
) -> NDArray[np.float64]:
    """Evaluate the full Gaussian KDE p at every observed row (self included)."""
    n_windows, n_features = X.shape
    log_kernel = -0.5 * _squared_distances(X, X) / (h * h)
    log_normalizer = (
        log(float(n_windows))
        + n_features * log(h)
        + 0.5 * n_features * log(2.0 * pi)
    )
    return logsumexp(log_kernel, axis=1) - log_normalizer


def _fixed_point_step(
    X: NDArray[np.float64],
    prototypes: NDArray[np.float64],
    *,
    a: float,
    h: float,
    reference_log_density: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Compute one synchronous published RDS fixed-point update."""
    log_kernel = -0.5 * _squared_distances(X, prototypes) / (h * h)
    log_f = logsumexp(log_kernel, axis=1)

    if a == 1.0:
        log_base = -log_f
    else:
        if reference_log_density is None:
            raise ValueError("The p(z_i) KDE is required when a != 1.")
        log_base = (a - 1.0) * reference_log_density - a * log_f

    log_weights = log_base[:, None] + log_kernel
    column_maxima = np.max(log_weights, axis=0, keepdims=True)
    if not np.all(np.isfinite(column_maxima)):
        raise FloatingPointError("All kernel weights vanished for at least one prototype.")
    weights = np.exp(log_weights - column_maxima)
    denominators = np.sum(weights, axis=0)
    if np.any(denominators <= 0.0) or not np.all(np.isfinite(denominators)):
        raise FloatingPointError("Invalid fixed-point weight denominator.")
    updated = (weights.T @ X) / denominators[:, None]
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError("The fixed-point update produced a non-finite prototype.")
    return updated


def _global_unique_projection(
    X: NDArray[np.float64], prototypes: NDArray[np.float64]
) -> tuple[NDArray[np.int64], float]:
    """Minimum-total-cost one-to-one prototype-to-observation assignment."""
    costs = _squared_distances(prototypes, X)
    row_indices, column_indices = linear_sum_assignment(costs)
    if row_indices.size != prototypes.shape[0]:
        raise RuntimeError("Rectangular assignment did not map every prototype.")
    selected = np.empty(prototypes.shape[0], dtype=np.int64)
    selected[row_indices] = column_indices.astype(np.int64, copy=False)
    total_cost = float(costs[row_indices, column_indices].sum())
    return selected, total_cost


def select_renyi_rds(
    X: ArrayLike,
    budget: float,
    *,
    budget_reference_n: int = 350,
    a: float = 1.0,
    bandwidth: str | float = "scott",
    epsilon: float = 1e-5,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = 42,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Select unique observed-window indices with professor-original RDS-IS.

    Parameters are label-free and recording-local.  Returned indices are
    zero-based positions into the original, unmodified X matrix.  The target
    K is calculated from budget_reference_n, not from X.shape[0].
    """
    data = _validate_matrix(X)
    n_windows, n_features = data.shape
    budget_reference_n = int(budget_reference_n)
    K = matlab_half_up_count(budget, budget_reference_n)
    if K > n_windows:
        raise ValueError(
            "Requested K exceeds the available observed windows: "
            f"K={K}, N_actual={n_windows}, N_budget_reference={budget_reference_n}, "
            f"budget={float(budget):g}. No clamping, padding, or duplication is allowed."
        )

    a = float(a)
    epsilon = float(epsilon)
    max_iter = int(max_iter)
    random_state = int(random_state)
    if not isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and greater than zero.")
    if not isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    standardized, _, _, constant_mask = zscore_for_selection(data)
    h, bandwidth_method = _resolve_bandwidth(bandwidth, n_windows, n_features)

    metadata: dict[str, Any] = {
        "algorithm": "professor_original_renyi_rds_is",
        "algorithm_version": "1.1.0",
        "index_base": 0,
        "N_actual": int(n_windows),
        "N_budget_reference": int(budget_reference_n),
        "n_features": int(n_features),
        "K": int(K),
        "budget": float(budget),
        "a": a,
        "bandwidth_method": bandwidth_method,
        "h": h,
        "preprocessing": {
            "method": "per_recording_zscore",
            "std_ddof": 1,
            "constant_columns": "retained_and_set_to_zero",
            "constant_feature_indices": np.flatnonzero(constant_mask).astype(int).tolist(),
            "selection_space_only": True,
        },
        "epsilon": epsilon,
        "max_iter": max_iter,
        "seed": random_state,
        "initialization": "random_observed_rows_without_replacement",
        "optimization": "synchronous_rds_fixed_point",
        "reference_density": "not_needed_at_a_equals_1"
        if a == 1.0
        else "full_gaussian_kde_including_self",
        "projection": "global_linear_assignment_squared_euclidean",
        "uses_labels": False,
        "uses_temporal_stratification": False,
    }

    if K == n_windows:
        selected = np.arange(n_windows, dtype=np.int64)
        metadata.update(
            {
                "iterations": 0,
                "converged": True,
                "final_displacement": 0.0,
                "projection_total_cost": 0.0,
                "selection_shortcut": "K_equals_N_all_rows_selected",
            }
        )
        return selected, metadata

    rng = np.random.default_rng(random_state)
    initial_indices = rng.choice(n_windows, size=K, replace=False)
    prototypes = standardized[initial_indices].copy()
    reference_log_density = (
        None if a == 1.0 else _full_reference_log_kde(standardized, h)
    )

    converged = False
    final_displacement = float("inf")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        updated = _fixed_point_step(
            standardized,
            prototypes,
            a=a,
            h=h,
            reference_log_density=reference_log_density,
        )
        final_displacement = float(
            np.max(np.linalg.norm(updated - prototypes, axis=1))
        )
        prototypes = updated
        if final_displacement < epsilon:
            converged = True
            break

    selected, projection_cost = _global_unique_projection(standardized, prototypes)
    if selected.size != K or np.unique(selected).size != K:
        raise RuntimeError("Internal error: projection did not produce exactly K unique indices.")
    if int(selected.min()) < 0 or int(selected.max()) >= n_windows:
        raise RuntimeError("Internal error: projection returned an out-of-range index.")

    metadata.update(
        {
            "iterations": int(iterations),
            "converged": bool(converged),
            "final_displacement": final_displacement,
            "projection_total_cost": projection_cost,
        }
    )
    return selected, metadata
