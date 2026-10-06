"""Frozen, deterministic FARW-v1 core for one already-scaled recording.

The core is label-free and returns original observed-window indices only.  It
never fits a per-recording feature scaler; use :mod:`farw.scaling` on the full
outer-training partition before calling :func:`select_farw`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scaling import require_finite_matrix


DEFAULT_BUDGET_REFERENCE_N = 350
DEFAULT_LAMBDA = 1.0
DEFAULT_BETA = 0.0
DEFAULT_EPSILON = 1e-6
DEFAULT_MAX_ACCEPTED_SWAPS = 100
GLOBAL_TIE_POLICY = "lowest_zero_based_original_window_index"


@dataclass(frozen=True)
class StructuralQuantities:
    """Exact structural quantities for one selected subset."""

    selected_indices: NDArray[np.int64]
    nearest_indices: NDArray[np.int64]
    second_indices: NDArray[np.int64]
    nearest_distances: NDArray[np.float64]
    second_distances: NDArray[np.float64]
    C: float
    phi: NDArray[np.float64]
    F1: float
    J: float

    def phi_by_index(self) -> dict[int, float]:
        return {
            int(index): float(value)
            for index, value in zip(self.selected_indices, self.phi, strict=True)
        }


@dataclass(frozen=True)
class SwapChoice:
    removed_index: int
    inserted_index: int
    improvement: float
    trial_state: StructuralQuantities


def fixed_budget_count(
    budget: float,
    budget_reference_n: int = DEFAULT_BUDGET_REFERENCE_N,
) -> int:
    """Return FARW-v1's fixed-reference half-up count.

    K = floor(budget * N_budget_reference + 0.5)
    """
    budget = float(budget)
    budget_reference_n = int(budget_reference_n)
    if not isfinite(budget) or not (0.0 < budget <= 1.0):
        raise ValueError("budget must be finite and in (0, 1].")
    if budget_reference_n < 1:
        raise ValueError("budget_reference_n must be at least 1.")
    return int(floor(budget * budget_reference_n + 0.5))


def candidate_pool_size(K: int) -> int:
    K = int(K)
    if K < 1:
        raise ValueError("K must be positive.")
    return max(5, int(ceil(sqrt(K))))


def pairwise_euclidean(X_scaled: ArrayLike, *, beta: float = DEFAULT_BETA) -> NDArray[np.float64]:
    """Pairwise Euclidean distances, optionally with beta*t/T time coordinate."""
    X = require_finite_matrix(X_scaled, name="X_scaled")
    beta = float(beta)
    if not isfinite(beta):
        raise ValueError("beta must be finite.")

    if beta != 0.0:
        n_windows = X.shape[0]
        time_coordinate = beta * (np.arange(n_windows, dtype=np.float64) + 1.0) / n_windows
        geometry = np.column_stack((X, time_coordinate))
    else:
        geometry = X

    squared_norms = np.einsum("ij,ij->i", geometry, geometry)
    squared_distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * geometry @ geometry.T
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    distances = np.sqrt(squared_distances, out=squared_distances)
    np.fill_diagonal(distances, 0.0)
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("Pairwise Euclidean distance calculation is non-finite.")
    return distances


def _validate_distance_matrix(distance_matrix: ArrayLike) -> NDArray[np.float64]:
    distances = require_finite_matrix(distance_matrix, name="distance_matrix")
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if np.any(distances < 0.0):
        raise ValueError("distance_matrix cannot contain negative distances.")
    return distances


def _normalize_selected(selected: Sequence[int] | NDArray[np.integer], n_windows: int) -> NDArray[np.int64]:
    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
    if indices.size < 1:
        raise ValueError("At least one selected index is required.")
    if np.unique(indices).size != indices.size:
        raise ValueError("selected indices must be unique.")
    if int(indices.min()) < 0 or int(indices.max()) >= n_windows:
        raise IndexError("selected index is outside [0, N_actual-1].")
    return np.sort(indices)


def medoid_seeded_farthest_first(distance_matrix: ArrayLike, K: int) -> NDArray[np.int64]:
    """Select exactly K observed indices with the frozen deterministic initializer.

    The returned array preserves traversal order.  Every argmin/argmax sees
    indices in increasing original order, so NumPy's first occurrence realizes
    the global lowest-index tie policy.
    """
    distances = _validate_distance_matrix(distance_matrix)
    n_windows = distances.shape[0]
    K = int(K)
    if not (1 <= K <= n_windows):
        raise ValueError(f"K must satisfy 1 <= K <= N_actual={n_windows}; received {K}.")

    medoid = int(np.argmin(np.sum(distances, axis=1, dtype=np.float64)))
    order = [medoid]
    selected_mask = np.zeros(n_windows, dtype=bool)
    selected_mask[medoid] = True
    nearest_to_set = distances[:, medoid].copy()

    while len(order) < K:
        scores = nearest_to_set.copy()
        scores[selected_mask] = -np.inf
        next_index = int(np.argmax(scores))
        order.append(next_index)
        selected_mask[next_index] = True
        np.minimum(nearest_to_set, distances[:, next_index], out=nearest_to_set)

    return np.asarray(order, dtype=np.int64)


def evenly_spaced_reference_indices(n_windows: int, *, max_reference_size: int = 16) -> NDArray[np.int64]:
    """Return half-up-rounded linspace indices including both endpoints when possible."""
    n_windows = int(n_windows)
    max_reference_size = int(max_reference_size)
    if n_windows < 1 or max_reference_size < 1:
        raise ValueError("n_windows and max_reference_size must be positive.")
    q_ref = min(max_reference_size, n_windows)
    if q_ref == 1:
        return np.asarray([0], dtype=np.int64)
    positions = np.linspace(0.0, float(n_windows - 1), q_ref)
    indices = np.floor(positions + 0.5).astype(np.int64)
    if np.unique(indices).size != q_ref:
        raise RuntimeError("Even reference-index construction produced duplicates.")
    return indices


def compute_reference_scale(
    distance_matrix: ArrayLike,
    *,
    max_reference_size: int = 16,
) -> tuple[float, NDArray[np.int64]]:
    """Compute fixed s_r from all positive window-to-reference distances."""
    distances = _validate_distance_matrix(distance_matrix)
    reference_indices = evenly_spaced_reference_indices(
        distances.shape[0], max_reference_size=max_reference_size
    )
    values = distances[:, reference_indices].reshape(-1)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError(
            "Degenerate recording: no positive window-to-reference distance exists, "
            "so s_r cannot be positive."
        )
    s_r = float(np.median(positive))
    if not isfinite(s_r) or s_r <= 0.0:
        raise ValueError("Degenerate recording: computed s_r is not finite and positive.")
    return s_r, reference_indices


def coverage_cost(distance_matrix: ArrayLike, selected: Sequence[int]) -> float:
    """Exact mean nearest-representative distortion C_r(S)."""
    distances = _validate_distance_matrix(distance_matrix)
    indices = _normalize_selected(selected, distances.shape[0])
    return float(np.mean(np.min(distances[:, indices], axis=1)))


def evaluate_structural_quantities(
    distance_matrix: ArrayLike,
    selected: Sequence[int],
    *,
    s_r: float,
    lambda_: float = DEFAULT_LAMBDA,
) -> StructuralQuantities:
    """Compute exact C, nearest/second-nearest assignments, phi, F1, and J."""
    distances = _validate_distance_matrix(distance_matrix)
    indices = _normalize_selected(selected, distances.shape[0])
    if indices.size < 2:
        raise ValueError("At least two distinct selected indices are required for singleton fragility.")
    s_r = float(s_r)
    lambda_ = float(lambda_)
    if not isfinite(s_r) or s_r <= 0.0:
        raise ValueError("s_r must be finite and strictly positive.")
    if not isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError("lambda must be finite and nonnegative.")

    selected_distances = distances[:, indices]
    nearest_positions = np.argmin(selected_distances, axis=1)
    rows = np.arange(distances.shape[0])
    nearest_distances = selected_distances[rows, nearest_positions]

    second_work = selected_distances.copy()
    second_work[rows, nearest_positions] = np.inf
    second_positions = np.argmin(second_work, axis=1)
    second_distances = selected_distances[rows, second_positions]

    nearest_indices = indices[nearest_positions]
    second_indices = indices[second_positions]
    increments = second_distances - nearest_distances
    phi = np.bincount(
        nearest_positions,
        weights=increments,
        minlength=indices.size,
    ).astype(np.float64) / distances.shape[0]
    C = float(np.mean(nearest_distances))
    F1 = float(np.max(phi))
    J = float(C / s_r + lambda_ * F1 / s_r)

    return StructuralQuantities(
        selected_indices=indices,
        nearest_indices=nearest_indices.astype(np.int64, copy=False),
        second_indices=second_indices.astype(np.int64, copy=False),
        nearest_distances=nearest_distances,
        second_distances=second_distances,
        C=C,
        phi=phi,
        F1=F1,
        J=J,
    )


def most_fragile_representative(state: StructuralQuantities) -> int:
    """Return j_star; selected indices are sorted so argmax ties go to lowest index."""
    return int(state.selected_indices[int(np.argmax(state.phi))])


def backup_score(
    distance_matrix: ArrayLike,
    state: StructuralQuantities,
    candidate: int,
    j_star: int,
) -> float:
    """Evaluate the manuscript-defined B_r(v | j_star)."""
    distances = _validate_distance_matrix(distance_matrix)
    candidate = int(candidate)
    j_star = int(j_star)
    if candidate < 0 or candidate >= distances.shape[0]:
        raise IndexError("Insertion candidate is outside the recording.")
    owner_rows = np.flatnonzero(state.nearest_indices == j_star)
    fallback = state.second_distances[owner_rows]
    recovered = fallback - np.minimum(fallback, distances[owner_rows, candidate])
    return float(np.sum(recovered, dtype=np.float64) / distances.shape[0])


def build_insertion_pool(
    distance_matrix: ArrayLike,
    state: StructuralQuantities,
    j_star: int,
    q: int,
) -> tuple[NDArray[np.int64], dict[int, float], list[int]]:
    """Build and rank the frozen backup insertion pool."""
    distances = _validate_distance_matrix(distance_matrix)
    n_windows = distances.shape[0]
    selected_set = set(int(index) for index in state.selected_indices)
    unselected = [index for index in range(n_windows) if index not in selected_set]
    if not unselected:
        return np.empty(0, dtype=np.int64), {}, []
    q = max(0, min(int(q), len(unselected)))
    if q == 0:
        return np.empty(0, dtype=np.int64), {}, []

    j_star = int(j_star)
    owner_rows = np.flatnonzero(state.nearest_indices == j_star).astype(int).tolist()
    available = [index for index in owner_rows if index not in selected_set]
    available_set = set(available)
    visited_neighbors: list[int] = []

    if len(available) < q:
        neighbors = [int(index) for index in state.selected_indices if int(index) != j_star]
        neighbors.sort(key=lambda index: (float(distances[j_star, index]), index))
        for representative in neighbors:
            visited_neighbors.append(representative)
            region = np.flatnonzero(state.nearest_indices == representative)
            for index_value in region:
                index = int(index_value)
                if index not in selected_set and index not in available_set:
                    available.append(index)
                    available_set.add(index)
            if len(available) >= q:
                break

    scores = {
        index: backup_score(distances, state, index, j_star)
        for index in available
    }
    ranked = sorted(available, key=lambda index: (-scores[index], index))
    return np.asarray(ranked[:q], dtype=np.int64), scores, visited_neighbors


def build_removal_pool(
    state: StructuralQuantities,
    j_star: int,
    q: int,
) -> tuple[NDArray[np.int64], dict[int, dict[str, int | float]]]:
    """Build the frozen rank-sum redundancy pool while protecting j_star."""
    j_star = int(j_star)
    eligible = [int(index) for index in state.selected_indices if int(index) != j_star]
    if not eligible:
        return np.empty(0, dtype=np.int64), {}
    q = max(0, min(int(q), len(eligible)))
    if q == 0:
        return np.empty(0, dtype=np.int64), {}

    phi_by_index = state.phi_by_index()
    ownership_size = {
        index: int(np.count_nonzero(state.nearest_indices == index))
        for index in eligible
    }
    phi_order = sorted(eligible, key=lambda index: (phi_by_index[index], index))
    ownership_order = sorted(eligible, key=lambda index: (ownership_size[index], index))
    rank_phi = {index: rank for rank, index in enumerate(phi_order, start=1)}
    rank_ownership = {index: rank for rank, index in enumerate(ownership_order, start=1)}
    details: dict[int, dict[str, int | float]] = {}
    for index in eligible:
        details[index] = {
            "phi": float(phi_by_index[index]),
            "ownership_size": ownership_size[index],
            "rank_phi": rank_phi[index],
            "rank_ownership": rank_ownership[index],
            "R": rank_phi[index] + rank_ownership[index],
        }
    ranked = sorted(eligible, key=lambda index: (int(details[index]["R"]), index))
    return np.asarray(ranked[:q], dtype=np.int64), details


def find_best_swap(
    distance_matrix: ArrayLike,
    current_state: StructuralQuantities,
    removal_pool: Sequence[int],
    insertion_pool: Sequence[int],
    *,
    s_r: float,
    lambda_: float,
) -> SwapChoice | None:
    """Evaluate every admitted pair exactly and return the deterministic best."""
    distances = _validate_distance_matrix(distance_matrix)
    selected_set = set(int(index) for index in current_state.selected_indices)
    removals = sorted(int(index) for index in np.asarray(removal_pool).reshape(-1))
    insertions = sorted(int(index) for index in np.asarray(insertion_pool).reshape(-1))
    best: SwapChoice | None = None

    for removed in removals:
        if removed not in selected_set:
            raise ValueError("Removal pool contains an unselected index.")
        for inserted in insertions:
            if inserted in selected_set:
                raise ValueError("Insertion pool contains an already-selected index.")
            trial = sorted((selected_set - {removed}) | {inserted})
            trial_state = evaluate_structural_quantities(
                distances, trial, s_r=s_r, lambda_=lambda_
            )
            improvement = float(current_state.J - trial_state.J)
            choice = SwapChoice(removed, inserted, improvement, trial_state)
            if best is None:
                best = choice
            elif improvement > best.improvement:
                best = choice
            elif improvement == best.improvement and (removed, inserted) < (
                best.removed_index,
                best.inserted_index,
            ):
                best = choice
    return best


def _phi_records(state: StructuralQuantities) -> list[dict[str, int | float]]:
    return [
        {"index": int(index), "phi": float(phi)}
        for index, phi in zip(state.selected_indices, state.phi, strict=True)
    ]


def select_farw(
    X_scaled: ArrayLike,
    budget: float,
    *,
    budget_reference_n: int = DEFAULT_BUDGET_REFERENCE_N,
    lambda_: float = DEFAULT_LAMBDA,
    beta: float = DEFAULT_BETA,
    epsilon: float = DEFAULT_EPSILON,
    max_accepted_swaps: int = DEFAULT_MAX_ACCEPTED_SWAPS,
    exhaustive_reference: bool = False,
    scaling_metadata: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Select a FARW-v1 subset from one already-scaled recording.

    ``exhaustive_reference=True`` is development-only.  It uses every
    unselected insertion and every selected removal except protected j_star;
    exact objective equations and all other rules remain unchanged.
    """
    X = require_finite_matrix(X_scaled, name="X_scaled")
    original = X.copy()
    n_windows, n_features = X.shape
    K = fixed_budget_count(budget, budget_reference_n)
    if K < 2:
        raise ValueError(
            f"FARW singleton fragility requires K >= 2; fixed budget produced K={K}."
        )
    if K > n_windows:
        raise ValueError(
            f"Fixed nominal budget is infeasible: K={K} exceeds N_actual={n_windows}. "
            "FARW does not clamp, pad, duplicate, synthesize, or truncate."
        )
    lambda_ = float(lambda_)
    beta = float(beta)
    epsilon = float(epsilon)
    max_accepted_swaps = int(max_accepted_swaps)
    if not isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError("lambda must be finite and nonnegative.")
    if not isfinite(beta):
        raise ValueError("beta must be finite.")
    if not isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative.")
    if max_accepted_swaps < 0:
        raise ValueError("max_accepted_swaps must be nonnegative.")

    distances = pairwise_euclidean(X, beta=beta)
    s_r, reference_indices = compute_reference_scale(distances)
    initialization_order = medoid_seeded_farthest_first(distances, K)
    state = evaluate_structural_quantities(
        distances, initialization_order, s_r=s_r, lambda_=lambda_
    )
    initial_state = state
    q_requested = candidate_pool_size(K)
    trace: list[dict[str, Any]] = []
    accepted_swaps = 0
    stop_reason: str

    if max_accepted_swaps == 0:
        stop_reason = "max_accepted_swaps_reached"
    else:
        while accepted_swaps < max_accepted_swaps:
            iteration = accepted_swaps + 1
            j_star = most_fragile_representative(state)
            selected_set = set(int(index) for index in state.selected_indices)

            if exhaustive_reference:
                insertion_pool = np.asarray(
                    [index for index in range(n_windows) if index not in selected_set],
                    dtype=np.int64,
                )
                removal_pool = np.asarray(
                    [index for index in state.selected_indices if int(index) != j_star],
                    dtype=np.int64,
                )
            else:
                insertion_pool, _, _ = build_insertion_pool(
                    distances, state, j_star, q_requested
                )
                removal_pool, _ = build_removal_pool(state, j_star, q_requested)

            trace_row: dict[str, Any] = {
                "iteration": iteration,
                "C": float(state.C),
                "F1": float(state.F1),
                "J": float(state.J),
                "j_star": j_star,
                "insertion_pool_size": int(insertion_pool.size),
                "removal_pool_size": int(removal_pool.size),
                "chosen_u": None,
                "chosen_v": None,
                "objective_improvement": None,
                "accepted": False,
                "J_after": float(state.J),
            }

            if insertion_pool.size == 0:
                trace.append(trace_row)
                stop_reason = "insertion_pool_empty"
                break

            choice = find_best_swap(
                distances,
                state,
                removal_pool,
                insertion_pool,
                s_r=s_r,
                lambda_=lambda_,
            )
            if choice is None:
                trace.append(trace_row)
                stop_reason = "no_admissible_improving_swap"
                break

            trace_row.update(
                {
                    "chosen_u": int(choice.removed_index),
                    "chosen_v": int(choice.inserted_index),
                    "objective_improvement": float(choice.improvement),
                    "J_after": float(choice.trial_state.J),
                }
            )
            if choice.improvement > epsilon:
                trace_row["accepted"] = True
                trace.append(trace_row)
                state = choice.trial_state
                accepted_swaps += 1
            else:
                trace.append(trace_row)
                stop_reason = "no_admissible_improving_swap"
                break
        else:
            stop_reason = "max_accepted_swaps_reached"

    selected_indices = state.selected_indices.copy()
    if selected_indices.size != K or np.unique(selected_indices).size != K:
        raise AssertionError("FARW violated exact budget or uniqueness.")
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= n_windows:
        raise AssertionError("FARW produced an out-of-range index.")
    if not np.array_equal(X, original):
        raise AssertionError("FARW modified X_scaled in place.")
    for row in trace:
        if row["accepted"] and not float(row["objective_improvement"]) > epsilon:
            raise AssertionError("An accepted swap did not strictly exceed epsilon.")
        if row["accepted"] and not float(row["J_after"]) < float(row["J"]):
            raise AssertionError("An accepted swap failed to strictly decrease J.")

    metadata: dict[str, Any] = {
        "method": "FARW-v1",
        "N_actual": int(n_windows),
        "N_features": int(n_features),
        "feature_space_protocol": "all_input_features_no_post_FARW_feature_selection",
        "N_budget_reference": int(budget_reference_n),
        "budget": float(budget),
        "budget_rounding": "floor(budget * N_budget_reference + 0.5)",
        "K": int(K),
        "selected_indices": selected_indices.astype(int).tolist(),
        "index_base": 0,
        "lambda": float(lambda_),
        "beta": float(beta),
        "distance": "euclidean_scaled_features" if beta == 0.0 else "euclidean_scaled_features_plus_beta_t_over_T",
        "epsilon": float(epsilon),
        "max_accepted_swaps": int(max_accepted_swaps),
        "accepted_swaps": int(accepted_swaps),
        "stop_reason": stop_reason,
        "s_r": float(s_r),
        "s_r_reference_indices": reference_indices.astype(int).tolist(),
        "s_r_rule": "median_all_positive_distances_to_half_up_linspace_q_ref_min_16_N",
        "C_initial": float(initial_state.C),
        "F1_initial": float(initial_state.F1),
        "J_initial": float(initial_state.J),
        "C_final": float(state.C),
        "F1_final": float(state.F1),
        "J_final": float(state.J),
        "phi_final": _phi_records(state),
        "initialization_order": initialization_order.astype(int).tolist(),
        "tie_policy": GLOBAL_TIE_POLICY,
        "swap_tie_pair_order": "lexicographic_(removed_index_inserted_index)",
        "candidate_pool_rule": {
            "mode": "exhaustive_reference_development_only"
            if exhaustive_reference
            else "restricted_FARW_v1",
            "q_formula": "max(5, ceil(sqrt(K)))",
            "q_requested": int(q_requested),
            "j_star_protected": True,
            "insertion": "top_q_B_after_fragile_region_then_nearest_representative_regions",
            "removal": "top_q_smallest_rank_phi_plus_rank_ownership",
        },
        "scaling_specification": dict(scaling_metadata)
        if scaling_metadata is not None
        else {
            "core_expectation": "X_scaled was scaled using complete-outer-training median/IQR statistics",
            "fit_per_recording": False,
        },
        "uses_labels": False,
        "returns_observed_indices_only": True,
        "exhaustive_reference": bool(exhaustive_reference),
        "iteration_trace": trace,
    }
    return selected_indices, metadata
