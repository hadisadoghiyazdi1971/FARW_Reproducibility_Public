"""Global K-medoids window selection without temporal stratification."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .common import (
    base_metadata,
    resolve_target,
    squared_distances,
    verify_selected_indices,
)


def _kmedoids_plus_plus(
    distances: NDArray[np.float64],
    k_target: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Seed distinct medoids with a reproducible distance-squared rule."""
    n_actual = distances.shape[0]
    medoids = np.empty(k_target, dtype=np.int64)
    selected = np.zeros(n_actual, dtype=bool)
    medoids[0] = int(rng.integers(n_actual))
    selected[medoids[0]] = True
    closest = distances[:, medoids[0]].copy()

    for position in range(1, k_target):
        weights = closest * closest
        weights[selected] = 0.0
        total = float(weights.sum())
        if total > 0.0 and np.isfinite(total):
            candidate = int(rng.choice(n_actual, p=weights / total))
        else:
            candidate = int(np.flatnonzero(~selected)[0])
        medoids[position] = candidate
        selected[candidate] = True
        closest = np.minimum(closest, distances[:, candidate])
    return medoids


def _assign_to_medoids(
    distances: NDArray[np.float64], medoids: NDArray[np.int64]
) -> NDArray[np.int64]:
    labels = np.argmin(distances[:, medoids], axis=1).astype(np.int64)
    # A medoid owns itself. This only changes equal-distance ties and prevents
    # empty clusters when different observed rows have identical features.
    labels[medoids] = np.arange(medoids.size, dtype=np.int64)
    return labels


def select_medoid(
    X: ArrayLike,
    budget: float,
    *,
    budget_reference_n: int = 350,
    random_state: int = 42,
    max_iter: int = 100,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Select global K-medoids from all actual windows.

    The objective is the sum of Euclidean distances from every standardized
    observation to its assigned observed medoid.  Updates alternate between
    nearest-medoid assignment and exact within-cluster medoid minimization.
    """
    max_iter = int(max_iter)
    random_state = int(random_state)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    standardized, constant_mask, n_actual, n_features, k_target = resolve_target(
        X, budget, budget_reference_n
    )
    metadata = base_metadata(
        method="global_kmedoids",
        n_actual=n_actual,
        n_features=n_features,
        budget_reference_n=budget_reference_n,
        budget=budget,
        k_target=k_target,
        random_state=random_state,
        constant_mask=constant_mask,
    )
    metadata.update(
        {
            "algorithm": "alternating_kmedoids_pam_style",
            "initialization": "kmedoids_plus_plus_observed_rows",
            "max_iter": max_iter,
            "representatives_are_observed_rows": True,
        }
    )

    if k_target == n_actual:
        selected = np.arange(n_actual, dtype=np.int64)
        metadata.update(
            {
                "iterations": 0,
                "converged": True,
                "objective_sum_euclidean": 0.0,
                "selection_shortcut": "K_target_equals_N_actual_all_rows_selected",
            }
        )
        return selected, metadata

    euclidean_distances = np.sqrt(squared_distances(standardized, standardized))
    rng = np.random.default_rng(random_state)
    medoids = _kmedoids_plus_plus(euclidean_distances, k_target, rng)
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        labels = _assign_to_medoids(euclidean_distances, medoids)
        updated = np.empty_like(medoids)
        for cluster in range(k_target):
            members = np.flatnonzero(labels == cluster)
            within_cluster_costs = euclidean_distances[np.ix_(members, members)].sum(axis=1)
            # members are in ascending original-index order; np.argmin thus
            # also supplies a deterministic smallest-index tie break.
            updated[cluster] = int(members[int(np.argmin(within_cluster_costs))])
        if np.array_equal(updated, medoids):
            converged = True
            break
        medoids = updated

    final_labels = _assign_to_medoids(euclidean_distances, medoids)
    objective = float(euclidean_distances[np.arange(n_actual), medoids[final_labels]].sum())
    selected = medoids.astype(np.int64, copy=False)
    verify_selected_indices(selected, k_target=k_target, n_actual=n_actual)
    metadata.update(
        {
            "iterations": int(iterations),
            "converged": bool(converged),
            "objective_sum_euclidean": objective,
        }
    )
    return selected, metadata
