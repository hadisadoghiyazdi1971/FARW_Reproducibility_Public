"""Global K-means-centroid window selection without temporal stratification."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

from .common import (
    base_metadata,
    global_unique_assignment,
    resolve_target,
    verify_selected_indices,
)


def select_centroid(
    X: ArrayLike,
    budget: float,
    *,
    budget_reference_n: int = 350,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tolerance: float = 1e-4,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Run global K-means, then globally project centroids to unique rows."""
    random_state = int(random_state)
    n_init = int(n_init)
    max_iter = int(max_iter)
    tolerance = float(tolerance)
    if n_init < 1:
        raise ValueError("n_init must be at least 1.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and greater than zero.")

    standardized, constant_mask, n_actual, n_features, k_target = resolve_target(
        X, budget, budget_reference_n
    )
    metadata = base_metadata(
        method="global_kmeans_centroid_projection",
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
            "clustering": "sklearn_kmeans_lloyd",
            "initialization": "k-means++",
            "n_init": n_init,
            "max_iter": max_iter,
            "tolerance": tolerance,
            "projection": "global_linear_assignment_squared_euclidean",
            "centroids_are_continuous": True,
        }
    )

    if k_target == n_actual:
        selected = np.arange(n_actual, dtype=np.int64)
        metadata.update(
            {
                "iterations": 0,
                "converged": True,
                "kmeans_inertia": 0.0,
                "projection_total_squared_cost": 0.0,
                "selection_shortcut": "K_target_equals_N_actual_all_rows_selected",
            }
        )
        return selected, metadata

    model = KMeans(
        n_clusters=k_target,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
        algorithm="lloyd",
    )
    model.fit(standardized)
    selected, projection_cost = global_unique_assignment(
        standardized, model.cluster_centers_
    )
    verify_selected_indices(selected, k_target=k_target, n_actual=n_actual)
    metadata.update(
        {
            "iterations": int(model.n_iter_),
            "converged": bool(model.n_iter_ < max_iter),
            "kmeans_inertia": float(model.inertia_),
            "projection_total_squared_cost": projection_cost,
        }
    )
    return selected, metadata

