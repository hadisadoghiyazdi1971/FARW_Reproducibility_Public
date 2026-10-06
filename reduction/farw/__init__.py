"""FARW-v1: deterministic fragility-aware reduction of observed windows."""

from .core import (
    DEFAULT_BETA,
    DEFAULT_BUDGET_REFERENCE_N,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA,
    DEFAULT_MAX_ACCEPTED_SWAPS,
    StructuralQuantities,
    build_insertion_pool,
    build_removal_pool,
    candidate_pool_size,
    compute_reference_scale,
    evaluate_structural_quantities,
    fixed_budget_count,
    medoid_seeded_farthest_first,
    pairwise_euclidean,
    select_farw,
)
from .scaling import RobustScalingModel, fit_robust_scaler, fit_robust_scaler_from_recordings

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_BUDGET_REFERENCE_N",
    "DEFAULT_EPSILON",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_ACCEPTED_SWAPS",
    "RobustScalingModel",
    "StructuralQuantities",
    "build_insertion_pool",
    "build_removal_pool",
    "candidate_pool_size",
    "compute_reference_scale",
    "evaluate_structural_quantities",
    "fit_robust_scaler",
    "fit_robust_scaler_from_recordings",
    "fixed_budget_count",
    "medoid_seeded_farthest_first",
    "pairwise_euclidean",
    "select_farw",
]
