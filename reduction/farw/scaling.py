"""Outer-training-fitted robust scaling for FARW-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatMatrix = NDArray[np.float64]


def require_finite_matrix(X: ArrayLike, *, name: str = "X") -> FloatMatrix:
    """Return a float64 2-D view/copy after strict shape and finite checks."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [N_windows, N_features] matrix.")
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature.")
    if not np.all(np.isfinite(data)):
        bad = np.argwhere(~np.isfinite(data))[0]
        raise ValueError(
            f"{name} contains NaN or Inf at zero-based row {int(bad[0])}, "
            f"feature {int(bad[1])}."
        )
    return data


@dataclass(frozen=True)
class RobustScalingModel:
    """Feature-wise median/IQR transform fitted on an outer-training matrix."""

    median: NDArray[np.float64]
    q1: NDArray[np.float64]
    q3: NDArray[np.float64]
    iqr: NDArray[np.float64]
    scale: NDArray[np.float64]
    zero_iqr_mask: NDArray[np.bool_]
    n_fit_rows: int
    n_features: int
    quantile_method: str = "linear"

    def transform(self, X: ArrayLike) -> FloatMatrix:
        data = require_finite_matrix(X)
        if data.shape[1] != self.n_features:
            raise ValueError(
                f"X has {data.shape[1]} features, but the fitted scaler expects "
                f"{self.n_features}."
            )
        scaled = (data - self.median) / self.scale
        if not np.all(np.isfinite(scaled)):
            raise FloatingPointError("Robust scaling produced a non-finite value.")
        return np.asarray(scaled, dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        return {
            "formula": "(x - outer_training_median) / outer_training_IQR",
            "fit_scope": "complete_outer_training_partition_only",
            "center": "feature_wise_median",
            "scale": "feature_wise_IQR_Q3_minus_Q1",
            "quantile_method": self.quantile_method,
            "zero_iqr_policy": "replace_scale_with_1",
            "zero_iqr_feature_indices": np.flatnonzero(self.zero_iqr_mask).astype(int).tolist(),
            "n_fit_rows": int(self.n_fit_rows),
            "n_features": int(self.n_features),
        }


def fit_robust_scaler(X_outer_training: ArrayLike) -> RobustScalingModel:
    """Fit the frozen FARW-v1 robust scaler on complete outer-training rows.

    NumPy's explicit ``method='linear'`` quantile convention is used so that
    the numerical definition is reproducible across calls.
    """
    data = require_finite_matrix(X_outer_training, name="X_outer_training")
    median = np.median(data, axis=0)
    q1, q3 = np.percentile(data, [25.0, 75.0], axis=0, method="linear")
    iqr = q3 - q1
    zero_iqr_mask = iqr == 0.0
    scale = iqr.copy()
    scale[zero_iqr_mask] = 1.0

    arrays = [median, q1, q3, iqr, scale, zero_iqr_mask]
    for value in arrays:
        value.setflags(write=False)

    return RobustScalingModel(
        median=median,
        q1=q1,
        q3=q3,
        iqr=iqr,
        scale=scale,
        zero_iqr_mask=zero_iqr_mask,
        n_fit_rows=int(data.shape[0]),
        n_features=int(data.shape[1]),
    )


def fit_robust_scaler_from_recordings(
    recordings: Iterable[ArrayLike],
) -> RobustScalingModel:
    """Fit one common scaler after concatenating all outer-training recordings."""
    matrices = [
        require_finite_matrix(X, name=f"outer_training_recording[{index}]")
        for index, X in enumerate(recordings)
    ]
    if not matrices:
        raise ValueError("At least one outer-training recording is required.")
    n_features = matrices[0].shape[1]
    if any(matrix.shape[1] != n_features for matrix in matrices[1:]):
        raise ValueError("All outer-training recordings must have the same feature count.")
    return fit_robust_scaler(np.concatenate(matrices, axis=0))
