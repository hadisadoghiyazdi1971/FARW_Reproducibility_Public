"""The single authoritative preprocessing implementation for active selectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


N_FEATURES = 40


def require_finite_40_matrix(X: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [N_actual, 40] matrix.")
    if data.shape[0] < 1 or data.shape[1] != N_FEATURES:
        raise ValueError(
            f"{name} must contain at least one row and exactly 40 features; "
            f"received shape {data.shape}."
        )
    if not np.all(np.isfinite(data)):
        row, feature = np.argwhere(~np.isfinite(data))[0]
        raise ValueError(
            f"{name} contains NaN/Inf at zero-based row {int(row)}, "
            f"feature {int(feature)}."
        )
    return data


def matrix_sha256(X: ArrayLike) -> str:
    data = require_finite_40_matrix(X)
    canonical = np.ascontiguousarray(data, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ZScoreResult:
    X_z: NDArray[np.float64]
    mean: NDArray[np.float64]
    sample_std: NDArray[np.float64]
    scale_used: NDArray[np.float64]
    constant_mask: NDArray[np.bool_]
    input_sha256: str
    standardized_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "per_recording_zscore",
            "formula": "(X - recording_mean) / recording_sample_std",
            "std_ddof": 1,
            "fit_scope": "individual_recording_only",
            "zero_std_policy": "replace_scale_with_1_and_set_transformed_column_to_zero",
            "constant_feature_indices": np.flatnonzero(self.constant_mask).astype(int).tolist(),
            "input_sha256": self.input_sha256,
            "standardized_sha256": self.standardized_sha256,
        }


def per_recording_zscore(X: ArrayLike) -> ZScoreResult:
    """Z-score one recording with sample standard deviation (ddof=1)."""
    data = require_finite_40_matrix(X, name="X_raw")
    original = data.copy()
    mean = np.mean(data, axis=0, dtype=np.float64)
    if data.shape[0] == 1:
        sample_std = np.zeros(data.shape[1], dtype=np.float64)
    else:
        sample_std = np.std(data, axis=0, ddof=1, dtype=np.float64)
    constant_mask = sample_std == 0.0
    scale_used = sample_std.copy()
    scale_used[constant_mask] = 1.0
    X_z = (data - mean) / scale_used
    X_z[:, constant_mask] = 0.0
    if not np.all(np.isfinite(X_z)):
        raise FloatingPointError("Per-recording Z-score produced NaN or Inf.")
    if not np.array_equal(data, original):
        raise AssertionError("Preprocessing modified the raw recording in place.")
    for array in (X_z, mean, sample_std, scale_used, constant_mask):
        array.setflags(write=False)
    return ZScoreResult(
        X_z=X_z,
        mean=mean,
        sample_std=sample_std,
        scale_used=scale_used,
        constant_mask=constant_mask,
        input_sha256=matrix_sha256(data),
        standardized_sha256=matrix_sha256(X_z),
    )

