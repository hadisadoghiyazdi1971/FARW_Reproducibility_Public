"""Strict label-free loader for corrected 40-feature MATLAB recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .per_recording_zscore import N_FEATURES, require_finite_40_matrix


@dataclass(frozen=True)
class RecordingFeatures:
    recording_id: str
    source_path: Path
    X: NDArray[np.float64]
    feature_names: tuple[str, ...]
    feature_schema_version: str


def _matlab_strings(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in np.atleast_1d(value).reshape(-1))


def load_recording_features(source_path: str | Path) -> RecordingFeatures:
    """Load only declared features; Label and FileName never enter selection."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature recording does not exist: {path}")
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    required = {"FeatureNames", "FeatureSchemaVersion", "SigData"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path.name}: missing variables {missing}.")
    feature_names = _matlab_strings(payload["FeatureNames"])
    if len(feature_names) != N_FEATURES or len(set(feature_names)) != N_FEATURES:
        raise ValueError(f"{path.name}: FeatureNames must contain 40 unique names.")
    if {"Label", "FileName"}.intersection(feature_names):
        raise ValueError(f"{path.name}: non-feature fields appear in FeatureNames.")
    rows = np.atleast_1d(payload["SigData"]).reshape(-1)
    X = np.empty((rows.size, N_FEATURES), dtype=np.float64)
    for row_index, row in enumerate(rows):
        fields = set(getattr(row, "_fieldnames", ()) or ())
        absent = [name for name in feature_names if name not in fields]
        if absent:
            raise ValueError(f"{path.name}: row {row_index} lacks {absent}.")
        for feature_index, name in enumerate(feature_names):
            value = np.asarray(getattr(row, name))
            if value.size != 1:
                raise ValueError(f"{path.name}: row {row_index}, {name} is not scalar.")
            X[row_index, feature_index] = float(value.reshape(-1)[0])
    X = require_finite_40_matrix(X, name=path.name)
    X.setflags(write=False)
    return RecordingFeatures(
        recording_id=path.stem,
        source_path=path.resolve(),
        X=X,
        feature_names=feature_names,
        feature_schema_version=str(payload["FeatureSchemaVersion"]),
    )

