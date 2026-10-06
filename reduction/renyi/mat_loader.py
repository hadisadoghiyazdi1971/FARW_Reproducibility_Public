"""Strict read-only loader for the FARW MATLAB feature files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


@dataclass(frozen=True)
class RecordingFeatures:
    recording_id: str
    source_path: Path
    X: NDArray[np.float64]
    feature_names: tuple[str, ...]
    feature_schema_version: str


def discover_recordings(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {root}")
    files = sorted(path for path in root.glob("*.mat") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .mat feature files found in: {root}")
    return files


def _matlab_strings(value: object) -> tuple[str, ...]:
    values = np.atleast_1d(value).reshape(-1)
    return tuple(str(item) for item in values)


def load_recording_features(
    source_path: str | Path,
    *,
    expected_n_features: int | None = 40,
) -> RecordingFeatures:
    """Load only the named FeatureNames fields; Label/FileName are not read."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {path}")

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    required = {"FeatureNames", "FeatureSchemaVersion", "SigData"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{path.name}: missing MATLAB variables: {missing}")

    feature_names = _matlab_strings(payload["FeatureNames"])
    if expected_n_features is not None and len(feature_names) != expected_n_features:
        raise ValueError(
            f"{path.name}: expected {expected_n_features} FeatureNames, found {len(feature_names)}."
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"{path.name}: FeatureNames contains duplicates.")
    forbidden = {"Label", "FileName"}.intersection(feature_names)
    if forbidden:
        raise ValueError(f"{path.name}: non-feature fields listed in FeatureNames: {sorted(forbidden)}")

    rows = np.atleast_1d(payload["SigData"]).reshape(-1)
    if rows.size < 1:
        raise ValueError(f"{path.name}: SigData contains no windows.")

    X = np.empty((rows.size, len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        fields = set(getattr(row, "_fieldnames", ()) or ())
        absent = [name for name in feature_names if name not in fields]
        if absent:
            raise ValueError(f"{path.name}: row {row_index} lacks feature fields {absent}.")
        for feature_index, feature_name in enumerate(feature_names):
            value = np.asarray(getattr(row, feature_name))
            if value.size != 1:
                raise ValueError(
                    f"{path.name}: row {row_index}, feature {feature_name} is not scalar."
                )
            X[row_index, feature_index] = float(value.reshape(-1)[0])

    if not np.all(np.isfinite(X)):
        bad = np.argwhere(~np.isfinite(X))[0]
        raise ValueError(
            f"{path.name}: NaN/Inf at row {int(bad[0])}, feature "
            f"{feature_names[int(bad[1])]}; no row was deleted."
        )

    return RecordingFeatures(
        recording_id=path.stem,
        source_path=path.resolve(),
        X=X,
        feature_names=feature_names,
        feature_schema_version=str(payload["FeatureSchemaVersion"]),
    )
