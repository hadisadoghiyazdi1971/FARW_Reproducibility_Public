"""Verify the included selected-window bank against its public Random archive.

This checks internal consistency and saved-file hashes. It does not independently
reproduce selection from the excluded descriptor recordings or archival bank.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
BANK = CONFIG / "public_selection_bank.json.gz"
ARCHIVE = CONFIG / "random_30seed_selected_indices.npz"
MANIFEST = CONFIG / "random_30seed_selected_indices_manifest.csv"
PROVENANCE = CONFIG / "selection_bank_provenance.json"
SEEDS = set(range(1, 31))
BUDGETS = {0.1, 0.2, 0.3, 0.4, 0.5}
METHODS = {"farw", "medoid", "centroid", "renyi"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_indices(case: dict) -> None:
    indices = case["selected_indices"]
    count = int(case["K"])
    total = int(case["N_actual"])
    if len(indices) != count or len(set(indices)) != count:
        raise ValueError(f"Incorrect selected-index cardinality: {case['recording_id']}")
    if not all(isinstance(index, int) and 0 <= index < total for index in indices):
        raise ValueError(f"Selected index outside recording: {case['recording_id']}")


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    expected_hashes = {
        BANK: provenance["public_bank_sha256"],
        ARCHIVE: provenance["saved_random_archive_sha256"],
        MANIFEST: provenance["saved_random_manifest_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Published SHA-256 mismatch: {path.name}")

    with gzip.open(BANK, "rt", encoding="utf-8") as stream:
        bank = json.load(stream)
    deterministic = bank["deterministic_cases"]
    random_cases = bank["random_cases"]
    if bank["index_base"] != 0 or set(bank["random_seeds"]) != SEEDS:
        raise ValueError("Bank index base or Random seed set is incorrect")
    if len(deterministic) != 3200 or len(random_cases) != 24000:
        raise ValueError("Unexpected selected-index case count")
    if provenance["deterministic_case_count"] != len(deterministic) or provenance["random_case_count"] != len(random_cases):
        raise ValueError("Case count differs from public provenance")

    deterministic_grid = Counter()
    for case in deterministic:
        check_indices(case)
        method, budget = case["method"], float(case["budget"])
        if method not in METHODS or budget not in BUDGETS:
            raise ValueError("Unexpected deterministic method or budget")
        deterministic_grid[(case["recording_id"], method, budget)] += 1
    recordings = {case["recording_id"] for case in deterministic}
    if len(recordings) != 160 or len(deterministic_grid) != 3200 or any(value != 1 for value in deterministic_grid.values()):
        raise ValueError("Incomplete deterministic recording-method-budget grid")

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 24000:
        raise ValueError("Unexpected Random manifest length")
    random_grid = Counter()
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        offsets = archive["offsets"]
        indices = archive["indices"]
        if len(offsets) != 24001 or int(offsets[0]) != 0 or int(offsets[-1]) != len(indices):
            raise ValueError("Random archive offsets are incomplete")
        for position, (case, row) in enumerate(zip(random_cases, rows)):
            check_indices(case)
            seed = int(row["seed"])
            budget = float(row["budget"])
            start, stop = int(offsets[position]), int(offsets[position + 1])
            selected = indices[start:stop]
            if (
                case["method"] != "random"
                or seed not in SEEDS
                or seed != int(case["seed"]) != int(archive["seed"][position])
                or budget not in BUDGETS
                or budget != float(case["budget"])
                or budget != sorted(BUDGETS)[int(archive["budget_index"][position])]
                or case["recording_id"] != row["recording_id"]
                or case["recording_id"] not in recordings
                or int(row["recording_index"]) != int(archive["recording_index"][position])
                or int(case["K"]) != int(row["K"]) != int(archive["K"][position])
                or int(case["N_actual"]) != int(row["N_actual"]) != int(archive["N_actual"][position])
                or start != int(row["offset_start"])
                or stop != int(row["offset_stop"])
                or int(row["index_base"]) != 0
                or stop - start != int(case["K"])
                or selected.tolist() != case["selected_indices"]
                or hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest() != row["selected_indices_sha256"]
            ):
                raise ValueError(f"Random bank/archive/manifest mismatch at case {position + 1}")
            random_grid[(case["recording_id"], seed, budget)] += 1
    if len(random_grid) != 24000 or any(value != 1 for value in random_grid.values()):
        raise ValueError("Incomplete Random recording-seed-budget grid")
    if {seed for _, seed, _ in random_grid} != SEEDS:
        raise ValueError("Random seeds differ from 1--30")
    print("PASS: public bank, Random archive and manifest agree; 3200 deterministic and 24000 Random cases; seeds 1--30 only")


if __name__ == "__main__":
    main()
