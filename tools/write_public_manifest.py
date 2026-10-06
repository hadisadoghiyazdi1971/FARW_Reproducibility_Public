"""Write SHA-256 inventory for public files (excluding this manifest itself)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PUBLIC_RELEASE_MANIFEST.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != OUTPUT)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relative_path", "size_bytes", "sha256"))
        for path in files:
            writer.writerow((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)))
    print(f"Manifest rows: {len(files)}")


if __name__ == "__main__":
    main()
