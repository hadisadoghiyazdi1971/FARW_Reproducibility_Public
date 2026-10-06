# Final minimal public-release audit

## Scope

This release provides final selection implementations, frozen selected-window indices, condition-disjoint scenario metadata, compact manuscript reference results, and direct regeneration of Figures 6 and 7. It does **not** provide the raw acoustic recordings or complete descriptor dataset and does not claim full end-to-end reproduction of the numerical experiments.

## Validation of the final copy

- The six historical analysis-runner folders were removed. No retained reduction, figure, or public-tool module imports them.
- The four retained dependencies are NumPy, SciPy, scikit-learn, and Matplotlib. Their status and version provenance are recorded in `requirements.txt` and `ENVIRONMENT.md`.
- Every retained Python file parsed and imported from the repository root.
- The public index verifier checked the frozen bank, Random NPZ archive, CSV manifest, published SHA-256 values, all 3,200 deterministic cases, and all 24,000 Random cases. The Random seed set is exactly 1–30. This is an internal-consistency verification, not an independent reselection from excluded descriptors.
- All 11 `config/` files and all 10 `reference_results/` CSVs are byte-for-byte unchanged from the verified source copy used to build this release.
- Both figure generators ran from the final repository using only its own reference CSVs. Figure 6 used the 30-seed Random structural mean; Figure 7 used the 30-seed Random scenario means and checked the distinct seed-wise-worst summary. The regenerated PDFs and PNGs matched the inherited outputs byte-for-byte.
- No raw recording, descriptor MAT, manuscript draft, Article 1/2 file, or absolute Windows path was found in this final copy. The `FROZEN_SOURCE` identifier in the Figure 7 script refers to a CSV under this repository's `reference_results/`; it is not an archival path.
- `PUBLIC_RELEASE_MANIFEST.csv` inventories every final file other than itself, with relative path, byte size, and SHA-256. The inventory was regenerated after cleanup.

## Boundaries and remaining decision

The selection implementations are provided as code, but no full raw-data experiment rerun is certified. Compact reference tables are evidence for the completed manuscript analyses, not substitute descriptor recordings. Before a public GitHub upload, the project owner must choose a code licence and confirm permission to distribute the figure assets or any additional material. See `LICENSE_NOTE.md`. No Git repository or remote has been created here.
