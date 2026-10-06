# FARW: minimal public reproducibility release

Fragility-Aware Reduction of Windows (FARW) selects a fixed number of observed training windows within each recording. This repository provides the final FARW and benchmark selection implementations, frozen selected-window indices, condition-disjoint scenario metadata, compact manuscript reference results, and scripts that regenerate Figures 6 and 7 from those results.

This is **not** a full end-to-end rerun package. The 160 raw acoustic recordings and complete 40-descriptor recording dataset are not bundled. The reference CSVs document completed analyses; they do not substitute for those experimental data. No claim is made that RFE, structural evaluation, classification, ablation, or cross-classifier experiments can be rerun from this package alone.

## Contents

| Path | Public role |
|---|---|
| `feature_extraction/extract_features_FARW_40_corrected_V8.m` | Final MATLAB source for constructing the 40 window descriptors. |
| `reduction/` | Final FARW, Global Medoid, Global Centroid, Rényi RDS/RDS-IS, and Random implementations with required preprocessing. |
| `config/public_selection_bank.json.gz` | Frozen observed-window indices for the four deterministic methods and Random. |
| `config/random_30seed_selected_indices.npz` and its CSV manifest | Saved Random selections using **exactly seeds 1–30**. |
| `config/scenario*_recordings.csv` | Train/test recording membership for the three condition-disjoint scenarios. |
| `config/final_protocol.json` and `config/selection_bank_provenance.json` | Protocol parameters and published index-bank provenance. |
| `reference_results/` | Compact final results for RFE, structural and classification analyses, the 40-versus-3 comparison, ablation, and cross-classifier evaluation. |
| `figures/` | Manuscript Figures 2, 4, and 5 as PDFs; Figure 6–7 plotting scripts and current PDF/PNG outputs. |
| `tools/verify_public_indices.py` | Checks the included bank, Random archive, manifest, case grids, and published file hashes. |
| `tools/write_public_manifest.py` | Regenerates the file-size/SHA-256 inventory for this repository. |

Window selection used all 40 descriptors after within-recording standardisation; downstream classification used the fixed P2P, EnR, and SE descriptors. The Random comparator uses seeds 1–30, not a single seed. Non-Random initialisation constants are unchanged.

The MATLAB V8 source implements extraction of the manuscript's 40 window descriptors; the raw recordings needed to run it are not bundled.

## Directly reproducible figures

Install the packages in `requirements.txt`, then run from the repository root:

```text
python -m figures.generate_structural_tradeoff
python -m figures.generate_scenario_mcc
```

Both scripts use only included CSVs under `reference_results/`; they do not rerun selectors or classifiers. Figure 6 uses the 30-seed Random structural mean, while Figure 7 uses the 30-seed scenario MCC means and checks the separate seed-wise minimum summary. The scripts write vector PDF and 600-dpi PNG outputs into `figures/`.

To check the included selected indices and their internal provenance:

```text
python tools/verify_public_indices.py
```

This check establishes agreement among the public bank, saved Random archive, and manifest; it cannot independently reconstruct selections from excluded descriptor recordings. `MANUSCRIPT_CODE_MAP.md` maps each manuscript item to implementation, reference result, or directly reproducible figure. `data/README.md` explains the data boundary. No Git repository or remote is supplied by this release folder.
