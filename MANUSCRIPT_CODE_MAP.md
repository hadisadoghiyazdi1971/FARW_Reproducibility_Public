# Manuscript-to-public-release map

The labels below distinguish provided implementation from compact reference evidence. Only Figures 6 and 7 are directly regenerated from the included numerical sources; the full experiments require data not bundled here.

| Manuscript item | Public path | Status |
|---|---|---|
| Feature extraction / 40-descriptor construction | `feature_extraction/extract_features_FARW_40_corrected_V8.m` | IMPLEMENTATION PROVIDED |
| FARW selection | `reduction/farw/` | IMPLEMENTATION PROVIDED |
| Global Medoid and Global Centroid | `reduction/medoid/`, `reduction/centroid/` | IMPLEMENTATION PROVIDED |
| Rényi RDS/RDS-IS and Random | `reduction/renyi/`, `reduction/random/` | IMPLEMENTATION PROVIDED |
| Frozen selected windows and seeds 1–30 | `config/public_selection_bank.json.gz`, `config/random_30seed_selected_indices.npz`, CSV manifest | REFERENCE RESULT PROVIDED |
| Condition-disjoint splits | `config/scenario*_recordings.csv`, `config/final_protocol.json` | REFERENCE RESULT PROVIDED |
| RFE ranking and fixed P2P–EnR–SE representation | `reference_results/rfe_ranking_by_scenario.csv` | REFERENCE RESULT PROVIDED |
| Deterministic coverage and fragility | `reference_results/structural_deterministic_source.csv` | REFERENCE RESULT PROVIDED |
| Random 30-seed structural means | `reference_results/random_30seed_structural_means.csv` | REFERENCE RESULT PROVIDED |
| Figure 6 | `figures/generate_structural_tradeoff.py` and the preceding structural CSVs | FIGURE DIRECTLY REPRODUCIBLE |
| Table 4 classification summary | `reference_results/classification_deterministic_summary.csv`, `reference_results/random_30seed_scenario_mean.csv`, `reference_results/random_30seed_seedwise_worst_diagnostic.csv` | REFERENCE RESULT PROVIDED |
| Final deterministic/FULL scenario cells | `reference_results/classification_unified_3f_source.csv` | REFERENCE RESULT PROVIDED |
| Figure 7 | `figures/generate_scenario_mcc.py`, final scenario CSVs | FIGURE DIRECTLY REPRODUCIBLE |
| Figure 2 | `figures/Figure_2.pdf` | MANUSCRIPT FIGURE PROVIDED |
| Figure 4 | `figures/Figure_4.pdf` | MANUSCRIPT FIGURE PROVIDED |
| Figure 5 | `figures/Figure_5.pdf` | MANUSCRIPT FIGURE PROVIDED |
| Table 5: 40 versus three descriptors | `reference_results/40f_vs_3f_summary.csv` | REFERENCE RESULT PROVIDED |
| Table 6: leave-one-feature-out ablation | `reference_results/ablation_summary.csv` | REFERENCE RESULT PROVIDED |
| Table 7: cross-classifier comparison | `reference_results/cross_classifier_mean_mcc.csv` | REFERENCE RESULT PROVIDED |

"Reference result provided" does not mean the underlying numerical experiment is rerunnable from this data-free repository.
