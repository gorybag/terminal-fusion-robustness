# Results

Filled by `tools/collect_results.py` from a local run (see that script). Folder map:

| Folder | Paper item |
| --- | --- |
| `L3/sim_port/` | Table 2 (`analytic_budget.csv`), Fig. 4 (`e1_latency_error.csv`), Fig. 5 (`e2_strategy_condition.csv`) |
| `L3/bench/` | Table 5, Fig. 6 (`A_missing_modality.csv`, `summary.json` per seed) |
| `L3/` | Tables 4 and 6 (`p_summary.md`), calibration sensitivity (`sensitivity.csv`) |
| `L1/` | Same tables for calibration level L1 |
| `port_detector/` | Table 7, Fig. 7a (aggregate evaluation JSON only) |
| `mmfi/` | Table 7, Fig. 7b |
| `figures/` | Final figures of the paper |
| `MANIFEST.csv` | Source path, size and SHA-256 of every copied file |
| `revision/` | Revision analyses (robust baselines, sensitivity, dropout sweep, CIs, density and node layouts, failure examples); see `docs/REVISION.md` |
| `revision2/` | Second-round revision analyses (ranking per seed, tuned baselines, stress scenarios, layout variation, calibration uncertainty); see `docs/REVISION2.md` |

Second revision of the manuscript: the new Table 6 (error metrics of all methods) is taken from
`revision/core_summary.json`; the former Tables 6 and 7 are Tables 7 and 8; the former Figs. 6 and 7 are the two
panels of Fig. 6; Fig. 3 uses `figures/Fig3_evaluation_setting_combined.png` (`tools/fig3_combined.py`).
