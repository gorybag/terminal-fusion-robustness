# Revision analyses (v1.1.0)

Additional analyses run for the revised manuscript. All use `p1_bench/revision_experiments.py`
with calibration level L3 (`calibration/calib_L3.json`) and the terminal profile; outputs are in `results/revision/`.

| Part | What it tests | Output |
| --- | --- | --- |
| analytic | Response band and latency budget for T_h = 5-15 s, e_budget = 0.25-1.0 m, speed limits 20-30 km/h; size of second-order terms | `analytic_sensitivity.csv`, `analytic_second_order.json` |
| core | Six methods incl. hard chi-square gate and fast covariance intersection; MAE, RMSE, median, p95, max; thresholds 0.25/0.5/1.0 m; episode-bootstrap CIs; paired Wilcoxon with Holm correction | `core_methods.csv`, `core_summary.json` |
| sensitivity | One-at-a-time perturbation of 16 assumed parameters (thermal, LiDAR rain/fog, GNSS-RTK float/jump, latency, clock); ranking stability (Kendall tau) | `sensitivity_parameters.csv` |
| dropout | Modality-dropout probability 0.1-0.6 x 1-8 augmented subsets | `dropout_sweep.csv` |
| pairs | Class accuracy and vehicle-worker risk underestimation with CIs, signed relative-distance error, overestimation with and without timestamp alignment (1 and 2 nodes) | `pairs_collision.csv`, `pairs_latency.csv` |
| sync | PTP / NTP / unsynchronized with CIs | `sync_levels.csv` |
| scale_nodes | Scene density x1 / x2 and node layouts N1 / N2 / N3 | `scale_nodes.csv` |
| examples | Failure examples: external truck through an occlusion gap; closing-pair distance error with and without timestamp alignment | `failure_examples.png/.pdf` |

Main findings (5 seeds unless noted): learned fusion with modality dropout ranked first against the hard-gate and
covariance-intersection baselines (macro 0.823 vs 0.448-0.701, Holm-corrected p < 0.001), the ranking of the four
fusion structures was unchanged under all 16 parameter perturbations (3 seeds), NTP was indistinguishable from PTP,
and the benefit of modality dropout grew with the dropout probability and the number of augmented subsets.

Reproduce: `python p1_bench/revision_experiments.py --part all --calib calibration/calib_L3.json --out results/revision`
(batch 2: `--part scale_nodes,examples`).
