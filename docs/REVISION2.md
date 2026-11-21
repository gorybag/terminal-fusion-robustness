# Second-round revision analyses (v1.2.0)

Additional analyses for the second revision of the manuscript. All use `p1_bench/revision2_experiments.py`
with calibration level L3 (`calibration/calib_L3.json`), the terminal profile and one node; outputs are in
`results/revision2/`. The standard setting reproduces the `core` part of `revision_experiments.py` exactly
(same seeds, episodes, models and numbers).

| Part | What it tests | Output |
| --- | --- | --- |
| seeds | Ranking of the six methods in each of the five seeds (each seed generates a different stack layout); seed-level mean and SD, episode-level mean with bootstrap CI; computing cost of every method on the same CPU | `core_by_seed.csv`, `core_by_seed_summary.json`, `cost_methods.csv` |
| tuning | Equivalent optimization of the model-based baselines: grid search over process-noise scale (7 values), gate threshold (4 values) and track timeout (4 values, late fusion and covariance intersection) on the training episodes; evaluation on the test episodes; learned-fusion hyperparameters fixed in advance | `tuning_baselines.csv`, `tuning_summary.json` |
| stress | Models trained under the standard conditions and evaluated under rain/fog in every episode, a correlated failure (all infrastructure sensors of the node drop out together for 20% of the time) and both combined with NTP offsets, doubled latency and a 75% RTK float rate | `stress_scenarios.csv` |
| variation | Node moved between stack rows and to the lane side; sparser and denser stacks (line-of-sight share reported); three seeds, 20 training and 15 test episodes | `variation_layout.csv` |
| calib | (a) Bootstrap 95% intervals of the nuScenes detection-curve coefficients, resampling annotations and scenes (needs the nuScenes-mini tables); (b) ranking with the calibrated intercepts shifted by ±0.5 and ±1.0 logit and the slopes scaled by x0.7–1.4 | `calib_uncertainty.json`, `calib_perturbation.csv` |

Unit of analysis: time steps are aggregated within an episode and the 15 sensor subsets are averaged per episode;
bootstrap intervals resample episodes and Wilcoxon tests are paired by episode with Holm correction.

Reproduce:

```
python p1_bench/revision2_experiments.py --part all --calib calibration/calib_L3.json \
       --nusc-root ~/data/nuscenes --out results/revision2
```

`--part calibci` computes only the coefficient intervals (about one minute). `--quick` runs a small smoke test.
`tools/fig3_combined.py` draws Fig. 3 of the revised manuscript (the same panels as `fig_scene` and
`fig_sensor_models` in `make_figures.py`, arranged in one row).

Main findings (five seeds unless noted): learned fusion with modality dropout ranked first in every seed (macro average
0.81–0.84, 0.11–0.13 above the next method) and the order of the six methods was the same in every seed;
the tuned baselines changed by at most 0.01 (0.69–0.71) and stayed 0.10–0.12 below it
(Holm-corrected p < 0.001); the ranking of the four structures was unchanged for the other node positions and stack
densities (three seeds; line-of-sight share 0.21–0.48) and for the perturbed calibrated coefficients (three seeds;
intercepts up to ±1.0 logit, slopes x0.7–1.4); under rain/fog in every episode, the correlated node outage and
their combination, its macro average fell to 0.773, 0.815 and 0.723 with a margin of 0.10–0.11 over the next method,
and its GNSS-RTK dependency rose from 0.14 to 0.33 in rain/fog; scene-bootstrap 95% intervals of the
vehicle detection probability at 30 m: LiDAR 0.82 (0.77–0.90) and camera 0.67 (0.60–0.80).
