# Equations and implementing functions

Equation numbers follow the paper.

| Eq. | Quantity | Implementation |
| --- | --- | --- |
| (1) | Near-field response band per object R_c and per closing pair R_ij | `p1_sim/p1_nearfield_fusion_sim.py` `main()` (`analytic_budget.csv`: `band_R_m = v_max * T_HORIZON`); pair bands in `p1_bench/nf_experiments.py` `_exp_p_nodes()` (`pair_budget`) |
| (2) | Latency error e(τ) = v·τ, budget τ_max, pair distance bias Δd_ij | `p1_sim/p1_nearfield_fusion_sim.py` `run_e1()` (`analytic_bias_m`), `main()` (`tau_max_ms_for_0p5m`); `nf_experiments._exp_p_nodes()` |
| (3) | Measurement model z_m (bias β_m ℓ(u), range-dependent noise σ_m) | `p1_bench/nf_sim.py` `sense()` |
| (4) | Detection probability with LOS and condition shift Δa | `nf_sim._logistic()`, `nf_sim.los_many()`, `nf_sim.sense()`; calibrated curves via `nf_sim.apply_calib()` |
| (5) | Capture, stamp and arrival times (offset δ, drift η, latency τ) | `nf_sim.sense()` (`CLOCKS`, `t_cap`, `t_stamp`, `t_arr`) |
| (6) | NIS and soft gating | `p1_bench/nf_fusion.py` `_apply()` (`CHI2_99`); OOSM re-filtering in `run_filter()` |
| (7) | Late fusion by inverse-variance weighting | `nf_fusion.late_fuse()` |
| (8) | Learned residual fusion and modality-dropout objective | `nf_fusion.learn_features()`, `build_training()` (`n_aug=4`, `p_keep=0.6`), `train_mlp()`, `learned_fuse()` |
| (9) | P_ε and macro average over 15 subsets | `p1_bench/nf_experiments.py` `exp_a()` (`macro_p_le`) |
| (10) | Modality dependency D_m (GNSS-RTK) | `make_sensitivity_fig.py` and `compare_calibration.py` (`d_gnss`); MM-Fi: `mmfi_dropout.py` (`dependency`) |
| (11) | Risk-underestimation rate ρ_over | `nf_experiments._exp_p_nodes()` (`overest_closing`); summary in `summarize_p.py` |
| (12) | Recall drop to detection-curve shift Δa_c | `aihub_port_prep.py` `cmd_convert()` (`--p-ref 0.9`) |
