# Calibration files

| Level | File | Content |
| --- | --- | --- |
| L0 | (none) | Assumed parameters in `p1_bench/nf_sim.py` (`SENSORS`, `CONDITIONS`, `CLOCKS`) |
| L1 | `calib_L1.json` | Daytime LiDAR and camera detection curves and sensor timing from nuScenes (`p1_bench/nf_calib_nuscenes.py`) |
| L3 | `calib_L3.json` | L1 plus camera condition offsets (`camera_port`) from the site-trained detector on the AI Hub validation images (`aihub_port_prep.py convert`) |

L2 (generic COCO detector) could not be applied (no individual yard tractor was localized), so no L2 file exists.
The files contain aggregate statistics only (fitted coefficients, recalls, sample counts).
