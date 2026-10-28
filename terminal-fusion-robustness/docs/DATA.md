# Data

None of the datasets below are redistributed in this repository. Only aggregate statistics derived from them
(calibration files in `calibration/`, result tables in `results/`) are published.

| Dataset | Use in the paper | How to obtain | Terms |
| --- | --- | --- | --- |
| nuScenes v1.0-mini | LiDAR and camera detection curves (L1–L3) | https://www.nuscenes.org (registration) | Non-commercial use, attribution |
| AI Hub "Logistics infrastructure data for connected ports" (validation set, crane-view YT images and labels) | Camera recall by condition (L2, L3) | https://www.aihub.or.kr (registration; access rules apply to users outside Korea) | AI Hub data use policy: no redistribution or third-party provision; attribution required |
| AI model provided with the dataset (YOLOv4, safety zone) | Site-trained detector (L3) | Same AI Hub page | AI Hub data use policy |
| YOLO11x (COCO-pretrained) | Generic detector (L2) | https://github.com/ultralytics/ultralytics | AGPL-3.0 (Ultralytics) |
| MM-Fi (environment E01, subjects S01–S10) | Real multi-sensor replication | https://github.com/ybhbingo/MMFi_dataset | Research use, see dataset terms |
| Physical AI Smart Spaces (MTMC_Tracking_2026) | 95th-percentile worker speed | https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces | CC BY 4.0 |

Optional: AI Hub "autonomous driving 3D data in special environments" is read by `adrive3d_prep.py` as a
consistency check of LiDAR condition offsets; it does not change the reported results.

Attribution required by AI Hub: this work used the "Logistics infrastructure data for connected ports" and the
associated AI model built through the "Intelligent Information Industry Infrastructure" program of the Ministry
of Science and ICT and the National Information Society Agency (NIA), Korea.
