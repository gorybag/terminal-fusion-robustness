#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
locate_data.py - find the downloaded datasets under a folder and write paths.env for run_pipeline.sh

    python locate_data.py <data folder>          e.g.  python locate_data.py ~/Desktop/pre_learn

Looks for (folder names may differ, the content decides):
  PORT_VL / PORT_VS  AI Hub port data: a zip whose entries contain SAF_CRANE_YT (labels) and its image zip
  PORT_VS_YT         folder with extracted SAF_CRANE_YT images (if already unzipped)
  ADRIVE_DIR         folder holding the AI Hub 3D label zips (<region>_<road>_<time>_<weather>_<scene>.json entries)
  NUSC_ROOT          folder that contains v1.0-mini/sample_annotation.json
  MMFI_E01           an E01 folder with S01.. subject folders
  PAISS_ROOT         MTMC_Tracking_2026 folder with train/ or val/ scenes that have ground_truth.json
  YOLO_CFG / YOLO_WEIGHTS  Darknet model files (*.cfg, *.weights), if present
Writes paths.env next to this script and prints what was found or is missing.
"""
import glob
import os
import re
import sys
import zipfile

ROOT = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "."))
HERE = os.path.dirname(os.path.abspath(__file__))
MAXDEPTH = 7


def walk(root):
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= MAXDEPTH:
            dirs[:] = []
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("videos", "depth_maps", "__pycache__")]
        yield d, dirs, files


def zip_names(path, n=3000):
    try:
        with zipfile.ZipFile(path) as z:
            return z.namelist()[:n]
    except Exception:
        return []


found = {}
zips, cfgs, weights = [], [], []
for d, dirs, files in walk(ROOT):
    low = {f.lower(): f for f in files}
    if "sample_annotation.json" in low and os.path.basename(d) == "v1.0-mini":
        found.setdefault("NUSC_ROOT", os.path.dirname(d))
    if os.path.basename(d) == "v1.0-mini" and "sample_annotation.json" not in low:
        pass
    if os.path.basename(d).upper() == "E01" and any(re.fullmatch(r"S\d{2}", x) for x in dirs):
        found.setdefault("MMFI_E01", d)
    if os.path.basename(d) == "MTMC_Tracking_2026" or (
            any(x in ("train", "val") for x in dirs) and glob.glob(os.path.join(d, "*", "Warehouse_*", "ground_truth.json"))):
        if glob.glob(os.path.join(d, "*", "Warehouse_*", "ground_truth.json")):
            found.setdefault("PAISS_ROOT", d)
    if any("SAF_CRANE_YT" in f.upper() and f.lower().endswith((".jpg", ".png")) for f in files[:500]):
        found.setdefault("PORT_VS_YT", d if os.path.basename(d).upper() != "YT" else os.path.dirname(d))
    for f in files:
        fl = f.lower()
        p = os.path.join(d, f)
        if fl.endswith(".zip"):
            zips.append(p)
        elif fl.endswith(".cfg"):
            cfgs.append(p)
        elif fl.endswith(".weights"):
            weights.append(p)

adrive_dirs = {}
for z in zips:
    names = zip_names(z)
    joined = " ".join(names[:3000]).upper()
    if "SAF_CRANE_YT" in joined or "/YT/" in joined:
        if any(n.lower().endswith(".json") for n in names):
            found.setdefault("PORT_VL", z)
        elif any(n.lower().endswith((".jpg", ".png")) for n in names):
            found.setdefault("PORT_VS", z)
    elif any(re.search(r"_\d{4}_[a-z]+_[a-z]+_\w+\.json$", n.lower()) for n in names[:200]):
        adrive_dirs[os.path.dirname(z)] = adrive_dirs.get(os.path.dirname(z), 0) + 1
    elif os.path.basename(z).lower() == "vs.zip" and "PORT_VS" not in found:
        found["PORT_VS"] = z                     # large image zip: entries may be listed late; accept by name
if adrive_dirs:
    found["ADRIVE_DIR"] = max(adrive_dirs, key=adrive_dirs.get)
if "PORT_VL" not in found:
    for z in zips:
        if os.path.basename(z).lower() == "vl.zip" and os.path.dirname(z) != found.get("ADRIVE_DIR"):
            found["PORT_VL"] = z
            break
if cfgs and weights:
    found["YOLO_CFG"], found["YOLO_WEIGHTS"] = sorted(cfgs)[0], sorted(weights)[0]

keys = ["PORT_VL", "PORT_VS", "PORT_VS_YT", "YOLO_CFG", "YOLO_WEIGHTS", "NUSC_ROOT", "ADRIVE_DIR", "MMFI_E01", "PAISS_ROOT"]
lines = [f'export {k}="{found[k]}"' for k in keys if k in found]
open(os.path.join(HERE, "paths.env"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"searched: {ROOT}\n")
for k in keys:
    print(f"  {k:13s} {found.get(k, '-- not found')}")
print(f"\nwritten: {os.path.join(HERE, 'paths.env')}")
if "PORT_VS_YT" not in found and "PORT_VS" in found:
    print(f'\nYT images not extracted yet:  unzip "{found["PORT_VS"]}" "*SAF_CRANE_YT*" -d "{os.path.dirname(found["PORT_VS"])}/VS_YT"'
          f'\nthen run this script again.')
if "NUSC_ROOT" not in found:
    print("\nnuScenes-mini: no folder named v1.0-mini with sample_annotation.json was found (unzip v1.0-mini.tgz first).")
