#!/usr/bin/env bash
# run_collect.sh - fill results/ from your local project. Edit the two paths below, then:
#   bash tools/run_collect.sh          (dry run)
#   bash tools/run_collect.sh --apply  (copy)
set -euo pipefail
PROJECT="$HOME/spaceiq_1"                                   # project folder that contains runs/
REPO="$(cd "$(dirname "$0")/.." && pwd)"                    # this repository (found automatically)
python3 "$REPO/tools/collect_results.py" --project "$PROJECT" --repo "$REPO" \
  --l3-run runs/20261021_1459 --l1-run runs/20261020_2237 \
  --site-eval runs/L3A --generic-eval runs/20261021_0955 \
  --mmfi-run runs/20261021_1148 --figures paper_figs "$@"
