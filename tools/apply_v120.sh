#!/usr/bin/env bash
# Run inside the repository clone: updates README, results/README.md and CITATION.cff for v1.2.0
set -euo pipefail
cd "${1:-$HOME/terminal-fusion-robustness}"
grep -q "Second-round revision analyses" README.md || cat >> README.md <<'MD'

## Second-round revision analyses (v1.2.0)

`p1_bench/revision2_experiments.py` adds the ranking in each seed, baselines tuned on the training episodes,
harder scenarios (rain/fog in every episode, correlated node outage, compound), node-position and stack-density
variations, and the uncertainty and perturbation of the calibrated detection curves. See `docs/REVISION2.md`;
outputs are in `results/revision2/`.
MD
grep -q "revision2/" results/README.md 2>/dev/null || cat >> results/README.md <<'MD'
| `revision2/` | Second-round revision analyses (ranking per seed, tuned baselines, stress scenarios, layout variation, calibration uncertainty); see `docs/REVISION2.md` |
MD
grep -q "Second revision of the manuscript" results/README.md || cat >> results/README.md <<'MD'

Second revision of the manuscript: the new Table 6 (error metrics of all methods) is taken from
`revision/core_summary.json`; the former Tables 6 and 7 are Tables 7 and 8; the former Figs. 6 and 7 are the two
panels of Fig. 6; Fig. 3 uses `figures/Fig3_evaluation_setting_combined.png` (`tools/fig3_combined.py`).
MD
sed -i '' 's/^version: "1.1.0"/version: "1.2.0"/' CITATION.cff
sed -i '' 's/^date-released: .*/date-released: "'"$(date +%Y-%m-%d)"'"/' CITATION.cff
echo "README, results/README.md and CITATION.cff updated:"; grep -n "^version\|^date-released" CITATION.cff
