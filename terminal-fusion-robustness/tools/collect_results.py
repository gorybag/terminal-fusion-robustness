#!/usr/bin/env python3
"""
collect_results.py - copy the result tables and figures reported in the paper from a local project into this
repository's results/ folder. Raw or per-image data from licensed datasets are never copied.

Dry run (prints the plan):
  python tools/collect_results.py --project ~/spaceiq_1 \
      --l3-run runs/20261021_1459 --l1-run runs/20261020_2237 \
      --site-eval runs/L3A --generic-eval runs/20261021_0955 --mmfi-run runs/20261021_1148 \
      --figures paper_figs
Add --apply to copy. Paths after --project are relative to the project folder.
"""
import argparse, csv, fnmatch, glob, hashlib, os, shutil

MAX_MB = 20
DENY = ['*port_labels*.csv', '*port_dets*.csv', '*.weights', '*.pt', '*.onnx', '*.zip', '*.npy', '*.npz',
        '*.jpg', '*.jpeg', '*.pcd', '*.bin', 'paths.env', '*.log']
PLAN = [  # (source key, glob relative to that folder, destination subfolder)
    ('l3', 'sim_port/*.csv', 'L3/sim_port'), ('l3', 'sim_port/summary.json', 'L3/sim_port'),
    ('l3', 'bench_port_s*/A_missing_modality.csv', 'L3/bench'), ('l3', 'bench_port_s*/summary.json', 'L3/bench'),
    ('l3', 'bench_port_s*/*sync*.csv', 'L3/bench'), ('l3', 'p_summary.md', 'L3'), ('l3', 'p_summary*.csv', 'L3'),
    ('l3', 'sensitivity.csv', 'L3'),
    ('l1', 'sim_port/*.csv', 'L1/sim_port'), ('l1', 'bench_port_s*/A_missing_modality.csv', 'L1/bench'),
    ('l1', 'bench_port_s*/summary.json', 'L1/bench'), ('l1', 'p_summary.md', 'L1'),
    ('site', 'port_eval*.json', 'port_detector/site_trained'), ('site', 'port_eval*recall*.csv', 'port_detector/site_trained'),
    ('generic', 'port_eval*.json', 'port_detector/generic'),
    ('mmfi', 'mmfi*.json', 'mmfi'), ('mmfi', 'mmfi*.csv', 'mmfi'),
    ('figs', '*.png', 'figures'), ('figs', '*.pdf', 'figures'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--l3-run', required=True)
    ap.add_argument('--l1-run')
    ap.add_argument('--site-eval')
    ap.add_argument('--generic-eval')
    ap.add_argument('--mmfi-run')
    ap.add_argument('--figures')
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(__file__), '..'))
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    proj = os.path.abspath(os.path.expanduser(a.project))
    repo = os.path.abspath(os.path.expanduser(a.repo))
    print('project :', proj); print('repo    :', repo)
    if not os.path.isdir(proj):
        raise SystemExit('ERROR: project folder not found: %s' % proj)
    if not os.path.isdir(os.path.join(repo, 'results')):
        raise SystemExit('ERROR: repository folder not found (no results/): %s  -> pass --repo <unzipped folder>' % repo)
    runs = sorted(glob.glob(os.path.join(proj, 'runs', '*')))
    src = {'l3': a.l3_run, 'l1': a.l1_run, 'site': a.site_eval, 'generic': a.generic_eval, 'mmfi': a.mmfi_run, 'figs': a.figures}
    missing = []
    for k, v in list(src.items()):
        if v and not os.path.isdir(os.path.join(proj, os.path.expanduser(v))):
            missing.append('%s -> %s' % (k, v)); src[k] = None
    if missing:
        print('\nWARNING: these source folders do not exist and are skipped:')
        for m in missing: print('   ', m)
        print('available run folders:')
        for r in runs: print('   ', os.path.relpath(r, proj))
        print()
    out = os.path.join(repo, 'results')
    rows, skipped = [], []
    for key, pat, dst in PLAN:
        if not src.get(key):
            continue
        for p in sorted(glob.glob(os.path.join(proj, src[key], pat))):
            name = os.path.basename(p)
            if any(fnmatch.fnmatch(name, d) for d in DENY):
                skipped.append((p, 'excluded type')); continue
            mb = os.path.getsize(p) / 1e6
            if mb > MAX_MB:
                skipped.append((p, '%.1f MB > %d MB' % (mb, MAX_MB))); continue
            sub = ''
            if 'bench_port_s' in p:   # keep the seed in the file name
                sub = os.path.basename(os.path.dirname(p)).replace('bench_port_', '') + '_'
            target = os.path.join(out, dst, sub + name)
            rows.append(dict(file=os.path.relpath(target, os.path.dirname(out)), source=os.path.relpath(p, proj),
                             bytes=os.path.getsize(p), sha256=sha(p)))
            print('%-60s <- %s' % (rows[-1]['file'], rows[-1]['source']))
            if a.apply:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(p, target)
    for p, why in skipped:
        print('SKIP', p, '(' + why + ')')
    if a.apply and rows:
        with open(os.path.join(out, 'MANIFEST.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('\n%d files %s' % (len(rows), 'copied' if a.apply else 'planned (add --apply to copy)'))


if __name__ == '__main__':
    main()
