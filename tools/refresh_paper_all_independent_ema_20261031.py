"""Refresh derived paper data from completed EMA-reset CSVs; never evaluate."""
import argparse
import contextlib
import csv
import difflib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import refresh_paper_independent_ema_20261031 as main
from plot.paper_csv_data import read_csv, sha, stats, independent_extra_rows
from plot.plot_alibaba_energy_only import load_data, SOURCE as ALIBABA
from plot.plot_workload_sensitivity_options import load_verified, GRIDS, MODES

OUT = ROOT / 'docs/figures/paper_refresh_all_independent_ema_20261031'
BACKUP = ROOT / 'paper/manuscript/backups/independent_ema_all_20261031'


def supplemental():
    ali, ali_sources, _ = load_data(csv_source=True, independent_ema=True)
    old_ali, _, _ = load_data(csv_source=True)
    for key in ali:
        if key[0] != 'dqn_edge':
            assert (ali[key] == old_ali[key]).all()
    sens, sens_sources = load_verified(csv_source=True, independent_ema=True)
    old_sens, _ = load_verified(csv_source=True)
    assert all(sens[k] == old_sens[k] for k in sens if k[1] != 'dqn_edge')
    return ali, ali_sources, sens, sens_sources


def write_csv(name, records):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit():
    _, tests = main.audit()
    _, ali_sources, sens, sens_sources = supplemental()
    OUT.mkdir(parents=True, exist_ok=True)
    cells, raw, protected = [], [], {}
    sources = [s['path'] for s in ali_sources]
    sources += [str(ROOT / path) for path in sens_sources if path.endswith('.csv')]
    for filename in dict.fromkeys(sources):
        path = Path(filename)
        data = read_csv(path)
        kind = 'alibaba' if path in [Path(s['path']) for s in ali_sources] else (
            'workflow_count' if 'workflow_count' in filename else 'arrival_rate')
        for actual in (path, path.with_suffix('.jsonl')):
            protected[str(actual.relative_to(ROOT))] = sha(actual)
        settings = {}
        for row in data:
            value = row['deadline_type'] if kind == 'alibaba' else (
                row['num_workflows'] if kind == 'workflow_count' else row['arrival_rate'])
            settings.setdefault((row['mode'], str(value)), []).append(row)
            raw.append(dict(experiment=kind, mode=row['mode'], setting=str(value),
                eval_seed=row['eval_seed'], num_workflows=row['num_workflows'],
                total_energy_J=row['total_energy'], deadline_success_rate=row['deadline_success_rate'],
                reliability_success_rate=row['reliability_success_rate'],
                source_csv=str(path.relative_to(ROOT))))
        for (mode, setting), sample in settings.items():
            result = dict(experiment=kind, mode=mode, setting=setting, **stats(sample))
            for metric in ('esp_at_90', 'reliability_attainment'):
                lo, hi = main.wilson(round(20*result[metric]), 20)
                result[metric+'_wilson95_low'], result[metric+'_wilson95_high'] = lo, hi
            cells.append(result)
    assert len(raw) == 1040 and len(cells) == 52
    assert len({(r['experiment'], r['mode'], r['setting'], r['eval_seed']) for r in raw}) == 1040
    assert sum('dqn_edge_independent_ema_extra' in r['source_csv'] for r in raw) == 260
    write_csv('supplementary_original_csv_rows_J.csv', raw)
    write_csv('supplementary_cells_J.csv', cells)
    changes = []
    for group in ('alibaba', 'workflow_count', 'arrival_rate'):
        records, _ = independent_extra_rows(group)
        for row in records:
            if row['deadline_success_rate'] < .9-1e-9 or row['reliability_success_rate'] < 1-1e-9:
                changes.append(dict(experiment=group, setting=row['setting'], seed=row['eval_seed'],
                    deadline_success_rate=row['deadline_success_rate'], reliability_success_rate=row['reliability_success_rate']))
    assert changes == [dict(experiment='arrival_rate', setting='rate_0p20', seed=742011,
                            deadline_success_rate=.86, reliability_success_rate=1.0)]
    metadata = dict(main_records=2160, supplementary_records=1040, changed_dqn_edge_records=800,
        unchanged_other_algorithm_records=2400, raw_sources_modified=False, units='J',
        protected_source_sha256=protected, main_audit=str(main.OUT.relative_to(ROOT)/'audit.json'),
        dqn_edge_ema_protocol='Restore checkpoint EMA before each realization; update within realization',
        main_seeds=list(range(20)), supplementary_seeds=list(range(742000,742020)),
        failures=changes, correction_family=81, backup=str(BACKUP.relative_to(ROOT)))
    (OUT/'audit.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    report = ('# All-experiment independent-EMA paper refresh\n\n'
        'Read original CSVs and cross-check JSONL. Only the 800 DQN-Edge records change: '
        'main 540, Alibaba 60, workflow count 100, arrival rate 100. Other algorithms retain 2400 unchanged records. '
        'Ablations and framework figure are unchanged. Raw files are not overwritten.\n\n'
        'Main workload seeds: 0--19. Supplementary workload seeds: 742000--742019. '
        'Each DQN-Edge realization starts from the saved checkpoint EMA and updates it within that realization.\n\n'
        'Arrival 0.20/s: DQN-Edge ESP@90=95%, Rel.=100%; seed 742011 has on-time fraction 0.86. '
        'All failures are retained. All other new DQN-Edge cells attain 100% for both metrics.\n\n'
        'Main energy inference: 81 exact two-sided paired Wilcoxon tests, with family-wide Bonferroni and BH corrections. '
        'Wilson intervals describe batch-level attainment and are not Wilcoxon tests. '
        'Inference remains conditional on the reported workloads and fixed checkpoints; it is not a population or training-repeat guarantee.\n\n'
        f'Backup: `{BACKUP.relative_to(ROOT)}`.\n\n'
        'Plot reproduction:\n\n'
        '- `python plot/plot_current_generalization_energy.py --independent-ema --separators`\n'
        '- `python plot/plot_alibaba_energy_only.py --independent-ema --separators`\n'
        '- `python plot/plot_workload_sensitivity_options.py --independent-ema --single-column`\n')
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print('Verified 3200 paper records; DQN-Edge 800 new, other algorithms 2400 unchanged.')
    return tests


def revised_manuscript(original, cells, tests):
    with contextlib.redirect_stdout(io.StringIO()):
        updated = main.manuscript_patch(original,cells,tests,all_experiments=True)
    ali, _, sens, _ = supplemental()
    reductions = []
    for deadline in main.DEADLINES:
        ours = ali['d3qn_sac',deadline].mean()
        baseline = min(ali[mode,deadline].mean() for mode in main.ORDER[:-1])
        reductions.append(100*(1-ours/baseline))
    old = r'are 24.25\%, 27.16\%, and 28.48\%.'
    new = 'are ' + ', '.join(f'{r:.2f}\\%' for r in reductions[:2]) + f', and {reductions[2]:.2f}\\%.'
    assert updated.count(old)==1
    updated=updated.replace(old,new,1)
    old = r'whereas DQN-Edge and DReaM-DQN attain 90\%.'
    new = (f"whereas DQN-Edge and DReaM-DQN attain {100*sens['arrival','dqn_edge',.2]['esp_at_90']:.0f}\\% "
           f"and {100*sens['arrival','dream_dqn',.2]['esp_at_90']:.0f}\\%, respectively.")
    assert updated.count(old)==1
    return updated.replace(old,new,1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex-patch',action='store_true')
    args=parser.parse_args()
    if args.tex_patch:
        original=main.TEX.read_text(encoding='utf-8')
        _,_,cells=main.main_panel(independent_ema=True)
        tests=json.loads((main.OUT/'audit.json').read_text())['tests']
        updated=revised_manuscript(original,cells,tests)
        diff=list(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),n=3))
        import re
        print('*** Begin Patch\n*** Update File: '+str(main.TEX))
        print(re.sub(r'^@@ .*@@$','@@',''.join(diff[2:]),flags=re.MULTILINE),end='')
        print('*** End Patch')
    else:
        audit()
