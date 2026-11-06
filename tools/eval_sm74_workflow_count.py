"""Paired SM74/Tight batch-count sweep; production evaluator and models unchanged."""
from __future__ import annotations

import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_sm74_ablation import read, rows, sha, save
from tools.eval_alibaba_sl111 import identity

OUT = ROOT / 'results/workflow_count_sm74_20261017'
CONFIG = ROOT / 'configs/v3_sm_74_new_eval_50wf.json'
INVENTORY = ROOT / 'results/success_seed_screening_20261013/panel_plan.json'
PROTOCOL = ROOT / 'docs/WORKFLOW_COUNT_SM74_20261017.md'
COUNTS = [50, 100, 150, 200, 250]
SEEDS = list(range(742000, 742020))
MODES = ['d3qn_sac', 'dqn_edge', 'dream_dqn', 'oo_actor_critic']
LABELS = dict(zip(MODES, ['D3QN+SAC', 'DQN-Edge', 'DReaM-DQN', 'OOAC']))


def check_sources(plan):
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen source changed: {path}')


def scenario(count):
    config = read(CONFIG)
    config['generator'].update(num_workflows=count, dataset_scale='small')
    config['simulation']['deadline_type'] = 'tight'
    return config


def validate(data, mode, count, plan, complete=False):
    keys = [r['eval_seed'] for r in data]
    if (keys != SEEDS[:len(keys)] or len(keys) > 20
            or (complete and len(keys) != 20)):
        raise ValueError('Missing, duplicate or reordered records')
    for r in data:
        if (r['mode'], r['scale'], r['deadline_type'], r['num_workflows']) != (mode, 'small', 'tight', count):
            raise ValueError('Wrong evaluation protocol')
        sample = plan['samples'][str(count)][str(r['eval_seed'])]
        for name in ('physical_sha256', 'deadlines_sha256', 'tasks', 'edges'):
            if r[name] != sample[name]:
                raise ValueError('Unpaired workload/deadline fingerprint')
        if not math.isfinite(r['total_energy']) or r['total_energy'] <= 0:
            raise ValueError('Invalid total energy')
        for name in ('deadline_success_rate', 'reliability_success_rate'):
            if not math.isfinite(r[name]) or not 0 <= r[name] <= 1:
                raise ValueError('Invalid success metric')
        if not math.isclose(r['energy_per_workflow'], r['total_energy']/count, rel_tol=1e-12):
            raise ValueError('Wrong normalization denominator')


def baseline_gate(actual, reference):
    for name in ('total_energy', 'deadline_success_rate', 'reliability_success_rate'):
        if not math.isclose(actual[name], reference[name], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'50-workflow Full reproduction failed: {name}')


def summarize(data):
    return dict(n=len(data), energy_mean_J=statistics.mean(r['total_energy'] for r in data),
                energy_sample_sd_J=statistics.stdev(r['total_energy'] for r in data),
                energy_per_workflow_mean_J=statistics.mean(r['energy_per_workflow'] for r in data),
                energy_per_workflow_sample_sd_J=statistics.stdev(r['energy_per_workflow'] for r in data),
                esp_at_90=statistics.mean(r['deadline_success_rate'] >= .9-1e-9 for r in data),
                reliability_attainment=statistics.mean(r['reliability_success_rate'] >= 1-1e-9 for r in data))


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError('Existing experiment: no duplicate prepare or automatic resume')
    old = read(INVENTORY)
    models = [r for r in old['runs'] if r['resource'] == 'sm']
    assert [r['mode'] for r in models] == MODES
    cfg = read(CONFIG)
    assert (cfg['resources']['num_edge'], cfg['resources']['num_cloud'],
            cfg['generator']['arrival_rate'], cfg['generator']['profile']) == (56, 18, .03, 'sec_job_fcfs_v1')
    paths = {CONFIG, INVENTORY, PROTOCOL, Path(__file__), ROOT/'tests/test_sm74_workflow_count.py'}
    code_inventory = ROOT/'results/ablation_sm74_new_eval_20261015/frozen_plan.json'
    paths.add(code_inventory)
    paths.update(ROOT/p for p in read(code_inventory)['source_sha256']
                 if Path(p).as_posix().startswith(('src/', 'tools/')) or p.endswith('.xml'))
    paths.update((ROOT/'src/baselines').rglob('*.py'))
    paths.update(ROOT/p for p in ('tools/train_dqn_edge.py', 'tools/train_dream.py',
                                 'tools/train_oo_actor_critic.py', 'src/device_utils.py'))
    paths.add(ROOT/'configs/resources_edge_cloud_dvfs_v3.json')
    for model in models:
        for field in ('checkpoint', 'training_config', 'config'):
            path = Path(model[field])
            if sha(path) != model[field+'_sha256']:
                raise ValueError('Model/config identity changed')
            paths.add(path)
        source = Path(model['source'])/'comparison_rows.jsonl'
        if sha(source) != old['original_file_sha256'][str(source)]:
            raise ValueError('Original result changed')
        paths.add(source)
    plan = dict(counts=COUNTS, modes=MODES, seeds=SEEDS, deadline='tight', dataset='scientific_small',
                servers=74, arrival_rate=.03, workers_per_model=1, torch_threads=1,
                model_initialization_seed=742000, server_seed=42, conditional_selection=False,
                model_reset='each count starts at the same checkpoint; seeds continuous within count',
                models=models, samples={}, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(cfg, 42)
    for count in COUNTS:
        plan['samples'][str(count)] = {}
        for seed in SEEDS:
            workflows = e.build_workflows(scenario(count), resources, seed)
            if len(workflows) != count or len({w.workflow_id for w in workflows}) != count:
                raise ValueError('Wrong workflow count or duplicate IDs')
            for previous in COUNTS[:COUNTS.index(count)]:
                prefix = identity(workflows[:previous])['physical_sha256']
                if prefix != plan['samples'][str(previous)][str(seed)]['physical_sha256']:
                    raise ValueError('Counts do not share the same physical workload prefix')
            sample = identity(workflows)
            if any(w.relative_deadline <= 0 or w.absolute_deadline <= w.arrival_time for w in workflows):
                raise ValueError('Nonpositive deadline budget')
            sample.update(arrival_first=workflows[0].arrival_time, arrival_last=workflows[-1].arrival_time,
                          relative_deadline_mean=statistics.mean(w.relative_deadline for w in workflows),
                          reference_response_mean=statistics.mean(w.reference_makespan for w in workflows))
            plan['samples'][str(count)][str(seed)] = sample
        print(f'[preflight] workflows={count} samples=20 paired_prefix=True', flush=True)
    check_sources(plan)
    OUT.mkdir()
    save(OUT/'frozen_plan.json', plan)
    for count in COUNTS:
        save(OUT/f'config_n{count}.json', scenario(count))
        plan['source_sha256'][str((OUT/f'config_n{count}.json').relative_to(ROOT))] = sha(OUT/f'config_n{count}.json')
    save(OUT/'seed_schedule.json', {'small': {'tight': SEEDS}})
    plan['source_sha256'][str((OUT/'seed_schedule.json').relative_to(ROOT))] = sha(OUT/'seed_schedule.json')
    save(OUT/'frozen_plan.json', plan)
    print('[prepared] 400 rollouts; no training; four serial algorithm sequences', flush=True)


def run_mode(mode):
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    run_root = OUT/mode
    run_root.mkdir()  # Refuse repeated launches, even with only partial results.
    save(run_root/'execution.json', dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                                       status='running', workers=1, torch_threads=1))
    model = next(r for r in plan['models'] if r['mode'] == mode)
    reference = {r['eval_seed']:r for r in rows(Path(model['source'])/'comparison_rows.jsonl')
                 if r['scale'] == 'small' and r['deadline_type'] == 'tight'}
    prefix = dict(d3qn_sac='d3qn', dqn_edge='dqn-edge', dream_dqn='dream', oo_actor_critic='oo')[mode]
    with (OUT/f'{mode}.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        for count in COUNTS:
            check_sources(plan)
            directory = run_root/f'n{count}'
            argv = ['eval_all_baselines', '--config', str(OUT/f'config_n{count}.json'),
                    f'--{prefix}-checkpoint', model['checkpoint'], '--modes', mode,
                    '--scales', 'small', '--deadline-types', 'tight', '--seed-start', '742000',
                    '--num-seeds', '20', '--workers', '1', '--seed-schedule', str(OUT/'seed_schedule.json'),
                    '--output-dir', str(directory)]
            if mode != 'd3qn_sac':
                argv += [f'--{prefix}-config', model['training_config']]
            with patch.object(sys, 'argv', argv):
                args = e.parse_args()
            original_build, original_write = e.build_workflows, e.write_rows
            samples = {}

            def checked_build(config, resources, seed, smoke_test=False):
                if smoke_test or config['generator']['num_workflows'] != count:
                    raise ValueError('Count protocol changed')
                workflows = original_build(config, resources, seed, smoke_test=False)
                actual = identity(workflows)
                expected = plan['samples'][str(count)][str(seed)]
                if any(actual[k] != expected[k] for k in actual):
                    raise ValueError('Runtime sample identity changed')
                samples[seed] = expected
                return workflows

            def checked_write(path, data):
                for r in data:
                    sample = samples[r['eval_seed']]
                    r.update({k:sample[k] for k in ('physical_sha256', 'deadlines_sha256', 'tasks', 'edges',
                             'arrival_first', 'arrival_last', 'relative_deadline_mean', 'reference_response_mean')})
                    r['selection_is_conditional'] = False
                    if mode == 'd3qn_sac' and count == 50:
                        baseline_gate(r, reference[r['eval_seed']])
                validate(data, mode, count, plan)
                original_write(path, data)

            with patch.object(e, 'build_workflows', checked_build), patch.object(e, 'write_rows', checked_write):
                _, data, _ = e.evaluate(args)
            validate(data, mode, count, plan, complete=True)
            manifest = read(directory/'experiment_manifest.json')
            if (manifest['workers'], manifest['model_initialization_seed'], manifest['server_seed'],
                manifest['evaluation_seed_schedule']) != (1, 742000, 42, {'small':{'tight':SEEDS}}):
                raise ValueError('Worker/state initialization differs')
            save(directory/'completion.json', dict(records=20, **summarize(data)))
            print(f'[count-complete] mode={mode} workflows={count} records=20', flush=True)
        check_sources(plan)
        save(run_root/'completion.json', dict(records=100, all_counts=COUNTS, source_hashes_verified=True))
        print(f'[model-complete] {mode} records=100', flush=True)
    return mode


def report():
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    summary = []
    complete = True
    for mode in MODES:
        for count in COUNTS:
            directory = OUT/mode/f'n{count}'
            path = directory/'comparison_rows.jsonl'
            data = rows(path) if path.exists() else []
            validate(data, mode, count, plan)
            done = (directory/'completion.json').exists()
            complete = complete and done
            if done:
                validate(data, mode, count, plan, complete=True)
                stats = summarize(data)
                if read(directory/'completion.json') != dict(records=20, **stats):
                    raise ValueError('Completion statistics disagree with raw results')
                summary.append(dict(mode=mode, num_workflows=count, **stats))
    save(OUT/'summary.json', dict(all_complete=complete, cells=summary))
    if summary:
        with (OUT/'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    lines = ['# SM/Tight 工作流数量敏感性', '', '原始未筛选种子；0.03/s固定，DDL按原共享参考重新生成。',
             '总计算能耗J均值±样本SD；不是到达率敏感性，不是新增训练重复。', '',
             '| Workflow数量 | 方法 | 能耗J | 单位工作流能耗J | ESP@90 | 可靠性达标 |',
             '|---:|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['num_workflows']} | {LABELS[r['mode']]} | {r['energy_mean_J']:.2f} ± {r['energy_sample_sd_J']:.2f} | "
                     f"{r['energy_per_workflow_mean_J']:.2f} | {r['esp_at_90']:.0%} | {r['reliability_attainment']:.0%} |")
    lines += ['', f'全部完成：{complete}；完整统计单元：{len(summary)}/20。失败全部保留。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if complete:
        if not all((OUT/m/'completion.json').exists() for m in MODES):
            raise ValueError('Missing model completion')
        plot(summary)
    return dict(all_complete=complete, completed_cells=len(summary))


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT/'plot'))
    from plot_main_style_options import configure_style
    configure_style()
    colors = dict(zip(MODES, ['#A8C0E4', '#F5AA83', '#BBDDB2', '#BAE0E3']))
    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    for mode in MODES[1:]+MODES[:1]:
        points = sorted((r for r in summary if r['mode'] == mode), key=lambda r:r['num_workflows'])
        assert [r['num_workflows'] for r in points] == COUNTS
        ax.errorbar(COUNTS, [r['energy_mean_J'] for r in points],
                    yerr=[r['energy_sample_sd_J'] for r in points], color=colors[mode],
                    linewidth=2, capsize=3, elinewidth=1, label=LABELS[mode])
    ax.set(xlabel='Number of workflows', ylabel='Total energy (J)', xticks=COUNTS)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6), useMathText=True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.25)
    fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    output = OUT/'figures'
    output.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'energy_vs_workflows.{suffix}', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'report':
        print(json.dumps(report()))
    else:
        lock = OUT/'execution.json'
        with lock.open('x', encoding='utf-8') as handle:
            json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                           models=MODES, concurrent_models=4, no_automatic_resume=True), handle)
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_mode, m):m for m in MODES}
            for future in as_completed(futures):
                print(f'[finished] mode={future.result()}', flush=True)
        print('[sweep-complete] '+json.dumps(report()), flush=True)


if __name__ == '__main__':
    main()
