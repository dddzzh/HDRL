"""Paired arrival-rate sweep with baseline-relative deadlines held fixed."""
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
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_sm74_workflow_count import (CONFIG, MODES, LABELS, SEEDS, baseline_gate,
                                           check_sources, summarize)
from tools.eval_sm74_ablation import read, rows, sha, save
from tools.eval_alibaba_sl111 import identity

OUT = ROOT/'results/arrival_rate_sm74_20261017'
SOURCE = ROOT/'results/workflow_count_sm74_20261017'
PROTOCOL = ROOT/'docs/ARRIVAL_RATE_SM74_20261017.md'
RATES = [.03, .01, .06, .10, .20]  # Reproduction gate first; independent initialization per rate.
BASE_RATE = .03
FIELDS = ('physical_sha256', 'deadlines_sha256', 'tasks', 'edges',
          'workload_sha256', 'relative_budgets_sha256')


def tag(rate):
    return f'rate_{rate:.2f}'.replace('.', 'p')


def scenario(rate):
    if rate not in RATES:
        raise ValueError('Rate outside frozen grid')
    config = read(CONFIG)
    config['generator'].update(num_workflows=50, dataset_scale='small', arrival_rate=rate)
    config['simulation']['deadline_type'] = 'tight'
    return config


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample_identity(workflows):
    result = identity(workflows)
    result['workload_sha256'] = hash_json([
        (w.workflow_id, w.reliability, w.energy_budget,
         [(t.id, t.workload_cycles, t.parents, t.children) for t in w.tasks],
         [(e.src, e.dst, e.data_size_mb) for e in w.edges]) for w in workflows])
    result['relative_budgets_sha256'] = hash_json([
        (w.workflow_id, w.reference_makespan, w.deadline_multiplier, w.relative_deadline)
        for w in workflows])
    return result


def retime(base, rate):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Arrival rate must be finite and positive')
    workflows = copy.deepcopy(base)
    for w in workflows:
        # Keep baseline floats exactly at the reproduction point.
        if rate != BASE_RATE:
            w.arrival_time *= BASE_RATE/rate
            w.inter_arrival_time *= BASE_RATE/rate
            w.absolute_deadline = w.arrival_time + w.relative_deadline
        w.arrival_rate = rate
    return workflows


def validate(data, mode, rate, plan, complete=False):
    seeds = [r['eval_seed'] for r in data]
    if seeds != SEEDS[:len(seeds)] or len(seeds) > 20 or (complete and len(seeds) != 20):
        raise ValueError('Missing, duplicate or reordered records')
    for r in data:
        if (r['mode'], r['scale'], r['deadline_type'], r['num_workflows'], r['arrival_rate']) != (
                mode, 'small', 'tight', 50, rate):
            raise ValueError('Protocol differs from frozen design')
        if r['selection_is_conditional'] or r['deadline_policy'] != 'fixed_relative_budget_at_0.03':
            raise ValueError('Wrong sample selection or deadline policy')
        expected = plan['samples'][tag(rate)][str(r['eval_seed'])]
        if any(r[k] != expected[k] for k in FIELDS):
            raise ValueError('Unpaired workload, arrival or deadline fingerprint')
        if not math.isfinite(r['total_energy']) or r['total_energy'] <= 0:
            raise ValueError('Invalid energy')
        for key in ('deadline_success_rate', 'reliability_success_rate'):
            if not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
                raise ValueError('Invalid success metric')
        if not math.isclose(r['energy_per_workflow'], r['total_energy']/50, rel_tol=1e-12):
            raise ValueError('Wrong energy denominator')


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError('Existing experiment; never prepare twice or auto-resume')
    old = read(SOURCE/'frozen_plan.json')
    check_sources(old)
    if not read(SOURCE/'summary.json')['all_complete']:
        raise ValueError('Baseline experiment is not complete')
    config = scenario(BASE_RATE)
    assert (config['resources']['num_edge'], config['resources']['num_cloud']) == (56, 18)
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
    paths = {ROOT/p for p in old['source_sha256']}
    paths.update([SOURCE/'frozen_plan.json', SOURCE/'summary.json', PROTOCOL, Path(__file__),
                  ROOT/'tests/test_sm74_arrival_rate.py'])
    for model in old['models']:
        paths.update((SOURCE/model['mode']/'n50'/name) for name in (
            'comparison_rows.jsonl', 'completion.json', 'experiment_manifest.json'))
    plan = dict(rates=RATES, baseline_rate=BASE_RATE, modes=MODES, seeds=SEEDS,
                num_workflows=50, deadline='tight', servers=74, models=old['models'],
                conditional_selection=False, deadline_policy='fixed_relative_budget_at_0.03',
                model_initialization_seed=742000, server_seed=42, workers_per_model=1,
                torch_threads=1, concurrent_models=4, training=False, samples={},
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    bases = {}
    for seed in SEEDS:
        with redirect_stdout(io.StringIO()):
            bases[seed] = e.build_workflows(config, resources, seed)
        assert len(bases[seed]) == 50
        expected = old['samples']['50'][str(seed)]
        actual = identity(bases[seed])
        if any(actual[k] != expected[k] for k in actual):
            raise ValueError('Base sample no longer matches completed count experiment')
    for rate in RATES:
        plan['samples'][tag(rate)] = {}
        for seed, base in bases.items():
            workflows = retime(base, rate)
            sample, reference = sample_identity(workflows), sample_identity(base)
            for key in ('workload_sha256', 'relative_budgets_sha256'):
                if sample[key] != reference[key]:
                    raise ValueError('Retiming changed tasks or relative deadline budgets')
            if any(w.absolute_deadline <= w.arrival_time for w in workflows):
                raise ValueError('Invalid deadline budget')
            sample.update(arrival_first=workflows[0].arrival_time, arrival_last=workflows[-1].arrival_time,
                          relative_deadline_mean=statistics.mean(w.relative_deadline for w in workflows))
            plan['samples'][tag(rate)][str(seed)] = sample
        print(f'[preflight] rate={rate} seeds=20 fixed_workload_and_relative_budgets=True', flush=True)
    OUT.mkdir()
    for rate in RATES:
        path = OUT/f'{tag(rate)}.json'
        save(path, scenario(rate))
        plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    schedule = OUT/'seed_schedule.json'
    save(schedule, {'small':{'tight':SEEDS}})
    plan['source_sha256'][str(schedule.relative_to(ROOT))] = sha(schedule)
    save(OUT/'frozen_plan.json', plan)
    check_sources(plan)
    print('[prepared] records=400 models=4 reproduction_gate=80 training=False', flush=True)


def diagnostics(env, metrics):
    import numpy as np
    from src.simulator.metrics import compute_summary
    workflows = env._build_workflow_results()
    if len(workflows) != 50 or any(w['finish_time'] is None for w in workflows):
        raise ValueError('Simulation did not complete all 50 workflows')
    summary = compute_summary(workflows, env.event_log, env.servers)
    for key in ('total_energy', 'deadline_success_rate', 'reliability_success_rate'):
        if not math.isclose(summary[key], metrics[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'Rollout and independent environment metric disagree: {key}')
    cd = [w['response_time']/w['relative_deadline'] for w in workflows]
    events = sorted([(w['arrival_time'], 1) for w in workflows] +
                    [(w['finish_time'], -1) for w in workflows])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    horizon = max(w['finish_time'] for w in workflows)-min(w['arrival_time'] for w in workflows)
    utilization = sum(e['exec_time'] for e in env.event_log)/(len(env.servers)*horizon)
    if not 0 <= utilization <= 1+1e-9:
        raise ValueError('Impossible active-window server utilization')
    extra = dict(avg_task_waiting_time=summary['avg_task_waiting_time'],
                 max_task_waiting_time=summary['max_task_waiting_time'],
                 avg_workflow_response_time=summary['avg_workflow_response_time'],
                 response_over_budget_p95=float(np.percentile(cd, 95)),
                 response_over_budget_max=max(cd), active_window_utilization=utilization,
                 peak_active_workflows=peak, arrival_window_s=max(w['arrival_time'] for w in workflows)-
                 min(w['arrival_time'] for w in workflows),
                 min_workflow_reliability=min(w['reliability'] for w in workflows),
                 min_reliability_margin=min(w['reliability']-w['reliability_req'] for w in workflows))
    detail = [{k:v for k,v in w.items() if k != 'task_plans'} for w in workflows]
    return extra, detail


def run_mode(mode):
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    directory = OUT/mode
    directory.mkdir()  # Atomic duplicate/partial restart refusal.
    save(directory/'execution.json', dict(pid=os.getpid(), status='running',
         started_utc=datetime.now(timezone.utc).isoformat(), workers=1, torch_threads=1))
    model = next(m for m in plan['models'] if m['mode'] == mode)
    reference = {r['eval_seed']:r for r in rows(SOURCE/mode/'n50/comparison_rows.jsonl')}
    prefix = dict(d3qn_sac='d3qn', dqn_edge='dqn-edge', dream_dqn='dream', oo_actor_critic='oo')[mode]
    with (OUT/f'{mode}.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            for rate in RATES:
                check_sources(plan)
                output = directory/tag(rate)
                argv = ['eval_all_baselines', '--config', str(OUT/f'{tag(rate)}.json'),
                        f'--{prefix}-checkpoint', model['checkpoint'], '--modes', mode,
                        '--scales', 'small', '--deadline-types', 'tight', '--seed-start', '742000',
                        '--num-seeds', '20', '--workers', '1', '--seed-schedule', str(OUT/'seed_schedule.json'),
                        '--output-dir', str(output)]
                if mode != 'd3qn_sac':
                    argv += [f'--{prefix}-config', model['training_config']]
                with patch.object(sys, 'argv', argv):
                    args = e.parse_args()
                original_build, original_env, original_write = e.build_workflows, e.build_env, e.write_rows
                current = {}

                def checked_build(config, resources, seed, smoke_test=False):
                    if smoke_test or config['generator']['arrival_rate'] != rate:
                        raise ValueError('Unexpected evaluation configuration')
                    base_config = copy.deepcopy(config)
                    base_config['generator']['arrival_rate'] = BASE_RATE
                    workflows = retime(original_build(base_config, resources, seed), rate)
                    sample = sample_identity(workflows)
                    expected = plan['samples'][tag(rate)][str(seed)]
                    if any(sample[k] != expected[k] for k in sample):
                        raise ValueError('Runtime workload/deadline identity changed')
                    current.update(seed=seed, started=time.perf_counter())
                    return workflows

                def checked_env(*args, **kwargs):
                    env = original_env(*args, **kwargs)
                    current['env'] = env
                    return env

                def checked_write(path, data):
                    r = data[-1]
                    if r['eval_seed'] != current['seed']:
                        raise ValueError('Diagnostics seed mismatch')
                    sample = plan['samples'][tag(rate)][str(r['eval_seed'])]
                    extra, detail = diagnostics(current['env'], r)
                    if rate == BASE_RATE:
                        baseline_gate(r, reference[r['eval_seed']])
                    r.update({k:sample[k] for k in FIELDS})
                    r.update(extra, arrival_rate=rate, selection_is_conditional=False,
                             deadline_policy=plan['deadline_policy'], evaluation_wall_seconds=time.perf_counter()-current['started'])
                    validate(data, mode, rate, plan)
                    details = path/'workflow_details'
                    details.mkdir(exist_ok=True)
                    save(details/f"seed_{r['eval_seed']}.json", dict(mode=mode, rate=rate,
                         eval_seed=r['eval_seed'], workflows=detail))
                    original_write(path, data)

                with patch.object(e, 'build_workflows', checked_build), patch.object(e, 'build_env', checked_env), patch.object(e, 'write_rows', checked_write):
                    _, data, _ = e.evaluate(args)
                validate(data, mode, rate, plan, complete=True)
                manifest = read(output/'experiment_manifest.json')
                if (manifest['workers'], manifest['model_initialization_seed'], manifest['server_seed'],
                    manifest['evaluation_seed_schedule']) != (1, 742000, 42, {'small':{'tight':SEEDS}}):
                    raise ValueError('Invalid model initialization or seed sequence')
                manifest.update(arrival_rate=rate, deadline_policy=plan['deadline_policy'],
                                frozen_plan_sha256=sha(OUT/'frozen_plan.json'))
                save(output/'experiment_manifest.json', manifest)
                save(output/'completion.json', dict(records=20, **summarize(data)))
                print(f'[rate-complete] {mode} rate={rate} records=20', flush=True)
            check_sources(plan)
            save(directory/'completion.json', dict(records=100, rates=RATES, source_hashes_verified=True))
            print(f'[model-complete] {mode} records=100', flush=True)
        except Exception:
            save(directory/'failure.json', dict(traceback=traceback.format_exc(), automatic_restart=False))
            raise
    return mode


def report():
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    summary, failures = [], []
    for mode in MODES:
        for rate in sorted(RATES):
            path = OUT/mode/tag(rate)
            data = rows(path/'comparison_rows.jsonl') if (path/'comparison_rows.jsonl').exists() else []
            done = (path/'completion.json').exists()
            validate(data, mode, rate, plan, complete=done)
            if not done:
                continue
            stats = summarize(data)
            if read(path/'completion.json') != dict(records=20, **stats):
                raise ValueError('Raw results and completion disagree')
            extra = {k:statistics.mean(r[k] for r in data) for k in (
                'avg_task_waiting_time', 'avg_workflow_response_time', 'response_over_budget_p95',
                'active_window_utilization', 'peak_active_workflows')}
            extra['minimum_reliability_margin'] = min(r['min_reliability_margin'] for r in data)
            extra['joint_attainment'] = statistics.mean(r['deadline_success_rate'] >= .9-1e-9 and
                                                       r['reliability_success_rate'] >= 1-1e-9 for r in data)
            summary.append(dict(mode=mode, arrival_rate=rate, **stats, **extra))
            failures += [dict(mode=mode, arrival_rate=rate, eval_seed=r['eval_seed'],
                         deadline_success_rate=r['deadline_success_rate'], reliability_success_rate=r['reliability_success_rate'])
                         for r in data if r['deadline_success_rate'] < .9-1e-9 or r['reliability_success_rate'] < 1-1e-9]
    complete = len(summary) == 20 and all((OUT/m/'completion.json').exists() for m in MODES)
    save(OUT/'summary.json', dict(all_complete=complete, cells=summary, failures=failures))
    if summary:
        with (OUT/'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    lines = ['# SM/Tight 固定相对预算的到达率敏感性', '',
             '50工作流；未筛选种子；总计算能耗J均值±样本SD。失败不删除。',
             '相对DDL固定为0.03/s基准，不按新负载放宽。部分完成不代表最终结论。', '',
             '| 到达率/s | 方法 | 能耗J | ESP@90 | 可靠性达标 | 联合达标 | 任务等待s | 活动利用率 | 最小R裕量 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['arrival_rate']:.2f} | {LABELS[r['mode']]} | {r['energy_mean_J']:.2f} ± {r['energy_sample_sd_J']:.2f} | "
                     f"{r['esp_at_90']:.0%} | {r['reliability_attainment']:.0%} | {r['joint_attainment']:.0%} | "
                     f"{r['avg_task_waiting_time']:.3f} | {r['active_window_utilization']:.2%} | {r['minimum_reliability_margin']:.6f} |")
    lines += ['', f'全部完成：{complete}；完整单元：{len(summary)}/20；真实未达标记录：{len(failures)}。',
              '有限批次、固定训练模型；不能宣称稳态容量、多训练seed或总体100%保证。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if complete:
        plot(summary)
    return dict(all_complete=complete, completed_cells=len(summary), failed_records=len(failures))


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
        points = sorted((r for r in summary if r['mode'] == mode), key=lambda r:r['arrival_rate'])
        ax.errorbar([r['arrival_rate'] for r in points], [r['energy_mean_J'] for r in points],
                    yerr=[r['energy_sample_sd_J'] for r in points], color=colors[mode],
                    linewidth=2, capsize=3, elinewidth=1, label=LABELS[mode])
    ax.set(xlabel='Workflow arrival rate (s$^{-1}$)', ylabel='Total energy (J)', xticks=sorted(RATES))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5), useMathText=True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.25)
    fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    output = OUT/'figures'
    output.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'energy_vs_arrival_rate.{suffix}', dpi=600)
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
        with (OUT/'execution.json').open('x', encoding='utf-8') as handle:
            json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                           concurrent_models=4, automatic_restart=False), handle)
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_mode, mode) for mode in MODES]
            for future in as_completed(futures):
                print(f'[finished] {future.result()}', flush=True)
        print('[arrival-sweep-complete] '+json.dumps(report()), flush=True)


if __name__ == '__main__':
    main()
