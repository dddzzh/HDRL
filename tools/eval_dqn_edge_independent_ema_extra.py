"""Independent-rollout EMA DQN-Edge reruns for Alibaba and SM sensitivity.

Reuse frozen workload fingerprints, original models, generators and rollout code.
Never write manuscript/figures, overwrite old data, or resume partial runs.
"""
import argparse
import copy
import io
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_dqn_edge_independent_ema as ema
from tools import eval_alibaba_sl111 as ali
from tools import eval_sm74_workflow_count as count
from tools import eval_sm74_arrival_rate as rate
from tools.eval_sm74_ablation import read, rows, save, sha, check_sources

OUT = ROOT/'results/dqn_edge_independent_ema_extra_20261031'
GROUPS = ('alibaba', 'workflow_count', 'arrival_rate')
PLANS = dict(alibaba=ali.OUTPUT/'evaluation_plan.json',
             workflow_count=count.OUT/'frozen_plan.json',
             arrival_rate=rate.OUT/'frozen_plan.json')
_plan = _group = _old = None


def cases(group):
    if group == 'alibaba':
        return [dict(setting=d, deadline=d, count=100, rate=.03, seed=s)
                for d in ali.DEADLINES for s in ali.SEEDS]
    if group == 'workflow_count':
        return [dict(setting=f'n{n}', deadline='tight', count=n, rate=.03, seed=s)
                for n in count.COUNTS for s in count.SEEDS]
    if group == 'arrival_rate':
        return [dict(setting=rate.tag(r), deadline='tight', count=50, rate=r, seed=s)
                for r in rate.RATES for s in rate.SEEDS]
    raise ValueError(group)


def scenario(case, group):
    if group == 'alibaba':
        config = read(ali.CONFIG)
        config['simulation']['deadline_type'] = case['deadline']
        return config
    if group == 'workflow_count':
        return count.scenario(case['count'])
    return rate.scenario(case['rate'])


def build_sample(case, group, config, resources, builder, old):
    if group == 'arrival_rate':
        base = copy.deepcopy(config)
        base['generator']['arrival_rate'] = rate.BASE_RATE
        workflows = rate.retime(builder(base, resources, case['seed']), case['rate'])
        actual = rate.sample_identity(workflows)
    else:
        workflows = builder(config, resources, case['seed'])
        actual = ali.identity(workflows)
    sample_key = str(case['count']) if group == 'workflow_count' else case['setting']
    expected = old['samples'][sample_key][str(case['seed'])]
    if len(workflows) != case['count'] or any(actual[k] != expected[k] for k in actual):
        raise ValueError(f'Frozen input/deadline mismatch: {group} {case}')
    return workflows, actual


def prepare():
    if OUT.exists():
        raise FileExistsError('Already prepared; do not duplicate')
    main = read(ema.OUT/'frozen_plan.json')
    check_sources(main)
    sources = dict(main['source_sha256'])
    models = {}
    for group in GROUPS:
        old = read(PLANS[group])
        if group == 'alibaba':
            ali.check_sources(old)
        else:
            count.check_sources(old)
        resource = 'sl_111' if group == 'alibaba' else 'sm_74'
        model = copy.deepcopy(next(m for m in main['models'] if m['resource'] == resource))
        old_model = next(m for m in old['models'] if m['mode'] == 'dqn_edge')
        for field in ('checkpoint_sha256', 'training_config_sha256'):
            assert model[field] == old_model[field], (group, field)
        if group == 'alibaba':
            model['config'] = str(ali.CONFIG.relative_to(ROOT))
            model['config_sha256'] = sha(ali.CONFIG)
        models[group] = model
        for source, digest in old['source_sha256'].items():
            path = ROOT/source
            sources[str(path.relative_to(ROOT))] = digest
        sources[str(PLANS[group].relative_to(ROOT))] = sha(PLANS[group])
        old_root = ali.OUTPUT if group == 'alibaba' else (count.OUT if group == 'workflow_count' else rate.OUT)
        for path in old_root.rglob('comparison_rows.*'):
            if path.suffix in ('.csv', '.jsonl'):
                sources[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), ROOT/'tests/test_dqn_edge_independent_ema_extra.py'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    plan = dict(models=models, source_sha256=sources, cases={g: cases(g) for g in GROUPS},
        workers_per_group=2, maximum_rollout_workers=6, torch_threads=1,
        model_initialization_seed=742000, server_seed=42, training=False,
        ema_protocol='Restore checkpoint-loaded EMA before every rollout; update normally within rollout',
        arrival_deadline_policy='fixed_relative_budget_at_0.03',
        preserve_completed_main_540=True, automatic_restart=False, automatic_partial_resume=False,
        manuscript_updates=False, figure_updates=False)
    check_sources(plan)
    OUT.mkdir()
    save(OUT/'frozen_plan.json', plan)
    (OUT/'README.md').write_text(
        '# Supplementary DQN-Edge independent-EMA reruns\n\n'
        '2026-10-31 user authorized all remaining DQN-Edge experiments without cross-seed EMA.\n'
        'Main experiment 540 rows already complete; not repeated. Alibaba 60, workflow count 100, arrival rate 100.\n'
        'Each rollout starts from the identical saved checkpoint EMA; within-rollout updates remain enabled.\n'
        'Original checkpoints, seeds 742000--742019, workload/deadline fingerprints and physical configs retained.\n'
        'Arrival-rate sweep retains the baseline 0.03/s relative deadline budgets.\n'
        'Each group has two rollout workers; Torch threads 1 per worker, at most six workers overall.\n'
        'Original data are preserved. No automatic restart, partial resume, manuscript replacement or figure update.\n'
        'User is restoring the earlier manuscript; wait for complete verified outputs before a future replacement.\n'
        'Other algorithms have no DQN attention EMA and are not rerun. All failures are retained.\n', encoding='utf-8')
    report()
    print('[extra-prepared] 260 rollouts; no inference started', flush=True)


def initialize(group):
    global _plan, _group, _old
    _plan = read(OUT/'frozen_plan.json')
    _group = group
    _old = read(PLANS[group])
    ema.initialize(_plan['models'][group])


def evaluate_case(case):
    e, _, resources, runtime, initial, weights_sha = ema._runtime
    config = scenario(case, _group)
    with redirect_stdout(io.StringIO()):
        workflows, identity = build_sample(case, _group, config, resources, e.build_workflows, _old)
    ema.reset_ema(runtime['agent'], initial)
    start_sha = ema.ema_sha(runtime['agent'].smoothed_weights)
    captured = {}
    original_env = e.build_env
    def checked_env(*args, **kwargs):
        captured['env'] = original_env(*args, **kwargs)
        return captured['env']
    def checked_build(actual_config, actual_resources, seed, smoke_test=False):
        assert seed == case['seed'] and not smoke_test and actual_config == config
        assert actual_resources is resources
        return workflows
    with patch.object(e, 'build_workflows', checked_build), patch.object(e, 'build_env', checked_env), redirect_stdout(io.StringIO()):
        metrics = e.run_dqn_edge(config, resources, runtime, case['seed'])
    assert ema.network_sha(runtime['agent']) == weights_sha, 'Network changed during inference'
    env = captured['env']
    from src.simulator.metrics import compute_summary
    detail = env._build_workflow_results()
    assert len(detail) == case['count'] and all(w['finish_time'] is not None for w in detail)
    independently = compute_summary(detail, env.event_log, env.servers)
    for field in ('total_energy', 'deadline_success_rate', 'reliability_success_rate'):
        assert math.isclose(independently[field], metrics[field], rel_tol=1e-9, abs_tol=1e-9), field
    row = dict(scale='small', deadline_type=case['deadline'], mode='dqn_edge', eval_seed=case['seed'],
        num_workflows=case['count'], arrival_rate=case['rate'], setting=case['setting'],
        **e.add_normalized_metrics(metrics, case['count']), **{k:v for k,v in identity.items() if k != 'job_ids'},
        selection_is_conditional=False, ema_reset_before_rollout=True, ema_initial_sha256=start_sha,
        ema_final_sha256=ema.ema_sha(runtime['agent'].smoothed_weights), network_sha256=weights_sha)
    if _group == 'alibaba':
        row.update(dataset_source='alibaba_v2018', servers=111)
    elif _group == 'arrival_rate':
        extra, _ = rate.diagnostics(env, metrics)
        row.update(extra, deadline_policy='fixed_relative_budget_at_0.03')
    detail = [{k:v for k,v in w.items() if k != 'task_plans'} for w in detail]
    return row, detail


def validate(data, group, plan, complete=False):
    expected = plan['cases'][group]
    assert len(data) <= len(expected)
    if complete:
        assert len(data) == len(expected)
    model = plan['models'][group]
    for row, case in zip(data, expected):
        assert (row['setting'], row['eval_seed'], row['num_workflows'], row['arrival_rate']) == (
            case['setting'], case['seed'], case['count'], case['rate'])
        assert row['mode'] == 'dqn_edge' and row['ema_reset_before_rollout'] is True
        assert row['ema_initial_sha256'] == model['ema_initial_sha256']
        assert row['network_sha256'] == model['network_sha256']
        assert math.isfinite(row['total_energy']) and row['total_energy'] > 0
        assert all(0 <= row[k] <= 1 for k in ('deadline_success_rate', 'reliability_success_rate'))


def run(group):
    from tools import eval_all_baselines as e
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    directory = OUT/group
    directory.mkdir()  # Refuse duplicate launch, partial overwrite or implicit resume.
    save(directory/'experiment_manifest.json', dict(group=group, model=plan['models'][group],
        workers=2, torch_threads=1, model_initialization_seed=742000, server_seed=42,
        plan_sha256=sha(OUT/'frozen_plan.json'), ema_protocol=plan['ema_protocol'],
        training=False, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()))
    data = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(group,)) as pool:
            for row, detail in pool.map(evaluate_case, plan['cases'][group]):
                data.append(row)
                validate(data, group, plan)
                details = directory/'workflow_details'/row['setting']
                details.mkdir(parents=True, exist_ok=True)
                save(details/f"seed_{row['eval_seed']}.json", dict(eval_seed=row['eval_seed'], workflows=detail))
                e.write_rows(directory, data)
                print(f"[extra-ema] {group} {len(data)}/{len(plan['cases'][group])} {row['setting']} seed={row['eval_seed']} E={row['total_energy']:.2f}J", flush=True)
        validate(data, group, plan, complete=True)
        check_sources(plan)
        failures = [dict(setting=r['setting'], eval_seed=r['eval_seed'],
                        deadline_success_rate=r['deadline_success_rate'], reliability_success_rate=r['reliability_success_rate'])
                    for r in data if r['deadline_success_rate'] < .9-1e-9 or r['reliability_success_rate'] < 1-1e-9]
        save(directory/'completion.json', dict(records=len(data), ema_reset_count=len(data),
             source_hashes_verified=True, failures=failures))
        report()
        print(f'[extra-ema-complete] {group} records={len(data)}', flush=True)
    except Exception:
        save(directory/'failure.json', dict(traceback=traceback.format_exc(), automatic_restart=False))
        raise


def report():
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    lines = ['# Supplementary DQN-Edge independent-EMA progress', '',
             'Main experiment: 540 rows already complete; not repeated. No manuscript/figure updates.', '',
             '| Experiment | Written | Planned | Complete |', '|---|---:|---:|---|']
    total = completed = 0
    for group in GROUPS:
        folder = OUT/group
        data = rows(folder/'comparison_rows.jsonl') if (folder/'comparison_rows.jsonl').exists() else []
        done = (folder/'completion.json').exists()
        validate(data, group, plan, complete=done)
        total += len(data)
        completed += done
        lines.append(f"| {group} | {len(data)} | {len(plan['cases'][group])} | {done} |")
    lines += ['', f'Total {total}/260; completed groups {completed}/3. Partial results are not final.']
    tmp = OUT/f'REPORT.{os.getpid()}.tmp'
    tmp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tmp.replace(OUT/'REPORT.md')
    save(OUT/'summary.json', dict(records=total, expected=260, complete_groups=completed, all_complete=completed == 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--prepare', action='store_true')
    commands.add_argument('--run', choices=GROUPS)
    commands.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run(args.run)
    else:
        report()
