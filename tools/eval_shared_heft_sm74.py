"""Opt-in shared-HEFT deadline evaluation; all original FCFS files stay untouched."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import io
import multiprocessing
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import eval_deadline_credit_ablation as credit
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources, rank_rounds
from tools.eval_v3_scenarios import result_cells
from src.datasets.shared_heft_deadlines import assign_shared_heft_deadlines

RUN = ROOT / 'results/deadline_reference_sm74_budget001/shared_heft/seed42_20261024_075301_286790_20940'
OUT = ROOT / 'results/shared_heft_eval_sm74_20261024'
RAW = ROOT / 'results/raw_sm74_generalization_20261024'
MODE = 'full_budget001_shared_heft'
REFERENCE = 'shared_environment_online_heft_rho1'
EXPECTED_SHA = 'c3167e8cf93e0ef075682f7c51c768ecdc1d6ebb8dfbe3a145397938ba2d2e04'
SCALES = {'small': 'small', 'medium': 'middle', 'large': 'large'}
DDLS = ('loose', 'medium', 'tight')
FCFS_BUILD = credit.build_sample
_base_runtime = None


def select_model():
    from tools import eval_all_baselines as e
    from tools.train_deadline_reference_experiment import validate_config
    manifest = read(RUN / 'deadline_reference_manifest.json')
    if (manifest['status'], manifest['training_reference'], manifest['validation_feasible']) != ('completed', 'shared_heft', True):
        raise ValueError('Wrong or incomplete training run')
    check_sources(manifest)
    config = read(RUN / 'resolved_config.json')
    if config != manifest['config']:
        raise ValueError('Training manifest/config mismatch')
    clean = copy.deepcopy(config)
    clean['training'].pop('run_name')
    validate_config(clean)
    if [r['episode'] for r in rows(RUN / 'train_history.jsonl')] != list(range(1, 1201)):
        raise ValueError('Incomplete training history')
    inputs = rows(RUN / 'training_inputs.jsonl')
    if [r['seed'] for r in inputs] != list(range(43, 1243)) or any(r['training_reference'] != 'shared_heft' for r in inputs):
        raise ValueError('Wrong training input reference')
    candidates = rank_rounds([RUN / 'eval_history.jsonl'])
    if [c['global_episode'] for c in candidates] != list(range(20, 1201, 20)):
        raise ValueError('Incomplete validation history')
    best = max(candidates, key=lambda c: c['rank'])
    checkpoint = RUN / 'best_d3qn_checkpoint.pt'
    payload = e.torch.load(checkpoint, map_location='cpu', weights_only=False)
    if best['rank'][0] != 1 or best['local_episode'] != 1200 or sha(checkpoint) != EXPECTED_SHA:
        raise ValueError('Unexpected validation-selected checkpoint')
    if payload['episode'] != 1200 or payload['config'] != config or payload['checkpoint_format'] != 'hierarchical_reliability_aware_d3qn_v1':
        raise ValueError('Wrong checkpoint payload')
    return dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=EXPECTED_SHA,
                episode=1200, training_budget=.01, training_reference='shared_heft',
                validation_reference=manifest['validation_reference'], selected_validation=best,
                validation_candidates=candidates)


def build_reference(config, resources, rate, deadline, seed):
    if rate != .03:
        raise ValueError('Only the frozen .03/s grid is authorized')
    original, fcfs_sample = FCFS_BUILD(config, resources, rate, deadline, seed)
    simulation = dict(config['simulation'], deadline_type=deadline)
    with redirect_stdout(io.StringIO()):
        updated, audit = assign_shared_heft_deadlines(original, resources, simulation, seed)
    sample = credit.sample_identity(updated)
    for key in ('physical_sha256', 'workload_sha256', 'tasks', 'edges'):
        if sample[key] != fcfs_sample[key]:
            raise ValueError('Reference replacement changed physical inputs')
    for old, new, record in zip(original, updated, audit):
        if old.deadline_multiplier != new.deadline_multiplier or new.deadline_reference != REFERENCE:
            raise ValueError('Changed deadline multiplier or wrong reference')
        if new.relative_deadline != new.deadline_multiplier * record['shared_reference'] or new.absolute_deadline != new.arrival_time + new.relative_deadline:
            raise ValueError('Incorrect HEFT deadline formula')
        # Existing helper labels its input isolated; here that input is FCFS.
        record['fcfs_reference'] = record.pop('isolated_reference')
        record['fcfs_relative_deadline'] = record.pop('isolated_relative_deadline')
    return updated, sample, dict(fcfs_sample=fcfs_sample, workflow_deadlines=audit)


def build_sample(*args):
    workflows, sample, _ = build_reference(*args)
    return workflows, sample


def validate(data, plan, complete=False):
    expected = [(s, d, n) for s in SCALES for d in DDLS for n in credit.SEEDS]
    actual = [(r['scale'], r['deadline_type'], r['eval_seed']) for r in data]
    if len(data) > 180 or actual != expected[:len(data)] or (complete and len(data) != 180):
        raise ValueError('Missing, duplicate or reordered cases')
    if any(r['deadline_reference'] != REFERENCE for r in data):
        raise ValueError('Not the shared HEFT protocol')
    for scale in SCALES:
        group = [dict(r, scale='small') for r in data if r['scale'] == scale]
        credit.validate(group, MODE, dict(samples=plan['samples'][scale]))


def prepare_case(case):
    from tools import eval_all_baselines as e
    scale, ddl, seed = case
    config = read(credit.CONFIG)
    config['generator']['dataset_scale'] = SCALES[scale]
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
    if len(resources['servers']) != 74:
        raise ValueError('Wrong resource scale')
    _, sample, audit = build_reference(config, resources, .03, ddl, seed)
    return case, sample, audit


def prepare():
    if OUT.exists():
        raise FileExistsError('Already prepared; no overwrite')
    model = select_model()
    raw_plan = read(RAW / 'frozen_plan.json')
    check_sources(raw_plan)
    samples = {s: {credit.tag(.03,d): {} for d in DDLS} for s in SCALES}
    audits = copy.deepcopy(samples)
    cases = [(s,d,n) for s in SCALES for d in DDLS for n in credit.SEEDS]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        for index, (case, sample, audit) in enumerate(pool.map(prepare_case, cases), 1):
            scale, ddl, seed = case
            tag = credit.tag(.03, ddl)
            if audit['fcfs_sample'] != raw_plan['samples'][scale][tag][str(seed)]:
                raise ValueError('Base inputs differ from prior FCFS grid')
            samples[scale][tag][str(seed)] = sample
            audits[scale][tag][str(seed)] = audit
            print(f'[heft-inputs] {index}/180 {scale}/{ddl} seed={seed} physical_inputs_preserved=True', flush=True)
    sources = dict(raw_plan['source_sha256'])
    sources.update(read(RUN / 'deadline_reference_manifest.json')['source_sha256'])
    files = [Path(__file__), ROOT / 'tests/test_shared_heft_evaluation.py', ROOT / model['checkpoint'], RAW / 'frozen_plan.json']
    files += [RUN / name for name in ('resolved_config.json', 'deadline_reference_manifest.json', 'train_history.jsonl', 'eval_history.jsonl', 'training_inputs.jsonl')]
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    plan = dict(models={MODE:model}, config=credit.CONFIG.relative_to(ROOT).as_posix(), samples=samples,
                source_sha256=sources, workers=2, torch_threads=1, model_initialization_seed=742000,
                server_seed=42, seeds=credit.SEEDS, num_workflows=50, servers=74, training=False,
                selection_is_conditional=False, deadline_reference=REFERENCE,
                limitation='Exploratory changed-deadline protocol, not directly comparable to FCFS results; one training seed; validation still used legacy isolated HEFT.')
    OUT.mkdir()
    save(OUT / 'deadline_audit.json', audits)
    plan['deadline_audit_sha256'] = sha(OUT / 'deadline_audit.json')
    save(OUT / 'frozen_plan.json', plan)
    report()
    print('[heft-prepared] checkpoint=1200 target=180', flush=True)


def initialize():
    global _base_runtime
    credit.OUT = OUT
    credit.initialize(MODE)
    _base_runtime = credit._runtime


def evaluate_case(case):
    scale, ddl, seed = case
    variant, plan, config, resources, agent, objective = _base_runtime
    local = copy.deepcopy(config)
    local['generator']['dataset_scale'] = SCALES[scale]
    credit._runtime = (variant, dict(plan, samples=plan['samples'][scale]), local, resources, agent, objective)
    # Process-local and restored after each call; no production module/file changes.
    with patch.object(credit, 'build_sample', build_sample):
        row, detail = credit.evaluate_case((.03, ddl, seed))
    row.update(scale=scale, deadline_reference=REFERENCE)
    return row, detail


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT / 'frozen_plan.json')
    check_sources(plan)
    if sha(OUT / 'deadline_audit.json') != plan['deadline_audit_sha256']:
        raise ValueError('Deadline audit changed')
    directory = OUT / 'workflow_details'
    directory.mkdir()  # Atomic no-duplicate/no-partial-resume guard.
    save(OUT / 'execution.json', dict(status='running', pid=os.getpid(), workers=2, plan_sha256=sha(OUT/'frozen_plan.json')))
    data = []
    with (OUT / 'evaluation.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        cases = [(s,d,n) for s in SCALES for d in DDLS for n in credit.SEEDS]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
            for row, detail in pool.map(evaluate_case, cases):
                data.append(row)
                validate(data, plan)
                save(directory / f"{row['scale']}_{row['deadline_type']}_{row['eval_seed']}.json", detail)
                e.write_rows(OUT, data)
                print(f"[heft-eval] {len(data)}/180 {row['scale']}/{row['deadline_type']} seed={row['eval_seed']} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
        validate(data, plan, complete=True)
        check_sources(plan)
        failures = [{k:r[k] for k in ('scale','deadline_type','eval_seed','deadline_success_rate','reliability_success_rate')}
                    for r in data if r['deadline_success_rate'] < .9-1e-9 or r['reliability_success_rate'] < 1-1e-9]
        save(OUT / 'completion.json', dict(records=180, failures=failures, source_hashes_verified=True))
        save(OUT / 'execution.json', dict(status='completed', pid=os.getpid(), workers=2, plan_sha256=sha(OUT/'frozen_plan.json')))
        report()
        print(f'[heft-complete] records=180 failures={len(failures)}', flush=True)


def report():
    plan = read(OUT / 'frozen_plan.json')
    check_sources(plan)
    path = OUT / 'comparison_rows.jsonl'
    data = rows(path) if path.exists() else []
    complete = (OUT / 'completion.json').exists()
    validate(data, plan, complete)
    lines = ['# Shared-HEFT训练模型：Shared-HEFT期限评估', '',
             '固定第1200轮验证最优模型，budget=0.01；74台、50工作流、0.03/s、连续未筛选742000—742019。',
             '保持FCFS测试物理输入与原2x/3x系数，仅替换期限参考为共享环境online HEFT rho=1。实际执行为学习策略。',
             '原FCFS代码、配置、数据均保留；本结果不能直接当作原FCFS实验的改善。训练模型的验证仍是legacy isolated HEFT。',
             'ESP@90：每批至少45/50按时；Rel：每批所有工作流可靠性≥0.9。能耗J均值±样本SD。', '',
             '| 场景 | DDL | 能耗 J | ESP@90 | Rel | 状态 |', '|---|---|---:|---:|---:|---|']
    for scale, label in zip(SCALES, ('SM','MM','LM')):
        for ddl in DDLS:
            group = [r for r in data if r['scale']==scale and r['deadline_type']==ddl]
            lines.append(f'| {label} | {ddl} | {result_cells(group,20)} |')
    lines += ['', f'完成：{complete}；{len(data)}/180。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def start():
    # Exclusive launch log also guards the preparation stage against duplicates.
    if OUT.exists():
        raise FileExistsError('Experiment already prepared or started')
    with OUT.with_name(OUT.name + '_launch.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(f'[heft-start] pid={os.getpid()} prepare_then_evaluate=True workers=2', flush=True)
        prepare()
        run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','report','start'))
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'report':report, 'start':start}[args.action]()
