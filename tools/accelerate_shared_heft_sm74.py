"""User-authorized 2-to-12 worker handoff; reuse only complete, verified rows."""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from itertools import zip_longest
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import eval_shared_heft_sm74 as original
from tools.eval_sm74_ablation import read, rows, save, sha, check_sources

OUT = original.OUT
ACCEL = OUT / 'acceleration_12'
WORKERS = 12


def key(row):
    return row['scale'], row['deadline_type'], row['eval_seed']


def ordered_cases():
    return [(s,d,n) for s in original.SCALES for d in original.DDLS for n in original.credit.SEEDS]


def remaining_cases(completed):
    seen = {key(r) for r in completed}
    pending = {s:[c for c in ordered_cases() if c[0]==s and c not in seen] for s in original.SCALES}
    return [c for group in zip_longest(*pending.values()) for c in group if c is not None]


def validate_case(row, case, plan):
    if key(row) != case or row['mode'] != original.MODE or row['deadline_reference'] != original.REFERENCE:
        raise ValueError('Wrong case, model or deadline reference')
    if row['num_workflows'] != 50 or row['arrival_rate'] != .03 or row['selection_is_conditional']:
        raise ValueError('Changed evaluation protocol')
    expected = plan['samples'][case[0]][original.credit.tag(.03,case[1])][str(case[2])]
    if any(row[k] != expected[k] for k in original.credit.SAMPLE_KEYS):
        raise ValueError('Changed workload/deadline fingerprint')
    if not math.isfinite(row['total_energy']) or row['total_energy'] <= 0:
        raise ValueError('Invalid energy')
    for field in ('deadline_success_rate','reliability_success_rate'):
        if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
            raise ValueError('Invalid constraint metric')


def contiguous_data(completed):
    data = []
    for case in ordered_cases():
        if case not in completed:
            break
        data.append(completed[case])
    return data


def progress(completed, initial_count):
    from tools.eval_v3_scenarios import result_cells
    all_data = [completed[c] for c in ordered_cases() if c in completed]
    counts = {s:sum(r['scale']==s for r in all_data) for s in original.SCALES}
    save(ACCEL/'progress.json',dict(records=len(all_data),preserved=initial_count,
         newly_completed=len(all_data)-initial_count,counts=counts,workers=WORKERS))
    lines = ['# Shared HEFT 12路进度', '', '只重排未完成任务，原始已完成记录保留。未满20条的格子不是最终结论。', '',
             '| 场景 | DDL | 能耗 J | ESP@90 | Rel | 状态 |','|---|---|---:|---:|---:|---|']
    for scale,label in zip(original.SCALES,('SM','MM','LM')):
        for ddl in original.DDLS:
            group = [r for r in all_data if r['scale']==scale and r['deadline_type']==ddl]
            lines.append(f'| {label} | {ddl} | {result_cells(group,20)} |')
    lines += ['',f'实际完成：{len(all_data)}/180；含复用{initial_count}条。']
    (ACCEL/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    if (OUT/'completion.json').exists():
        raise FileExistsError('Already complete; no duplicate run')
    if sha(OUT/'deadline_audit.json') != plan['deadline_audit_sha256']:
        raise ValueError('Deadline audit changed')
    data = rows(OUT/'comparison_rows.jsonl')
    original.validate(data,plan)
    for row in data:
        validate_case(row,key(row),plan)
    ACCEL.mkdir()  # Exclusive handoff: no automatic retries or partial restarts.
    snapshot = ACCEL/'original_snapshot'
    snapshot.mkdir()
    for name in ('comparison_rows.jsonl','comparison_rows.csv','execution.json','evaluation.log','REPORT.md'):
        shutil.copy2(OUT/name,snapshot/name)
    initial = copy.deepcopy(data)
    pending = remaining_cases(data)
    if len(data)+len(pending) != 180 or len(set(pending)) != len(pending):
        raise ValueError('Missing or duplicated scheduled case')
    manifest = dict(user_authorized=True,stopped_original_pids=[2652,27844,57876],pid=os.getpid(),
        previous_workers=2,workers=WORKERS,torch_threads=1,preserved_records=len(data),new_cases=pending,
        plan_sha256=sha(OUT/'frozen_plan.json'),original_plan_workers=plan['workers'],
        scheduling_only_change=True,model_initialization_seed=742000,
        snapshot_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        adapter_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),ROOT/'tests/test_accelerate_shared_heft_sm74.py')})
    save(ACCEL/'handoff_manifest.json',manifest)
    save(OUT/'execution.json',dict(status='running',pid=os.getpid(),workers=WORKERS,
         preserved_records=len(data),handoff_manifest='acceleration_12/handoff_manifest.json',plan_sha256=manifest['plan_sha256']))
    completed = {key(r):r for r in data}
    progress(completed,len(initial))
    with (ACCEL/'evaluation.log').open('x',encoding='utf-8',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(f'[heft-12-start] preserved={len(data)} remaining={len(pending)} workers={WORKERS}',flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=multiprocessing.get_context('spawn'),initializer=original.initialize) as pool:
            futures = {pool.submit(original.evaluate_case,case):case for case in pending}
            for future in as_completed(futures):
                case = futures[future]
                row,detail = future.result()
                validate_case(row,case,plan)
                if case in completed:
                    raise ValueError('Refusing to overwrite completed case')
                stem = f'{case[0]}_{case[1]}_{case[2]}'
                save(ACCEL/(stem+'_row.json'),row)
                save(OUT/'workflow_details'/(stem+'.json'),detail)
                completed[case] = row
                prefix = contiguous_data(completed)
                original.validate(prefix,plan)
                if prefix[:len(initial)] != initial:
                    raise ValueError('Preserved results changed')
                e.write_rows(OUT,prefix)
                e.write_rows(ACCEL,[completed[c] for c in ordered_cases() if c in completed])
                progress(completed,len(initial))
                print(f"[heft-12] {len(completed)}/180 {case[0]}/{case[1]} seed={case[2]} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}",flush=True)
        data = contiguous_data(completed)
        original.validate(data,plan,complete=True)
        check_sources(plan)
        for name,digest in manifest['snapshot_sha256'].items():
            if sha(snapshot/name) != digest:
                raise ValueError('Original snapshot changed')
        for name,digest in manifest['adapter_source_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Acceleration adapter changed')
        failed = [{k:r[k] for k in ('scale','deadline_type','eval_seed','deadline_success_rate','reliability_success_rate')}
                  for r in data if r['deadline_success_rate']<.9-1e-9 or r['reliability_success_rate']<1-1e-9]
        save(OUT/'completion.json',dict(records=180,failures=failed,source_hashes_verified=True,
             preserved_records=len(initial),workers=WORKERS,original_snapshot_verified=True))
        save(OUT/'execution.json',dict(status='completed',pid=os.getpid(),workers=WORKERS,preserved_records=len(initial),
             handoff_manifest='acceleration_12/handoff_manifest.json',plan_sha256=manifest['plan_sha256']))
        original.report()
        print(f'[heft-12-complete] records=180 failures={len(failed)}',flush=True)


if __name__ == '__main__':
    run()
