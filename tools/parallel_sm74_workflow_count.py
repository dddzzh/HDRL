"""Planned count-level handoff; preserve current serial rollouts in memory."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_sm74_workflow_count as base

OUT = base.OUT
CONTROL = OUT/'parallel_handoff'
# Exact creation identities observed before the user-authorized concurrency change.
TARGETS = {30928:'2026-10-17T02:12:30.559324+00:00',
           45428:'2026-10-17T02:12:30.563014+00:00',
           41204:'2026-10-17T02:12:30.566639+00:00',
           31524:'2026-10-17T02:12:30.569803+00:00'}


def process_handle(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x0800 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    values = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = values[0].dwLowDateTime + (values[0].dwHighDateTime << 32)
    expected = datetime.fromisoformat(TARGETS[pid]).timestamp()
    if abs((ticks/10_000_000-11644473600)-expected) > .0001:
        raise ValueError('PID creation identity changed; refuse process operation')
    return handle


def set_suspended(pid, suspended):
    handle = process_handle(pid)
    native = ctypes.WinDLL('ntdll')
    operation = native.NtSuspendProcess if suspended else native.NtResumeProcess
    operation.argtypes, operation.restype = [wintypes.HANDLE], ctypes.c_long
    try:
        result = operation(handle)
        if result != 0:
            raise OSError(f'Process operation refused, NTSTATUS={result}')
    finally:
        close = ctypes.WinDLL('kernel32').CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close(handle)


def future_cells(current):
    return [(mode, n) for n in base.COUNTS for mode in base.MODES if n > current[mode]]


def evaluate_cell(cell):
    mode, count = cell
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    from tools.eval_alibaba_sl111 import identity
    configure_torch_runtime(1)
    plan = base.read(OUT/'frozen_plan.json')
    base.check_sources(plan)
    target = OUT/mode/f'n{count}'
    target.mkdir()  # Count ownership: never duplicate a current or completed cell.
    base.save(target/'parallel_owner.json', dict(pid=os.getpid(), count=count, mode=mode,
              serial_seed_order=base.SEEDS, workers=1, source_plan_sha256=base.sha(OUT/'frozen_plan.json')))
    with (CONTROL/f'{mode}_n{count}.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        model = next(r for r in plan['models'] if r['mode'] == mode)
        prefix = dict(d3qn_sac='d3qn', dqn_edge='dqn-edge', dream_dqn='dream', oo_actor_critic='oo')[mode]
        argv = ['eval_all_baselines', '--config', str(OUT/f'config_n{count}.json'),
                f'--{prefix}-checkpoint', model['checkpoint'], '--modes', mode, '--scales', 'small',
                '--deadline-types', 'tight', '--seed-start', '742000', '--num-seeds', '20',
                '--workers', '1', '--seed-schedule', str(OUT/'seed_schedule.json'), '--output-dir', str(target)]
        if mode != 'd3qn_sac':
            argv += [f'--{prefix}-config', model['training_config']]
        with patch.object(sys, 'argv', argv):
            args = e.parse_args()
        original_build, original_write = e.build_workflows, e.write_rows
        generated = {}

        def checked_build(config, resources, seed, smoke_test=False):
            workflows = original_build(config, resources, seed, smoke_test=smoke_test)
            sample = identity(workflows)
            expected = plan['samples'][str(count)][str(seed)]
            if smoke_test or any(sample[k] != expected[k] for k in sample):
                raise ValueError('Paired physical/DDL identity changed')
            generated[seed] = expected
            return workflows

        def checked_write(directory, data):
            for r in data:
                sample = generated[r['eval_seed']]
                r.update({k:sample[k] for k in ('physical_sha256', 'deadlines_sha256', 'tasks', 'edges',
                         'arrival_first', 'arrival_last', 'relative_deadline_mean', 'reference_response_mean')})
                r['selection_is_conditional'] = False
            base.validate(data, mode, count, plan)
            original_write(directory, data)

        with patch.object(e, 'build_workflows', checked_build), patch.object(e, 'write_rows', checked_write):
            _, data, _ = e.evaluate(args)
        base.validate(data, mode, count, plan, complete=True)
        manifest = base.read(target/'experiment_manifest.json')
        if (manifest['workers'], manifest['model_initialization_seed'], manifest['server_seed'],
            manifest['evaluation_seed_schedule']) != (1,742000,42,{'small':{'tight':base.SEEDS}}):
            raise ValueError('State/order/worker protocol changed')
        base.check_sources(plan)
        base.save(target/'completion.json', dict(records=20, **base.summarize(data)))
        print(f'[parallel-count-complete] {mode} n={count} records=20', flush=True)
    return cell


def main():
    CONTROL.mkdir()  # Exclusive one-time handoff, never automatic restart/resume.
    plan = base.read(OUT/'frozen_plan.json')
    base.check_sources(plan)
    base.save(CONTROL/'source_identity.json', dict(source_plan_sha256=base.sha(OUT/'frozen_plan.json'),
              runner_sha256=base.sha(Path(__file__)), tests_sha256=base.sha(ROOT/'tests/test_count_parallel_handoff.py'),
              targets=TARGETS, cpu_logical=20, max_new_workers=12, max_total_evaluation_workers=16,
              policy='current count stays live; only unstarted counts are parallelized'))
    suspended = []
    current = {}
    try:
        for mode in base.MODES:
            pid = base.read(OUT/mode/'execution.json')['pid']
            if pid not in TARGETS:
                raise ValueError('Unexpected current experiment worker')
            set_suspended(pid, True)
            suspended.append(pid)
            active = [n for n in base.COUNTS if (OUT/mode/f'n{n}').exists()]
            current[mode] = max(active)
            source = OUT/mode/f'n{current[mode]}'/'comparison_rows.jsonl'
            base.save(CONTROL/f'{mode}_retained_rows.json', base.rows(source) if source.exists() else [])
        jobs = future_cells(current)
        if not jobs:
            raise ValueError('No unstarted counts available to parallelize')
        base.save(CONTROL/'handoff.json', dict(current_counts=current, suspended=suspended,
                  jobs=jobs, status='new_counts_starting', no_model_or_seed_change=True))
        print(f'[handoff] current={current}; new_cells={len(jobs)}; max_new_workers=12', flush=True)
        with ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context('spawn')) as pool:
            submitted = {pool.submit(evaluate_cell, cell):cell for cell in jobs}
            next_counts = {m:min(n for mm,n in jobs if mm==m) for m in base.MODES if any(mm==m for mm,_ in jobs)}
            # Wait for real first rows: the original evaluator's existing explicit-seed
            # duplicate guard then safely prevents it from entering the handed-off count.
            try:
                while not all((OUT/m/f'n{n}'/'comparison_rows.jsonl').exists() for m,n in next_counts.items()):
                    for future in submitted:
                        if future.done():
                            future.result()
                    time.sleep(2)
            except BaseException as error:
                base.save(CONTROL/'failure.json', dict(error=repr(error), stage='first_row_reservation',
                          automatic_restart=False))
                for pid in list(suspended):
                    set_suspended(pid, False)
                    suspended.remove(pid)
                print('[handoff-error] original processes resumed; no automatic restart: '+repr(error), flush=True)
                raise
            for pid in list(suspended):
                set_suspended(pid, False)
                suspended.remove(pid)
            base.save(CONTROL/'handoff.json', dict(current_counts=current, jobs=jobs, status='original_counts_resumed',
                      resumed_once=list(TARGETS), no_model_or_seed_change=True,
                      expected_old_guard='Explicit-seed evaluation must start fresh; sequential policy state is not restored'))
            print('[handoff] all four original in-memory counts resumed; no row repeated', flush=True)
            for future, cell in submitted.items():
                print(f'[finished] {future.result()}', flush=True)
        # Original processes finish their retained count, then encounter the duplicate
        # guard at the next count. That expected guard is a planned handoff, not a crash.
        while not all((OUT/m/f'n{n}'/'completion.json').exists() for m,n in current.items()):
            time.sleep(5)
        for mode in base.MODES:
            for count in base.COUNTS:
                target = OUT/mode/f'n{count}'
                base.validate(base.rows(target/'comparison_rows.jsonl'), mode, count, plan, complete=True)
                if not (target/'completion.json').exists():
                    raise ValueError('Incomplete count during final handoff verification')
            retained = base.read(CONTROL/f'{mode}_retained_rows.json')
            final = base.rows(OUT/mode/f'n{current[mode]}'/'comparison_rows.jsonl')
            if final[:len(retained)] != retained:
                raise ValueError('Pre-switch records changed')
            base.save(OUT/mode/'completion.json', dict(records=100, all_counts=base.COUNTS, source_hashes_verified=True))
        print('[sweep-complete] '+json.dumps(base.report()), flush=True)
    finally:
        for pid in list(suspended):
            set_suspended(pid, False)
            suspended.remove(pid)


if __name__ == '__main__':
    main()
