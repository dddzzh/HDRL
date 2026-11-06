"""Frozen SM74 upper-deadline sensitivity; baseline reproduction gates the sweep."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_sm74_ablation import read, rows, sha, save

OUT = ROOT / 'results/sensitivity_sm74_deadline_20261016'
CONFIG = ROOT / 'configs/v3_sm_74_new_eval_50wf.json'
REFERENCE = ROOT / 'results/legacy_train_new_eval_20261009/sm_74'
SAMPLES = ROOT / 'results/ablation_sm74_new_eval_20261015/frozen_plan.json'
PROTOCOL = ROOT / 'docs/SENSITIVITY_SM74_PLAN_20261016.md'
ORIGINAL = 3.558333333333308
POINTS = [('original', ORIGINAL), ('lambda_0', 0.), ('lambda_1p5', 1.5),
          ('lambda_4p5', 4.5), ('lambda_6', 6.)]
DEADLINES = ['medium', 'tight']
SEEDS = list(range(742000, 742020))
CASES = [(d, s) for d in DEADLINES for s in SEEDS]
_runtime = None


def set_multiplier(agent, value):
    controller = agent.high_agent.deadline_lagrangian
    if not math.isfinite(value) or not 0 <= value <= controller.lambda_max:
        raise ValueError('Multiplier outside the frozen training range')
    controller.lambda_deadline = float(value)
    if agent.high_agent.deadline_lambda != value:
        raise ValueError('Requested multiplier is not read by the upper agent')
    if agent.low_agent is None or not agent.high_agent.deadline_head_enabled:
        raise ValueError('SAC and the trained deadline head must remain enabled')


def controller_state(agent):
    return [copy.deepcopy(c.state_dict()) for c in (
        agent.high_agent.deadline_lagrangian,
        agent.high_agent.reliability_lagrangian,
        agent.low_agent.deadline_lagrangian,
        agent.low_agent.reliability_lagrangian)]


def weights_hash(agent):
    upper = agent.high_agent.agent_R
    networks = [getattr(upper, name) for name in (
        'energy_q_eval', 'energy_q_target', 'deadline_q_eval',
        'deadline_q_target', 'reliability_q_eval', 'reliability_q_target')]
    networks += [getattr(agent.low_agent, name) for name in (
        'actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2')]
    digest = hashlib.sha256()
    for network in networks:
        for name, tensor in network.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_sources(plan):
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')


def validate(data, point, value, plan, complete=False):
    keys = [(r['deadline_type'], r['eval_seed']) for r in data]
    if keys != CASES[:len(keys)] or len(keys) > 40 or (complete and len(keys) != 40):
        raise ValueError('Missing, duplicated, unexpected or reordered seed records')
    for row in data:
        if (row['point'] != point or row['lambda_deadline'] != value
                or row['num_workflows'] != 50 or row['scale'] != 'small'
                or row['mode'] != 'd3qn_sac'):
            raise ValueError('Row protocol mismatch')
        sample = plan['samples'][row['deadline_type']][str(row['eval_seed'])]
        if any(row[k] != sample[k] for k in ('physical_sha256', 'deadlines_sha256', 'tasks', 'edges')):
            raise ValueError('Workload/deadline identity mismatch')
        if not math.isfinite(row['total_energy']) or row['total_energy'] <= 0:
            raise ValueError('Invalid energy')
        for key in ('deadline_success_rate', 'reliability_success_rate'):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError('Invalid attainment metric')
        if row['low_buffer_size'] <= 0:
            raise ValueError('Expected an enabled SAC frequency agent')


def compare_reference(row, original):
    if not math.isclose(row['total_energy'], original['total_energy'], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Original-point energy mismatch at {row['deadline_type']}/{row['eval_seed']}")
    for key in ('deadline_success_rate', 'reliability_success_rate'):
        if not math.isclose(row[key], original[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Original-point {key} mismatch')


def summarize(group):
    energy = [r['total_energy'] for r in group]
    esp = statistics.mean(r['deadline_success_rate'] >= .9-1e-9 for r in group)
    reliability = statistics.mean(r['reliability_success_rate'] >= 1-1e-9 for r in group)
    return dict(n=len(group), energy_mean_J=statistics.mean(energy),
                energy_sample_sd_J=statistics.stdev(energy), esp_at_90=esp,
                reliability_attainment=reliability,
                feasible=esp >= .9-1e-9 and reliability == 1.)


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError('Experiment exists; no duplicate preparation or automatic restart')
    provenance = read(REFERENCE / 'checkpoint_provenance.json')
    checkpoint = Path(provenance['source_checkpoint'])
    if (sha(checkpoint) != provenance['checkpoint_sha256']
            or sha(CONFIG) != provenance['config_sha256']):
        raise ValueError('Original checkpoint or test config changed')
    payload = e.torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload['episode'] != 1200 or not payload['use_low_agent']
            or payload['deadline_lagrangian']['deadline_lambda'] != ORIGINAL
            or payload['config']['agent']['high']['lagrangian']['lambda_max'] != 6):
        raise ValueError('Checkpoint identity or saved multiplier differs from protocol')
    config = read(CONFIG)
    if (config['resources']['num_edge'], config['resources']['num_cloud'],
        config['generator']['num_workflows'], config['generator']['arrival_rate']) != (56,18,50,.03):
        raise ValueError('Test setting differs')
    old_plan = read(SAMPLES)
    original = [r for r in rows(REFERENCE / 'comparison_rows.jsonl')
                if r['scale'] == 'small' and r['deadline_type'] in DEADLINES]
    if [(r['deadline_type'], r['eval_seed']) for r in original] != CASES:
        raise ValueError('Reference rows incomplete or reordered')
    paths = [checkpoint, CONFIG, REFERENCE/'checkpoint_provenance.json',
             REFERENCE/'comparison_rows.jsonl', SAMPLES, PROTOCOL, Path(__file__),
             ROOT/'tests/test_sm74_lambda_sensitivity.py']
    paths += [ROOT/name for name in old_plan['source_sha256']
              if name.startswith(('src/', 'tools/')) or name.endswith('.xml')
              or name == 'configs/resources_edge_cloud_dvfs_v3.json']
    paths += [ROOT/'src/device_utils.py']
    plan = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_episode=1200,
                config=str(CONFIG.relative_to(ROOT)), original_lambda=ORIGINAL,
                points=[dict(name=n, value=v) for n,v in POINTS],
                deadlines=DEADLINES, seeds=SEEDS, workers=2, torch_threads=1,
                model_initialization_seed=742000, server_seed=42,
                num_workflows=50, training=False, baseline_gate_records=40,
                samples={d:old_plan['samples'][d] for d in DEADLINES},
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
                reference_rows=original)
    check_sources(plan)
    OUT.mkdir()
    save(OUT/'frozen_plan.json', plan)
    print('[prepared] points=5 total_records=200 baseline_gate=40 training=False', flush=True)


def initialize(point, value):
    global _runtime
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    plan = read(OUT/'frozen_plan.json')
    with redirect_stdout(io.StringIO()):
        _, _, agent, objective = e.load_d3qn(ROOT/plan['checkpoint'], 742000)
        resources = e.build_resources(read(CONFIG), 42)
    set_multiplier(agent, value)
    _runtime = (point, value, plan, agent, objective, resources,
                controller_state(agent), weights_hash(agent))


def evaluate_case(case):
    from tools import eval_all_baselines as e
    from tools.eval_alibaba_sl111 import identity
    point,value,plan,agent,objective,resources,controls,network_hash = _runtime
    deadline,seed = case
    config = e.apply_deadline_type(read(CONFIG), deadline)
    if controller_state(agent) != controls:
        raise ValueError('Controller state leaked between episodes')
    with redirect_stdout(io.StringIO()):
        workflows = e.build_workflows(config, resources, seed, smoke_test=False)
        sample = identity(workflows)
        if sample != plan['samples'][deadline][str(seed)]:
            raise ValueError('Frozen sample identity differs')
        env = e.build_env(workflows, resources, config['simulation'], seed)
        result = e.collect_d3qn_episode(env,agent,objective,e.HighRolloutBuffer(),e.LowRolloutBuffer(),
                 training=False,high_deterministic=True,low_deterministic=True,
                 fixed_low_rho_one=False,high_algorithm='d3qn')
    if controller_state(agent) != controls or weights_hash(agent) != network_hash:
        raise ValueError('Network or controller updated during frozen inference')
    metrics = e.add_normalized_metrics(result['metrics'],50)
    if 'lambda_deadline' in metrics and metrics['lambda_deadline'] != value:
        raise ValueError('Rollout metric reports a different executed multiplier')
    return e.json_ready({**metrics,**sample,'point':point,'lambda_deadline':value,
         'deadline_type':deadline,'eval_seed':seed,'mode':'d3qn_sac','scale':'small',
         'num_workflows':50})


def report():
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    results = []
    baseline_path = OUT/'original/comparison_rows.jsonl'
    baseline = {(r['deadline_type'],r['eval_seed']):r for r in rows(baseline_path)} if baseline_path.exists() else {}
    lines = ['# SM 上层 Deadline 乘子敏感性：原始配对结果', '',
             '冻结 Full，74 台、50 工作流/seed、0.03/s。失败全部保留；不重训、不选新模型。',
             '能耗为 kJ 均值 ± 样本 SD；ESP@90 与可靠性达标为按种子统计的比例。', '',
             '| 乘子 | Deadline | n/20 | 能耗 (kJ) | ESP@90 | 可靠性达标 | 可行 |',
             '|---:|---|---:|---:|---:|---:|---|']
    for point,value in POINTS:
        path = OUT/point/'comparison_rows.jsonl'
        data = rows(path) if path.exists() else []
        validate(data,point,value,plan)
        for deadline in DEADLINES:
            group = [r for r in data if r['deadline_type']==deadline]
            if len(group) != 20:
                lines.append(f'| {value:.15g} | {deadline} | {len(group)}/20 | pending | - | - | - |')
                continue
            stats = summarize(group)
            diffs = [r['total_energy']-baseline[deadline,r['eval_seed']]['total_energy'] for r in group]
            stats.update(point=point,lambda_deadline=value,deadline_type=deadline,
                         paired_energy_delta_mean_J=statistics.mean(diffs),
                         paired_energy_delta_sample_sd_J=statistics.stdev(diffs))
            results.append(stats)
            lines.append(f"| {value:.15g} | {deadline} | 20/20 | {stats['energy_mean_J']/1000:.2f} ± {stats['energy_sample_sd_J']/1000:.2f} | {stats['esp_at_90']:.0%} | {stats['reliability_attainment']:.0%} | {stats['feasible']} |")
    save(OUT/'summary.json', results)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT/'frozen_plan.json')
    check_sources(plan)
    if (OUT/'run_started.json').exists():
        raise FileExistsError('No duplicate launch, restart or partial resume')
    save(OUT/'run_started.json',dict(training=False,records=200,workers=2))
    references = {(r['deadline_type'],r['eval_seed']):r for r in plan['reference_rows']}
    started = time.monotonic()
    try:
        for point,value in POINTS:
            check_sources(plan)
            if point != 'original' and not (OUT/'baseline_verified.json').exists():
                raise ValueError('Baseline reproduction gate not passed')
            directory = OUT/point
            directory.mkdir()
            save(directory/'manifest.json',dict(point=point,lambda_deadline=value,
                 plan_sha256=sha(OUT/'frozen_plan.json'),workers=2,training=False))
            data=[]
            print(f'[point-start] {point} lambda={value:.17g}',flush=True)
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize,initargs=(point,value)) as pool:
                for row in pool.map(evaluate_case,CASES):
                    data.append(row)
                    validate(data,point,value,plan)
                    e.write_rows(directory,data)
                    if point == 'original':
                        compare_reference(row,references[row['deadline_type'],row['eval_seed']])
                    print(f"[eval] {point} {len(data)}/40 {row['deadline_type']} {row['eval_seed']} energy_J={row['total_energy']:.6f} deadline={row['deadline_success_rate']:.4f} reliability={row['reliability_success_rate']:.4f}",flush=True)
            validate(data,point,value,plan,complete=True)
            check_sources(plan)
            save(directory/'completion.json',dict(records=40,weights_and_controllers_unchanged=True,
                                                  source_hashes_verified=True))
            if point == 'original':
                save(OUT/'baseline_verified.json',dict(records=40,all_reference_metrics_matched=True,
                                                       elapsed_seconds=time.monotonic()-started))
            report()
            print(f'[point-complete] {point} records=40 elapsed_s={time.monotonic()-started:.1f}',flush=True)
        save(OUT/'completion.json',dict(records=200,points=5,baseline_verified=True,
                                       elapsed_seconds=time.monotonic()-started))
        print('[sensitivity-complete] records=200 points=5 baseline_verified=True',flush=True)
    except Exception as exc:
        save(OUT/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc(),
                                   elapsed_seconds=time.monotonic()-started,automatic_restart=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    action=parser.parse_args().action
    {'prepare':prepare,'run':run,'report':report}[action]()
