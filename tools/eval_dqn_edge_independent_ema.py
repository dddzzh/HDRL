"""Isolated DQN-Edge reevaluation: restore checkpoint EMA before each rollout.

Reuse the original loaders, scenario builders and episode runner unchanged.
Never resume partial rows or overwrite the historical serial evaluation.
"""
import argparse
import copy
import hashlib
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
import statistics
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.eval_sm74_ablation import read, rows, save, sha, check_sources
from tools.eval_baselines_seed_0_19 import SCALES, DDLS, RESOURCES, validate, failures

OUT=ROOT/'results/dqn_edge_independent_ema_0_19_20261029'
OLD=ROOT/'results/baselines_seed_block_0_19_20261026'
_runtime=None


def ema_sha(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def reset_ema(agent,initial):
    agent.smoothed_weights=initial.copy()
    if ema_sha(agent.smoothed_weights)!=ema_sha(initial):
        raise ValueError('EMA reset mismatch')


def network_sha(agent):
    digest=hashlib.sha256()
    for name in ('time_online','energy_online','attention'):
        for key,value in getattr(agent,name).state_dict().items():
            digest.update((name+key).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initialize(model):
    global _runtime
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    with redirect_stdout(io.StringIO()):
        config=e.load_json(ROOT/model['config'])
        resources=e.build_resources(config,42)
        payload,runtime=e.load_dqn_edge(e.load_json(ROOT/model['training_config']),
                                      ROOT/model['checkpoint'],resources,742000)
    assert int(payload['episode'])==model['episode']
    assert runtime['variant']==model['payload_variant']
    initial=runtime['agent'].smoothed_weights.copy()
    assert initial.shape==(model['servers'],2)
    weights_sha=network_sha(runtime['agent'])
    if 'ema_initial_sha256' in model:
        assert ema_sha(initial)==model['ema_initial_sha256']
        assert weights_sha==model['network_sha256']
    _runtime=(e,config,resources,runtime,initial,weights_sha)


def evaluate_case(case):
    e,config,resources,runtime,initial,weights_sha=_runtime
    scale,deadline,seed=case
    reset_ema(runtime['agent'],initial)
    start_sha=ema_sha(runtime['agent'].smoothed_weights)
    scenario=copy.deepcopy(config)
    scenario.setdefault('generator',{})['dataset_scale']=e.DATASET_SCALE_NAMES[scale]
    scenario=e.apply_deadline_type(scenario,deadline)
    with redirect_stdout(io.StringIO()):
        metrics=e.run_dqn_edge(scenario,resources,runtime,seed)
    assert network_sha(runtime['agent'])==weights_sha,'Inference changed network parameters'
    return dict(scale=scale,deadline_type=deadline,mode='dqn_edge',eval_seed=seed,
                num_workflows=50,**e.add_normalized_metrics(metrics,50),
                ema_initial_sha256=start_sha,ema_final_sha256=ema_sha(runtime['agent'].smoothed_weights),
                ema_reset_before_rollout=True,network_sha256=weights_sha)


def prepare():
    if OUT.exists(): raise FileExistsError('Experiment already prepared; do not duplicate')
    old=read(OLD/'frozen_plan.json'); check_sources(old)
    models=[m for m in old['models'] if m['mode']=='dqn_edge']
    assert [m['resource'] for m in models]==list(RESOURCES)
    sources=dict(old['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    sources['tests/test_dqn_edge_independent_ema.py']=sha(ROOT/'tests/test_dqn_edge_independent_ema.py')
    for model in models:
        for key in ('checkpoint','training_config','config'):
            assert sha(ROOT/model[key])==model[key+'_sha256']
        initialize(model)
        agent=_runtime[3]['agent']
        model.update(ema_initial_sha256=ema_sha(_runtime[4]),network_sha256=_runtime[5],
                     inference_device=str(agent.device),attention_ema_alpha=agent.config.attention_ema_alpha)
    plan=dict(models=models,source_sha256=sources,workflow_seeds=list(range(20)),
              model_initialization_seed=742000,server_seed=42,num_workflows=50,
              arrival_rate=.03,scales=list(SCALES),deadline_types=list(DDLS),
              ema_protocol='restore saved checkpoint EMA before every independent seed rollout; update normally within rollout',
              seed_selection='same post-hoc adopted contiguous block 0-19; no further selection',
              training=False,production_code_changes=False,workers_per_resource=2,
              maximum_rollout_workers=6,automatic_resume=False,old_results_untouched=True)
    OUT.mkdir()
    save(OUT/'frozen_plan.json',plan)
    print('Prepared exactly 3 models / 540 rollouts; checkpoint EMA reset per seed.',flush=True)


def run(resource):
    from tools import eval_all_baselines as e
    plan=read(OUT/'frozen_plan.json'); check_sources(plan)
    model=next(m for m in plan['models'] if m['resource']==resource)
    directory=OUT/resource
    directory.mkdir()  # Atomic duplicate/partial-run refusal.
    manifest=dict(model=model,protocol=plan['ema_protocol'],workers=2,torch_threads=1,
                  model_initialization_seed=742000,server_seed=42,workflow_seeds=list(range(20)),
                  plan_sha256=sha(OUT/'frozen_plan.json'),pid=os.getpid(),training=False)
    save(directory/'experiment_manifest.json',manifest)
    cases=[(s,d,n) for s in SCALES for d in DDLS for n in range(20)]
    data=[]
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(model,)) as pool:
            for row in pool.map(evaluate_case,cases):
                data.append(row)
                validate(data,model)
                if any(r['ema_initial_sha256']!=data[0]['ema_initial_sha256'] for r in data):
                    raise ValueError('Seeds did not start from the same checkpoint EMA')
                e.write_rows(directory,data)
                print(f"[independent-ema] {resource} {len(data)}/180 {row['scale']}/{row['deadline_type']} seed={row['eval_seed']} E={row['total_energy']:.2f}J DSR={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}",flush=True)
        validate(data,model,complete=True); check_sources(plan)
        e.write_summary(directory,e.summarize_rows(data),{})
        save(directory/'completion.json',dict(records=180,failures=failures(data),
             source_hashes_verified=True,ema_reset_count=180,ema_initial_sha256=data[0]['ema_initial_sha256']))
        report()
        print(f'[independent-ema-complete] {resource} records=180',flush=True)
    except Exception:
        save(directory/'failure.json',dict(traceback=traceback.format_exc(),automatic_restart=False))
        raise


def report():
    plan=read(OUT/'frozen_plan.json'); check_sources(plan)
    cells=[]; total=0; completed=0
    text=['# DQN-Edge逐种子恢复检查点EMA：独立重评估','',
          '原模型、0—19种子、50wf、0.03/s、资源seed42、FCFS期限不变。仅改变EMA重置边界；每个种子从相同检查点EMA开始，种子内部正常更新。不重训，不删除失败，不覆盖旧结果。',
          '本组种子仍是此前按结果采用的条件样本；独立重置不消除种子选择带来的局限。','',
          '| 资源 | 规模 | DDL | n | 能耗 J均值±样本SD | ESP@90 | Rel |','|---|---|---|---:|---:|---:|---:|']
    for model in plan['models']:
        folder=OUT/model['resource']; path=folder/'comparison_rows.jsonl'
        data=rows(path) if path.exists() else []
        done=(folder/'completion.json').exists(); validate(data,model,complete=done)
        total+=len(data); completed+=done
        for s in SCALES:
            for d in DDLS:
                group=[r for r in data if (r['scale'],r['deadline_type'])==(s,d)]
                if not group: continue
                energy=[r['total_energy'] for r in group]
                c=dict(resource=model['resource'],scale=s,deadline_type=d,n=len(group),
                       energy_mean_J=statistics.mean(energy),energy_sample_sd_J=statistics.stdev(energy) if len(energy)>1 else None,
                       esp_at_90=statistics.mean(r['deadline_success_rate']>=.9-1e-9 for r in group),
                       reliability_attainment=statistics.mean(r['reliability_success_rate']>=1-1e-9 for r in group))
                cells.append(c); sd=f"{c['energy_sample_sd_J']:.2f}" if c['energy_sample_sd_J'] is not None else '—'
                text.append(f"| {model['resource']} | {s} | {d} | {len(group)} | {c['energy_mean_J']:.2f} ± {sd} | {c['esp_at_90']:.0%} | {c['reliability_attainment']:.0%} |")
    text+=['',f'已写入{total}/540，完成{completed}/3模型。不完整数据不是最终结果。']
    save(OUT/'summary.json',dict(records=total,complete_models=completed,all_complete=completed==3,cells=cells))
    # Parent resource jobs can finish together: replace the report atomically.
    tmp=OUT/f'REPORT.{os.getpid()}.tmp'; tmp.write_text('\n'.join(text)+'\n',encoding='utf-8'); tmp.replace(OUT/'REPORT.md')
    print(f'[independent-ema-report] {total}/540 models={completed}/3',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true'); g.add_argument('--run',choices=RESOURCES); g.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.run: run(args.run)
    else: report()
