"""Frozen SM74 ablations; original Full reference is reused, never rerun."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import os
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"results/ablation_sm74_new_eval_20261015"
BASE=ROOT/"results/ablations/sm_74_legacy_train_20wf"
FULL=ROOT/"results/legacy_train_new_eval_20261009/sm_74"
CONFIG=ROOT/"configs/v3_sm_74_new_eval_50wf.json"
DEADLINES=["loose","medium","tight"]
SEEDS=list(range(742000,742020))
VARIANTS={"heft_sac":("HEFT+SAC","heft",False),
          "heft_rho1":("HEFT, rho=1","heft",True),
          "full_rho1":("Full checkpoint, rho=1 (diagnostic)","d3qn",True),
          "no_deadline_rho1":("No Deadline Head, rho=1 (final, validation infeasible)","d3qn",True)}
_runtime=None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    temp=path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def rank_rounds(history_paths):
    from tools.hierarchical_common import combine_checkpoint_metrics,metric_rank
    candidates=[]
    for path in history_paths:
        groups=defaultdict(list)
        for row in rows(path):
            groups[row["episode"]].append(row)
        for local,block in groups.items():
            if len(block)!=2 or {r["eval_seed"] for r in block}!={542000,642000}:
                raise ValueError(f"Incomplete or duplicate validation blocks: {path}/{local}")
            if any(r["eval_episodes"]!=20 for r in block):
                raise ValueError("Validation must use 20 seeds per block")
            combined=combine_checkpoint_metrics(block)
            rank=metric_rank(combined,reliability_target=1,deadline_esp_threshold=.9,deadline_esp_target=1)
            candidates.append(dict(history=str(path.relative_to(ROOT)),local_episode=local,
                  global_episode=block[0]["global_episode"],energy=combined["total_energy"],rank=list(rank),
                  min_block_esp=min(r["deadline_esp_0p90"] for r in block),
                  min_block_reliability=min(r["reliability_esp_1p00"] for r in block)))
    return candidates


def prepare():
    from tools import eval_all_baselines as e
    from tools.eval_alibaba_sl111 import identity
    if OUT.exists():
        raise ValueError("Existing ablation experiment: do not prepare twice")
    heft_hist=sorted(p for name in ["heft_sac","heft_sac_resume_ep0854","heft_sac_resume_ep0860"]
                     for p in (BASE/name).glob("*/eval_history.jsonl"))
    candidates=rank_rounds(heft_hist)
    best=max(candidates,key=lambda r:r["rank"])
    if best["rank"][0]!=1 or best["global_episode"]!=240:
        raise ValueError("Validation audit differs from the declared model selection")
    heft=ROOT/Path(best["history"]).parent/"best_heft_checkpoint.pt"
    head_dir=BASE/"high_no_deadline_rho1/20261014_155123"
    head_hist=head_dir/"eval_history.jsonl"
    head_ranks=rank_rounds([head_hist])
    if len(candidates)!=59 or len(head_ranks)!=60 or any(r["rank"][0]==1 for r in head_ranks):
        raise ValueError("Training/validation completion differs from declared protocol")
    original=read(FULL/"checkpoint_provenance.json")
    full=Path(original["source_checkpoint"])
    if sha(full)!=original["checkpoint_sha256"] or sha(CONFIG)!=original["config_sha256"]:
        raise ValueError("Original Full sources changed")
    files={}
    for path in [full,heft,head_dir/"final_checkpoint.pt",CONFIG,FULL/"checkpoint_provenance.json",
                 FULL/"comparison_rows.jsonl",FULL/"comparison_rows.csv",head_hist,*heft_hist]:
        files[str(path.relative_to(ROOT))]=sha(path)
    for name in ["tools/eval_sm74_ablation.py","docs/ABLATION_SM74_EVALUATION_20261015.md",
                 "tools/eval_all_baselines.py","tools/hierarchical_common.py","tools/eval_alibaba_sl111.py",
                 "tools/train_hierarchical_d3qn.py","tools/train_reliability_aware_hierarchical_d3qn.py",
                 "src/D3QN/reliability_aware_high.py","src/D3QN/dual_head_constrained.py",
                 "src/D3QN/hierarchical_d3qn_agent.py","src/sac/hierarchical_low_sac.py",
                 "src/simulator/edge_cloud_env.py","src/simulator/resource.py","src/simulator/reliability.py",
                 "src/simulator/metrics.py","src/sac/reward.py","src/shared/hierarchical_state_reward.py",
                 "src/datasets/sec_job_workflow_generator.py","src/datasets/pegasus_workflow_generator.py",
                 "configs/resources_edge_cloud_dvfs_v3.json"]:
        files[name]=sha(ROOT/name)
    for path in (ROOT/"data/sec_job_reference").glob("*.xml"):
        files[str(path.relative_to(ROOT))]=sha(path)
    definitions={}
    for name,(label,algorithm,fixed) in VARIANTS.items():
        path=heft if name.startswith("heft") else full if name=="full_rho1" else head_dir/"final_checkpoint.pt"
        payload=e.torch.load(path,map_location="cpu",weights_only=False)
        if payload["episode"]!=(240 if name.startswith("heft") else 1200):
            raise ValueError("Checkpoint episode mismatch")
        if payload["config"]["resources"]!={**read(CONFIG)["resources"]}:
            raise ValueError("Training and test resource definitions differ")
        if name=="no_deadline_rho1" and (payload["use_low_agent"] or payload["config"]["agent"]["high"]["deadline_head_enabled"]):
            raise ValueError("No-head checkpoint flags differ")
        definitions[name]=dict(label=label,algorithm=algorithm,fixed_rho_one=fixed,
                  checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),episode=payload["episode"],
                  loader_config=payload["config"],validation_feasible=name!="no_deadline_rho1")
    config=read(CONFIG)
    with redirect_stdout(io.StringIO()):
        resources=e.build_resources(config,42)
        samples={d:{str(n):identity(e.build_workflows(e.apply_deadline_type(config,d),resources,n,smoke_test=False))
                     for n in SEEDS} for d in DEADLINES}
    reference=[r for r in rows(FULL/"comparison_rows.jsonl") if r["scale"]=="small"]
    validate_rows(reference,"d3qn_sac",complete=True)
    OUT.mkdir()
    plan=dict(config=str(CONFIG.relative_to(ROOT)),num_workflows=50,servers=74,workers_per_variant=2,
              model_initialization_seed=742000,server_seed=42,scale="small",deadlines=DEADLINES,seeds=SEEDS,
              variants=definitions,samples=samples,source_sha256=files,original_full=str((FULL/"comparison_rows.jsonl").relative_to(ROOT)),
              no_head_protocol="fixed_final_1200_diagnostic_no_feasible_validation",selection=best,
              missing_validation_global_episodes=[860],causal_limit="No independently trained full-high rho1 in SM74")
    save(OUT/"frozen_plan.json",plan)
    save(OUT/"validation_selection.json",dict(heft_candidates=candidates,selected=best,no_head_candidates=head_ranks))
    print("[ablation-prepared] variants=4 new_rows=240 original_full_reused=60",flush=True)
    report()


def check_sources(plan):
    for name,expected in plan["source_sha256"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError(f"Frozen source changed: {name}")


def validate_rows(data,variant,complete=False,plan=None):
    expected=[(d,n) for d in DEADLINES for n in SEEDS]
    actual=[(r["deadline_type"],r["eval_seed"]) for r in data]
    if actual!=expected[:len(actual)] or len(actual)>60 or (complete and len(actual)!=60):
        raise ValueError("Ablation rows missing, duplicated or out of order")
    for row in data:
        if row["mode"]!=variant or row["scale"]!="small" or row["num_workflows"]!=50:
            raise ValueError("Ablation row protocol differs")
        if not math.isfinite(row["total_energy"]) or row["total_energy"]<=0:
            raise ValueError("Invalid energy")
        if any(not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ["deadline_success_rate","reliability_success_rate"]):
            raise ValueError("Invalid attainment rate")
        if variant in VARIANTS and VARIANTS[variant][2]:
            if not math.isclose(row["rho_mean"],1.,abs_tol=1e-12) or row["low_buffer_size"]!=0:
                raise ValueError("Fixed-rho protocol was not executed")
        if plan is not None:
            sample=plan["samples"][row["deadline_type"]][str(row["eval_seed"])]
            if any(row[k]!=sample[k] for k in ["physical_sha256","deadlines_sha256","tasks","edges"]):
                raise ValueError("Workload or deadline pairing mismatch")


def load_variant(run):
    from tools import eval_all_baselines as e
    payload=e.torch.load(ROOT/run["checkpoint"],map_location="cpu",weights_only=False)
    config=copy.deepcopy(payload["config"])
    agent_config=copy.deepcopy(config["agent"])
    agent_config["use_low_agent"]=not run["fixed_rho_one"]
    if payload["checkpoint_format"]!="hierarchical_reliability_aware_d3qn_v1":
        raise ValueError("Unexpected checkpoint family")
    from tools.train_reliability_aware_hierarchical_d3qn import ReliabilityAwareHierarchicalD3QNAgent
    agent=ReliabilityAwareHierarchicalD3QNAgent(agent_config,seed=742000)
    agent.load_checkpoint(ROOT/run["checkpoint"])
    agent.high_agent.epsilon=0.
    if run["fixed_rho_one"] and agent.low_agent is not None:
        raise ValueError("SAC must be disabled for fixed-rho diagnostics")
    expected_head=run.get("variant")!="no_deadline_rho1"
    if agent.high_agent.deadline_head_enabled!=expected_head:
        raise ValueError("Deadline head execution flag differs")
    return agent,e.HierarchicalObjective(config["objective"])


def initialize(variant):
    global _runtime
    from tools import eval_all_baselines as e
    e.torch.set_num_threads(1)
    plan=read(OUT/"frozen_plan.json")
    config=read(ROOT/plan["config"])
    definition={**plan["variants"][variant],"variant":variant}
    with redirect_stdout(io.StringIO()):
        resources=e.build_resources(config,42)
        agent,objective=load_variant(definition)
    _runtime=(variant,plan,config,definition,resources,agent,objective)


def evaluate_case(case):
    from tools import eval_all_baselines as e
    from tools.eval_alibaba_sl111 import identity
    variant,plan,base,run,resources,agent,objective=_runtime
    deadline,seed=case
    config=e.apply_deadline_type(base,deadline)
    with redirect_stdout(io.StringIO()):
        workflows=e.build_workflows(config,resources,seed,smoke_test=False)
        sample=identity(workflows)
        if sample!=plan["samples"][deadline][str(seed)]:
            raise ValueError("Frozen sample identity differs")
        env=e.build_env(workflows,resources,config["simulation"],seed)
        result=e.collect_d3qn_episode(env,agent,objective,e.HighRolloutBuffer(),e.LowRolloutBuffer(),
                   training=False,high_deterministic=True,low_deterministic=True,
                   fixed_low_rho_one=run["fixed_rho_one"],high_algorithm=run["algorithm"])
    return e.json_ready(dict(scale="small",deadline_type=deadline,eval_seed=seed,mode=variant,
                     num_workflows=50,**sample,**e.add_normalized_metrics(result["metrics"],50)))


def run(variant):
    from tools import eval_all_baselines as e
    plan=read(OUT/"frozen_plan.json")
    check_sources(plan)
    directory=OUT/variant
    directory.mkdir()  # Never restart or resume an existing model evaluation.
    save(directory/"experiment_manifest.json",dict(variant=variant,**plan["variants"][variant],
                workers=2,torch_threads=1,num_workflows=50,evaluation_seed_schedule={d:SEEDS for d in DEADLINES},
                model_initialization_seed=742000,server_seed=42,training=False,plan_sha256=sha(OUT/"frozen_plan.json")))
    print(f"[ablation-start] variant={variant} workers=2 frozen_checkpoint_ep={plan['variants'][variant]['episode']}",flush=True)
    data=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize,initargs=(variant,)) as pool:
        for row in pool.map(evaluate_case,[(d,n) for d in DEADLINES for n in SEEDS]):
            data.append(row)
            validate_rows(data,variant,plan=plan)
            e.write_rows(directory,data)
            print(f"[ablation-eval] variant={variant} {len(data)}/60 deadline={row['deadline_type']} seed={row['eval_seed']} "
                  f"energy_J={row['total_energy']:.2f} deadline_rate={row['deadline_success_rate']:.4f} reliability={row['reliability_success_rate']:.4f}",flush=True)
    validate_rows(data,variant,complete=True,plan=plan)
    check_sources(plan)
    failures=[dict(deadline_type=r["deadline_type"],eval_seed=r["eval_seed"],deadline_success_rate=r["deadline_success_rate"],
                   reliability_success_rate=r["reliability_success_rate"]) for r in data
              if r["deadline_success_rate"]<.9-1e-9 or r["reliability_success_rate"]<1-1e-9]
    save(directory/"completion.json",dict(records=60,source_hashes_verified=True,failures=failures))
    report()
    print(f"[ablation-complete] variant={variant} records=60 failures={len(failures)}",flush=True)


def report():
    plan=read(OUT/"frozen_plan.json")
    check_sources(plan)
    datasets={"Full (original, unfiltered)": [r for r in rows(ROOT/plan["original_full"]) if r["scale"]=="small"]}
    validate_rows(next(iter(datasets.values())),"d3qn_sac",complete=True)
    all_complete=True
    for name in VARIANTS:
        path=OUT/name/"comparison_rows.jsonl"
        data=rows(path) if path.exists() else []
        validate_rows(data,name,plan=plan)
        completed=(OUT/name/"completion.json").exists()
        all_complete=all_complete and completed
        if completed:
            validate_rows(data,name,complete=True,plan=plan)
        datasets[VARIANTS[name][0]]=data
    lines=["# SM74消融评估（原始未筛选种子）", "", "74台、Small、50工作流/seed、0.03/s、三档Deadline，各20种子。能耗J均值±样本SD。",
           "ESP@90=每批至少45/50按时的种子比例；可靠性达标=每批所有工作流R>=0.9的种子比例。", "",
           "HEFT+SAC按原段与恢复段共59轮双验证选择原第240轮；无头模型无可行验证，固定第1200轮末轮诊断。",
           "Full正常策略复用已有同口径60条；固定Full rho=1是冻结策略诊断，不是独立重训完整上层。",
           "HEFT恢复未保存replay；本实验单训练seed，不宣称独立重复或无头严格单变量因果贡献。", "",
           "| Deadline | Variant | Energy (J, mean ± sample SD) | ESP@90 | Reliability attainment | Status |",
           "|---|---|---:|---:|---:|---|"]
    for d in DEADLINES:
        for label,data in datasets.items():
            group=[r for r in data if r["deadline_type"]==d]
            if not group:
                lines.append(f"| {d} | {label} | — | — | — | pending |")
                continue
            energy=[r["total_energy"] for r in group]
            spread=f"{statistics.stdev(energy):.2f}" if len(energy)>1 else "—"
            esp=statistics.mean(r["deadline_success_rate"]>=.9-1e-9 for r in group)
            rel=statistics.mean(r["reliability_success_rate"]>=1-1e-9 for r in group)
            status="complete" if len(group)==20 else f"partial {len(group)}/20, not final"
            lines.append(f"| {d} | {label} | {statistics.mean(energy):.2f} ± {spread} | {esp:.0%} | {rel:.0%} | {status} |")
    lines += ["",f"整体状态：{'全部完成' if all_complete else '进行中，部分结果不能视为最终'}。",
              "所有失败保留；不满足约束的低能耗不作为优势。模型、配置、验证排名和样本指纹见frozen_plan.json与validation_selection.json。"]
    path=OUT/"REPORT.md"
    temporary=path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    temporary.replace(path)
    print(f"[ablation-report] all_complete={all_complete}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","run","report"])
    parser.add_argument("variant",nargs="?",choices=list(VARIANTS))
    args=parser.parse_args()
    if args.action=="run":
        if not args.variant:
            parser.error("run requires a variant")
        run(args.variant)
    else:
        {"prepare":prepare,"report":report}[args.action]()
