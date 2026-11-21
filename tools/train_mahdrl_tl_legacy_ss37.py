"""Train/evaluate the documented MAHDRL-TL+R adaptation on the unchanged legacy scene."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.baselines.mahdrl_tl_legacy import (
    FORMAT, UPSTREAM_COMMIT, LegacyTLScene, build_agents, run_episode,
)
from src.device_utils import configure_torch_runtime, resolve_device
from src.sac.two_sided_dual import ADAPTIVE_SETTINGS, AdaptiveDuals
from tools.hierarchical_common import build_resources, build_workflows


def configuration(requirement=.95, fault_scale=12., deadline_type="medium", *, small50_fifs=False, small20_fifs=False):
    original = json.loads((ROOT/"configs/sac_action_fix_ss37_20261105.json").read_text(encoding="utf-8"))
    if small20_fifs:
        original = json.loads((ROOT/"configs/ss37_fifs_small20_arr003_comparison.json").read_text(encoding="utf-8"))["config"]
    cfg = {k: deepcopy(original[k]) for k in ("generator", "resources", "simulation")}
    if small50_fifs:
        cfg["generator"].update(dataset_scale="small", num_workflows=50, arrival_rate=.1,
                                deadline_reference="shared_fcfs_first_idle_rho1")
        cfg["simulation"]["reliability_budget_state"] = True
    cfg["simulation"]["workflow_reliability_requirement"] = requirement
    cfg["simulation"]["deadline_type"] = deadline_type
    cfg["legacy_failure_rate_scale"] = fault_scale
    return cfg


def resources_for(cfg):
    resources = build_resources(cfg, 42)
    for server in resources["servers"]:
        server["failure_rate"] *= cfg["legacy_failure_rate_scale"]
    return resources


def episode(cfg, resources, agents, seed, **kwargs):
    workflows = build_workflows(cfg, resources, seed)
    scene = LegacyTLScene(workflows, resources, cfg["simulation"], seed)
    return dict(seed=seed, **run_episode(scene, agents, **kwargs))


def initialize_validation_worker(node_count, batch_size):
    global _validation_agents
    configure_torch_runtime(1)
    _validation_agents = build_agents(node_count, batch_size=batch_size, device="cpu")


def validation_chunk(cfg, resources, seeds, weights, price):
    for role, state in weights.items():
        _validation_agents[role].online.load_state_dict(state)
    return [episode(cfg, resources, _validation_agents, seed, risk_price=price) for seed in seeds]


def evaluation(cfg, resources, trained, evaluator, seeds, price, *, pool=None, workers=1):
    seeds = list(seeds)
    if pool is not None:
        weights = {role: {name: value.detach().cpu().clone()
            for name, value in agent.online.state_dict().items()} for role, agent in trained.items()}
        jobs = [pool.submit(validation_chunk, cfg, resources, seeds[i::workers], weights, price)
            for i in range(min(workers, len(seeds)))]
        by_seed = {row['seed']: row for job in jobs for row in job.result()}
        return [by_seed[seed] for seed in seeds]
    for role in trained:
        evaluator[role].online.load_state_dict(trained[role].online.state_dict())
    return [episode(cfg, resources, evaluator, seed, risk_price=price) for seed in seeds]


def evaluation_summary(rows):
    count = sum(r["workflow_count"] for r in rows)
    ddl = sum(r["deadline_missed_workflows"] for r in rows)
    rel = sum(r["reliability_missed_workflows"] for r in rows)
    return dict(instances=len(rows), workflow_count=count,
        esp90=sum(bool(r["ESP90"]) for r in rows),
        joint=sum(bool(r["ESP90"]) and r["reliability_missed_workflows"] == 0 for r in rows),
        deadline_missed_workflows=ddl, reliability_missed_workflows=rel,
        deadline_success_rate=1.-ddl/count, reliability_success_rate=1.-rel/count,
        mean_energy_J=float(np.mean([r["energy_J"] for r in rows])),
        mean_worst_hazard_ratio=float(np.mean([r["worst_hazard_ratio"] for r in rows])),
        max_worst_hazard_ratio=max(r["worst_hazard_ratio"] for r in rows))


def print_evaluation(ep, rows, *, block=False):
    s = evaluation_summary(rows)
    print(f"[{'VAL_BLOCK' if block else 'VAL'}] ep={ep} first_seed={rows[0]['seed']} last_seed={rows[-1]['seed']} "
        f"ESP90={s['esp90']}/{s['instances']} joint={s['joint']}/{s['instances']} "
        f"DDL_MISS={s['deadline_missed_workflows']}/{s['workflow_count']} "
        f"REL_MISS={s['reliability_missed_workflows']}/{s['workflow_count']} "
        f"DSR={s['deadline_success_rate']:.2%} RSR={s['reliability_success_rate']:.2%} "
        f"energy_J={s['mean_energy_J']:.2f} risk_max={s['max_worst_hazard_ratio']:.4f} "
        "policy=greedy epsilon=0 rho=1", flush=True)
    return s


def resume_protocol_compatible(previous, current):
    if previous == current:
        return True
    # Known driver revisions may migrate validation execution and reporting only.
    driver = "tools/train_mahdrl_tl_legacy_ss37.py"
    if previous.get("source_sha256", {}).get(driver) not in (
        "7663e1b825a2716f06bd988f6417b544eb8c270948f5323a2982e1347476b0ba",
        "fd15c617a665d37731ea30b4a3be61da588984701764aa8e06367dea7a4fb5fd",
        "659312ddffe7d1f3bc2978459ed38c870ec393940de47bf139baed1a39b91b93",
        "dcda7029a8c932ad5abbecb194f0297d42e815325c825ca85237c9a733c1818f"):
        return False
    comparable = deepcopy(previous)
    comparable["validate_every"] = current["validate_every"]
    comparable["source_sha256"][driver] = current["source_sha256"][driver]
    return comparable == current


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")


def append_evaluation_csv(path, ep, summary, price):
    row = dict(轮次=ep, 环境数=summary['instances'], 工作流数=summary['workflow_count'],
        ESP90达标批次=summary['esp90'], 联合达标批次=summary['joint'],
        超期工作流数=summary['deadline_missed_workflows'],
        可靠性违约工作流数=summary['reliability_missed_workflows'],
        平均能耗_kJ=round(summary['mean_energy_J']/1000, 2),
        按时完成率_DSR=f"{summary['deadline_success_rate']:.2%}",
        可靠性达标率_RSR=f"{summary['reliability_success_rate']:.2%}",
        lambda_R=price)
    header = not path.exists()
    with path.open('a', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(out, ep, payload, selected):
    checkpoint = out/f"episode_{ep:04d}.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    link = out/"latest.link"
    link.unlink(missing_ok=True)
    try:
        link.hardlink_to(checkpoint)
    except OSError:
        shutil.copyfile(checkpoint, link)
    link.replace(out/"latest.pt")
    if selected:
        shutil.copyfile(checkpoint, out/"best_esp90.tmp")
        (out/"best_esp90.tmp").replace(out/"best_esp90.pt")
        if payload["best_key"][0] == 40:
            shutil.copyfile(checkpoint, out/"best_feasible.tmp")
            (out/"best_feasible.tmp").replace(out/"best_feasible.pt")
            print(f"[BEST_FEASIBLE] ep={ep} joint=40/40 energy_J={-payload['best_key'][1]:.2f}", flush=True)


def run(stack):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument("--small20-fifs", action="store_true", help="Small/20, arrival 0.03, FIFS.")
    presets.add_argument("--small50-fifs", action="store_true",
        help="Use 50 Small workflows, arrival rate 0.10 and shared full-frequency FIFS deadlines for training and validation.")
    parser.add_argument("--validate-every", type=int, default=1,
        help="Validate after each training episode, as in the author's training loop.")
    parser.add_argument("--validation-count", type=int, choices=(1, 20), default=20)
    parser.add_argument("--validation-workers", type=int, default=1,
        help="Persistent CPU validation processes; training uses the selected agent device.")
    parser.add_argument("--reliability-required", type=float, default=.95)
    parser.add_argument("--failure-rate-scale", type=float, default=12.)
    parser.add_argument("--deadline-type", choices=("loose", "medium", "tight"))
    parser.add_argument("--scale", choices=("small","middle","large"))
    parser.add_argument("--eval-workflows", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=float, default=.03)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--no-reliability", action="store_true", help="Zero risk price; keep identical physics and report risk failures.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--evaluate", type=Path)
    parser.add_argument("--seed-start", type=int, default=742000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.evaluate and (args.scale is not None or args.eval_workflows is not None):
        parser.error("Scale and evaluation workflow count apply to evaluation only.")
    if args.eval_workflows is not None and args.eval_workflows < 1:
        parser.error("Evaluation workflow count must be positive.")
    if args.evaluate and (args.small50_fifs or args.small20_fifs):
        parser.error("--evaluate uses the environment stored in the checkpoint; omit --small50-fifs.")
    if (args.episodes < 1 or args.validate_every < 1 or args.batch_size < 1
            or not 1 <= args.validation_workers <= 61
            or not 0 <= args.warmup <= 1 or not 0 < args.reliability_required < 1
            or not math.isfinite(args.failure_rate_scale) or args.failure_rate_scale <= 0
            or args.num_seeds < 1):
        parser.error("Invalid episode, replay, environment or evaluation settings.")
    if Path(args.out).name != args.out or not args.out.startswith("legacy_ss37_mahdrl_tl_"):
        parser.error("Output must be a new legacy_ss37_mahdrl_tl_ directory under results/.")
    out = ROOT/"results"/args.out
    if out.exists():
        parser.error("Output exists; use a new name, including when resuming.")
    configure_torch_runtime(1)
    device = str(resolve_device(args.device))
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.model_seed)
    saved = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=False) if args.resume or args.evaluate else None
    if saved is not None and saved.get("format") != FORMAT:
        parser.error("Checkpoint learner differs from MAHDRL-TL single-Q reward-penalty v2.")
    cfg = saved["protocol"]["config"] if args.evaluate else configuration(
        args.reliability_required, args.failure_rate_scale, args.deadline_type or "medium", small50_fifs=args.small50_fifs, small20_fifs=args.small20_fifs)
    if args.evaluate:
        cfg = deepcopy(cfg)
        if args.deadline_type:
            cfg["simulation"]["deadline_type"] = args.deadline_type
        if args.scale:
            cfg["generator"]["dataset_scale"] = args.scale
        if args.eval_workflows:
            cfg["generator"]["num_workflows"] = args.eval_workflows
    resources = resources_for(cfg)
    protocol = dict(config=cfg, upstream_commit=UPSTREAM_COMMIT,
        variant="MAHDRL-TL+R legacy adaptation; TA + global node D3QN; no host agent",
        reliability_enabled=not args.no_reliability, risk_update=dict(ADAPTIVE_SETTINGS),
        risk_cost="marginal absolute placement excess over remaining 0.9B budget with same worst-case reserve; divided by B*N",
        risk_action_mask=False, rho=1., base_reward=dict(alpha=.75, energy_scale=.001, delay_scale=300.),
        learning_reward="original base reward minus current lambda_R times placement risk cost; single Q",
        task_deadline="legacy workflow deadline minus estimated critical-path tail and egress",
        phase_energy="sum of legacy per-task assignment energy; no host power or idle energy added",
        phase_delay="sum of booked task tardiness within phase; assignment-time credit",
        exploration="upstream TA epsilon-greedy; VA behavior_probabilities with 0.5 guided fastest exploration",
        batch_size=args.batch_size, warmup=args.warmup, model_seed=args.model_seed,
        validate_every=args.validate_every, validation_count=args.validation_count,
        checkpoint_policy="every-validation-plus-latest-and-best", evaluation_networks="online",
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "src/baselines/mahdrl_tl_legacy.py", "tools/train_mahdrl_tl_legacy_ss37.py",
            "third_party/MAHDRL/mahdrl/agents.py", "third_party/MAHDRL/mahdrl/networks.py",
            "third_party/MAHDRL/mahdrl/workflow.py")})
    if args.small20_fifs:
        spec = json.loads((ROOT/"configs/ss37_fifs_small20_arr003_comparison.json").read_text(encoding="utf-8"))
        protocol["training_seed_base"] = spec["protocol"]["training_seed_base"]
    if args.evaluate:
        protocol = dict(saved["protocol"], config=cfg)
    elif saved is not None and not resume_protocol_compatible(saved["protocol"], protocol):
        parser.error("Resume protocol differs; refusing to silently change the experiment.")
    resume_validation_change = (dict(previous=saved["protocol"], current=protocol)
        if args.resume and saved["protocol"] != protocol else None)
    agents = build_agents(len(resources["servers"]), batch_size=protocol["batch_size"], device=device)
    workers = min(args.validation_workers, args.num_seeds if args.evaluate else 2*args.validation_count)
    # Retain baseline initialization so worker count does not change parent RNG.
    evaluator = build_agents(len(resources["servers"]), batch_size=protocol["batch_size"], device="cpu")
    duals, start, best = AdaptiveDuals(), 1, (-1, -math.inf)
    best_source = None
    if saved is not None:
        for role, agent in agents.items():
            agent.restore(saved["agents"][role])
        duals = AdaptiveDuals(**saved["reliability_dual_state"])
        start, best, best_source = saved["episode"]+1, tuple(saved["best_key"]), saved["best_source"]
        if not args.evaluate:
            if start > args.episodes:
                parser.error("Resume total episodes must exceed the saved episode.")
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda") and saved.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(saved["cuda_rng"], device=device)
    del saved
    out.mkdir(parents=True)
    if args.resume and best_source:
        shutil.copyfile(best_source, out/"best_esp90.pt")
        if best[0] == 40:
            shutil.copyfile(best_source, out/"best_feasible.tmp")
            (out/"best_feasible.tmp").replace(out/"best_feasible.pt")
        best_source = str(out/"best_esp90.pt")
    (out/"manifest.json").write_text(json.dumps(dict(protocol=protocol, episodes=args.episodes,
        device=device, validation_device="cpu", resume=str(args.resume), evaluation=str(args.evaluate),
        validation_workers=workers,
        resume_validation_change=resume_validation_change,
        best_feasible_selection="40/40 ESP90-and-reliability batches; minimize mean energy"), indent=2), encoding="utf-8")
    def price():
        return duals.reliability if protocol["reliability_enabled"] else 0.
    pool = (stack.enter_context(ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'), initializer=initialize_validation_worker,
        initargs=(len(resources['servers']), protocol['batch_size']))) if workers > 1 else None)
    print(f"[device] training={device} validation=cpu validation_workers={workers}", flush=True)
    if args.evaluate:
        rows = evaluation(cfg, resources, agents, evaluator,
            range(args.seed_start, args.seed_start+args.num_seeds), price(), pool=pool, workers=workers)
        summary = dict(checkpoint=str(args.evaluate), seeds=[r["seed"] for r in rows],
            **print_evaluation(start-1, rows), rows=rows)
        (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_evaluation_csv(out/'summary.csv', start-1, summary, price())
        print(json.dumps({k:v for k,v in summary.items() if k != "rows"}), flush=True)
        return
    for ep in range(start, args.episodes+1):
        if (out/"stop_requested.json").exists():
            print("train-stopped", out, flush=True)
            return
        before = price()
        row = episode(cfg, resources, agents, protocol.get("training_seed_base", args.model_seed)+ep, training=True,
            risk_price=before, warmup=args.warmup)
        update = duals.update_reliability(row["worst_hazard_ratio"]) if protocol["reliability_enabled"] else dict(action="disabled")
        append(out/"train_history.jsonl", dict(episode=ep, **row, lambda_R_before=before,
            lambda_R_after=price(), reliability_dual_update=update))
        print(f"[TRAIN] ep={ep} seed={row['seed']} energy_J={row['energy_J']:.2f} "
            f"DDL_MISS={row['deadline_missed_workflows']}/{row['workflow_count']} "
            f"REL_MISS={row['reliability_missed_workflows']}/{row['workflow_count']} "
            f"lambda_R={price():.4f} epsilon={row['epsilon']} updates={row['updates']}", flush=True)
        selected = False
        if ep % args.validate_every == 0:
            seeds = list(range(542000,542000+args.validation_count))+list(range(642000,642000+args.validation_count))
            rows = evaluation(cfg, resources, agents, evaluator, seeds, price(), pool=pool, workers=workers)
            summary = print_evaluation(ep, rows)
            blocks = [rows[:args.validation_count], rows[args.validation_count:]]
            for block in blocks:
                print_evaluation(ep, block, block=True)
            key = (summary['joint'], -summary['mean_energy_J'])
            append(out/"eval_history.jsonl", dict(episode=ep, **summary,
                lambda_R=price(), blocks=[rows[:args.validation_count],rows[args.validation_count:]]))
            append_evaluation_csv(out/'eval_history.csv', ep, summary, price())
            if key > best:
                best, selected, best_source = key, True, str(out/"best_esp90.pt")
        if ep % args.validate_every == 0 or ep == args.episodes:
            payload = dict(format=FORMAT, protocol=protocol, episode=ep,
                agents={role:a.snapshot() for role,a in agents.items()},
                reliability_dual_state=duals.state(), best_key=best, best_source=best_source,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                **({"cuda_rng":torch.cuda.get_rng_state(device)} if device.startswith("cuda") else {}))
            save_checkpoint(out, ep, payload, selected)
    print("train-complete", out, flush=True)


def main():
    with ExitStack() as stack:
        run(stack)


if __name__ == "__main__":
    main()
