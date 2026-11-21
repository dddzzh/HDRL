"""Train or evaluate SS37 DQN-Edge with matched workloads and opt-in state, context and energy changes."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_dqn_edge as edge
SMALL20_REFERENCE = ROOT / "configs/ss37_fifs_small20_arr003_comparison.json"

BASE = ROOT / 'configs/dqn_edge_small_28edge_9cloud.json'
REFERENCE = ROOT / 'results/legacy_ss37_fifs_worst_scale_upperD_lowerR_train1500_20261118/manifest.json'
SMALL50_REFERENCE = ROOT / 'results/legacy_ss37_small50_fifs_riskfix_train1500_workers8_20261118/manifest.json'
FORMAT = 'dqn-edge-common-upper-matched-ss37-v1'
REPAIRED_FORMAT = 'dqn-edge-common-upper-matched-ss37-v2'
BLOCKS = (542000, 642000)


def configuration(*, reference=REFERENCE, match_upper_state=False, completed_task_context=False, energy_kj=False):
    ref = edge.load_json(reference)
    cfg = edge.load_json(BASE)
    for key in ('generator', 'resources', 'simulation', 'legacy_failure_rate_scale'):
        cfg[key] = ref['config'][key]
    if match_upper_state:
        cfg['agent']['shared']['state'] = dict(ref['config']['agent']['shared']['state'])
    if completed_task_context:
        cfg['agent']['shared']['state']['completed_task_context'] = True
    if energy_kj:
        cfg['objective']['energy_reward_mode'] = 'kilojoules'
    scale, count = ('small', 50) if reference == SMALL50_REFERENCE else ('small_middle', 20)
    arrival = .1
    if reference == SMALL20_REFERENCE:
        scale, count, arrival = 'small', 20, .03
    if (cfg['generator']['dataset_scale'], cfg['generator']['num_workflows'], cfg['generator']['arrival_rate'],
            cfg['simulation']['deadline_type'], cfg['generator']['deadline_reference'],
            cfg['simulation']['workflow_reliability_requirement'], cfg['legacy_failure_rate_scale']) != (
            scale, count, arrival, 'medium', 'shared_fcfs_first_idle_rho1', .95, 12):
        raise ValueError(f'Reference manifest no longer matches the {scale}/{count} SS37 FIFS protocol.')
    cfg['training'].update(seed=ref['seed'], total_episodes=ref['episodes'],
        eval_interval=ref['protocol']['validate_every'], eval_episodes=ref['protocol']['validation_count'],
        save_every=ref['protocol']['validate_every'], eval_seed=BLOCKS[0], eval_seed_secondary=BLOCKS[1])
    if cfg['training']['eval_episodes'] != 20:
        raise ValueError('The formal selection panel requires two complete 20-seed blocks.')
    for key in ('checkpoint_deadline_esp_threshold', 'checkpoint_deadline_esp_target',
                'checkpoint_reliability_target', 'checkpoint_dir'):
        cfg['training'].pop(key, None)
    return cfg


def build_runtime(cfg, resources, device):
    learner = edge.DQNEdgeConfig.from_mapping(dict(cfg['agent']['dqn_edge'], device=device))
    return edge.build_dqn_edge_runtime(cfg, resources, learner)


def episode(cfg, resources, runtime, seed, train):
    workflows = edge.build_workflows(cfg, resources, seed)
    env = edge.build_env(workflows, resources, cfg['simulation'], seed)
    metrics = edge.run_episode(env, runtime['scheduler'], training=train)
    results = env._build_workflow_results()
    expected = cfg['generator']['num_workflows']
    if not env.done or len(results) != expected:
        raise RuntimeError(f'Incomplete SS37 rollout for seed {seed}')
    if len({w['workflow_id'] for w in results}) != expected or any(w['finish_time'] is None for w in results):
        raise ValueError('Expected unique, completed workflows.')
    cd = math.fsum(max(0., (w['response_time'] - w['relative_deadline'] - 1e-8) / w['relative_deadline'])
                   for w in results) / expected
    cr = max(max(0., math.log(max(w['reliability'], 1e-300)) / math.log(w['reliability_req']) - 1.)
             for w in results)
    counts = dict(workflow_count=expected, deadline_missed_workflows=sum(not w['deadline_satisfied'] for w in results),
                  reliability_missed_workflows=sum(not w['reliability_met'] for w in results))
    if (cr == 0) != (counts['reliability_missed_workflows'] == 0):
        raise ValueError('Reliability excess and simulator success flags disagree.')
    return dict(seed=seed, scale=workflows[0].dataset_scale, energy_J=metrics['total_energy'],
        ESP90=10 * counts['deadline_missed_workflows'] <= expected, CD=cd, CR=cr, **counts,
        episode_reward=metrics['episode_reward'])


def evaluate(cfg, resources, runtime, checkpoint, seeds):
    agent = runtime['agent']
    weights = checkpoint['smoothed_weights']
    for name in ('time_online','energy_online','attention'):
        getattr(agent, name).load_state_dict(checkpoint[name])
    rows = []
    for seed in seeds:
        agent.smoothed_weights = weights.copy()
        rows.append(episode(cfg, resources, runtime, seed, False))
    return rows


def append_rows(path, rows):
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def copy_checkpoint(source, destination):
    temp = destination.with_suffix('.tmp')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def summary(rows):
    energy = sum(r['energy_J'] for r in rows)/len(rows)
    joint = sum(r['ESP90'] and r['reliability_missed_workflows'] == 0 for r in rows)
    return dict(environments=len(rows), joint=joint, ESP90=sum(r['ESP90'] for r in rows),
        DDL_MISS=sum(r['deadline_missed_workflows'] for r in rows),
        REL_MISS=sum(r['reliability_missed_workflows'] for r in rows), mean_energy_J=energy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda', help='Training device; validation uses CPU')
    parser.add_argument('--episodes', type=int)
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument('--small20-fifs', action='store_true', help='Small/20, arrival 0.03, FIFS.')
    presets.add_argument('--small50', action='store_true', help='Match the Small-only 50-workflow training and validation environment.')
    parser.add_argument('--completed-task-context', action='store_true',
                        help='Use latency and energy of finished tasks for attention.')
    parser.add_argument('--match-upper-state', action='store_true',
                        help='Use the reference upper state, including the observed workflow risk budget.')
    parser.add_argument('--energy-kj', action='store_true',
                        help='Use negative task energy in kJ with the existing energy coefficient.')
    parser.add_argument('--evaluate', type=Path, help='Evaluate a matched-protocol checkpoint on a separate seed block')
    parser.add_argument('--eval-workflows', type=int)
    parser.add_argument('--scale', choices=('small','middle','large','small_middle'))
    parser.add_argument('--deadline-type', choices=('loose','medium','tight'))
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--num-seeds', type=int)
    args = parser.parse_args()
    if not args.evaluate and any(v is not None for v in (args.scale,args.deadline_type,args.seed_start,args.num_seeds,args.eval_workflows)):
        parser.error('Test-set overrides require --evaluate.')
    if args.evaluate and args.episodes is not None:
        parser.error('--episodes applies to training.')
    if args.evaluate and (args.small20_fifs or args.small50 or args.match_upper_state or args.completed_task_context or args.energy_kj):
        parser.error('Evaluation restores the training protocol from the checkpoint.')
    if any(v is not None and v < 1 for v in (args.episodes,args.num_seeds,args.eval_workflows)):
        parser.error('Counts must be positive.')
    edge.configure_torch_runtime(1)
    cp = None
    if args.evaluate:
        cp = edge.torch.load(args.evaluate, map_location='cpu', weights_only=False)
        if cp.get('format') not in {FORMAT, REPAIRED_FORMAT}:
            parser.error('Expected a matched-protocol DQN-Edge checkpoint.')
        cfg = cp['experiment_config']
        if args.eval_workflows:
            cfg['generator']['num_workflows'] = args.eval_workflows
        if args.scale:
            cfg['generator']['dataset_scale'] = args.scale
        if args.deadline_type:
            cfg['simulation']['deadline_type'] = args.deadline_type
    else:
        reference = SMALL50_REFERENCE if args.small50 else REFERENCE
        if args.small20_fifs:
            reference = SMALL20_REFERENCE
        cfg = configuration(reference=reference, match_upper_state=args.match_upper_state,
                            completed_task_context=args.completed_task_context, energy_kj=args.energy_kj)
    resources = edge.build_resources(cfg, 42)
    for server in resources['servers']:
        server['failure_rate'] *= cfg['legacy_failure_rate_scale']
    evaluator = build_runtime(cfg, resources, 'cpu')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.evaluate:
        start = 742000 if args.seed_start is None else args.seed_start
        rows = evaluate(cfg, resources, evaluator, cp, range(start,start+(args.num_seeds or 20)))
        append_rows(args.out/'evaluation.csv', [dict(checkpoint=str(args.evaluate.resolve()), episode=cp['round'],
            deadline_type=cfg['simulation']['deadline_type'], **r) for r in rows])
        print('[TEST]', summary(rows), flush=True)
        return

    settings = cfg['training']
    checkpoint_format = (REPAIRED_FORMAT if args.match_upper_state or args.completed_task_context or args.energy_kj
                         else FORMAT)
    rounds = args.episodes or settings['total_episodes']
    cfg['agent']['dqn_edge']['device'] = args.device
    cfg['training']['total_episodes'] = rounds
    runtime = build_runtime(cfg, resources, args.device)
    train_seed_base = edge.load_json(reference)['protocol'].get('training_seed_base', 42)
    manifest = dict(format=checkpoint_format, config=cfg, reference=str(reference.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), algorithm_config=str(BASE.relative_to(ROOT)),
        algorithm_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(), train_seed=f'{train_seed_base} + one-based round',
        validation_blocks=BLOCKS, validation_device='cpu', evaluation_ema='restore training EMA before every seed',
        selection='maximum joint ESP90 and zero reliability excess, then minimum mean energy in J')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    best = (-1, float('-inf'))
    best_strict_energy = float('inf')
    best_reward = float('-inf')
    for ep in range(1,rounds+1):
        row = episode(cfg, resources, runtime, train_seed_base+ep, True)
        best_reward = max(best_reward,row['episode_reward'])
        append_rows(args.out/'train.csv',[dict(episode=ep,**row)])
        print(f"[TRAIN] ep={ep} seed={row['seed']} energy_J={row['energy_J']:.2f} "
              f"DDL_MISS={row['deadline_missed_workflows']}/{row['workflow_count']} "
              f"REL_MISS={row['reliability_missed_workflows']}/{row['workflow_count']}",flush=True)
        validate = ep % settings['eval_interval'] == 0
        if not validate and ep != rounds:
            continue
        # Existing DQN-Edge loaders expect a zero-based episode field.
        cp = runtime['agent'].checkpoint(ep-1,best_reward)
        cp.update(format=checkpoint_format, round=ep, experiment_config=cfg, variant=runtime['variant'],
                  state_config=runtime['state_config'],objective_config=runtime['objective_config'])
        path = args.out/f'episode_{ep:04d}.pt'
        tmp = path.with_suffix('.tmp')
        edge.torch.save(cp,tmp)
        tmp.replace(path)
        if validate:
            rows = evaluate(cfg,resources,evaluator,cp,[s for start in BLOCKS
                            for s in range(start,start+settings['eval_episodes'])])
            report = summary(rows)
            append_rows(args.out/'validation_seeds.csv',[dict(episode=ep,**r) for r in rows])
            append_rows(args.out/'validation.csv',[dict(episode=ep,**report)])
            print('[VAL]',ep,report,flush=True)
            rank = (report['joint'], -report['mean_energy_J'])
            if all(r['CD'] == 0 and r['CR'] == 0 for r in rows) and report['mean_energy_J'] < best_strict_energy:
                best_strict_energy = report['mean_energy_J']
                copy_checkpoint(path, args.out/'best.pt')
            if rank > best:
                best = rank
                copy_checkpoint(path, args.out/'best_esp90.pt')
                if rank[0] == 40:
                    copy_checkpoint(path, args.out/'best_feasible.pt')

if __name__=='__main__':
    main()
