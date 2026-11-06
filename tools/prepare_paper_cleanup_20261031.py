"""Stage verified current-paper evidence and an explicit cleanup plan. No deletion."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.paper_csv_data import main_panel, read_csv, sha

CURRENT = ROOT/'results/paper_current'
ARCHIVE = ROOT/'old/paper_cleanup_20261031'
CHECKPOINT_EXTENSIONS = {'.pt','.pth','.ckpt'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not CURRENT.exists() and not ARCHIVE.exists(), 'Refuse overwrite or duplicate cleanup'
    _, main_sources, _ = main_panel(independent_ema=True)
    datasets = []
    for source in main_sources:
        path = Path(source['path'])
        sample = read_csv(path)
        resource = next(s for s in ('ss_37','sm_74','sl_111') if s in str(path))
        mode = sample[0]['mode']
        datasets.append(dict(experiment='main_0_19',mode=mode,resource=resource,
            source_dir=str(path.parent.relative_to(ROOT)), target_dir=f'main_0_19/{resource}/{mode}',
            records=180, paper_records=180))
    extra=ROOT/'results/dqn_edge_independent_ema_extra_20261031'
    for group, old in [('alibaba','alibaba_sl111_100wf_20261015'),
                       ('workflow_count','workflow_count_sm74_20261017'),
                       ('arrival_rate','arrival_rate_sm74_20261017')]:
        for mode in ('dqn_edge','dream_dqn','oo_actor_critic','d3qn_sac'):
            source = extra/group if mode=='dqn_edge' else ROOT/'results'/old/mode
            count = sum(len(read_csv(p)) for p in source.rglob('comparison_rows.csv'))
            assert count == (60 if group=='alibaba' else 100)
            datasets.append(dict(experiment=group,mode=mode,resource='sl_111' if group=='alibaba' else 'sm_74',
                source_dir=str(source.relative_to(ROOT)),target_dir=f'{group}/{mode}',
                records=count,paper_records=count))
    ablation=ROOT/'results/ablation_sm74_new_eval_20261015'
    for mode in ('full_rho1','heft_sac','heft_rho1','no_deadline_rho1'):
        datasets.append(dict(experiment='ablation_sm74',mode=mode,resource='sm_74',
            source_dir=str((ablation/mode).relative_to(ROOT)),target_dir=f'ablation_sm74/{mode}',records=60,paper_records=60))
    datasets.append(dict(experiment='ablation_sm74',mode='cahdrl',resource='sm_74',
        source_dir='results/legacy_train_new_eval_20261009/sm_74',target_dir='ablation_sm74/cahdrl',records=180,paper_records=60))
    model_entries=[]
    for filename in ('results/legacy_seed_block_0_19_20261026/frozen_plan.json',
                     'results/baselines_seed_block_0_19_20261026/frozen_plan.json'):
        for m in read(ROOT/filename)['models']:
            mode=m.get('mode','d3qn_sac')
            resource=m.get('resource',m['id'])
            model_entries.append(dict(id=f'{resource}_{mode}',mode=mode,resource=resource,
                episode=m['episode'],source_checkpoint=m['checkpoint'].replace('\\','/'),
                sha256=m['checkpoint_sha256'],plan=filename))
    variants=read(ablation/'frozen_plan.json')['variants']
    for name in ('heft_sac','no_deadline_rho1'):
        m=variants[name]
        model_entries.append(dict(id=name,mode=name,resource='sm_74',episode=m['episode'],
            source_checkpoint=m['checkpoint'].replace('\\','/'),sha256=m['checkpoint_sha256'],
            plan=str((ablation/'frozen_plan.json').relative_to(ROOT))))
    assert len(model_entries)==14 and len({m['source_checkpoint'] for m in model_entries})==14
    for m in model_entries:
        source=ROOT/m['source_checkpoint'];assert sha(source)==m['sha256'],source
        m['checkpoint']=f"models/{m['id']}/{source.name}"
    CURRENT.mkdir();ARCHIVE.mkdir()
    copies={}
    def copy_file(source,target):
        source=Path(source); target=Path(target)
        assert source.is_file() and target.is_relative_to(CURRENT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert sha(source)==sha(target)
        else:
            shutil.copy2(source,target)
        digest=sha(source);assert sha(target)==digest
        copies[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(ROOT)),sha256=digest,size=source.stat().st_size)
    def copy_tree(source,target):
        for p in Path(source).rglob('*'):
            if p.is_file() and p.suffix.lower() not in CHECKPOINT_EXTENSIONS:
                copy_file(p,Path(target)/p.relative_to(source))
    for d in datasets:
        copy_tree(ROOT/d['source_dir'],CURRENT/d['target_dir'])
    for m in model_entries:
        source=ROOT/m['source_checkpoint']
        copy_file(source,CURRENT/m['checkpoint'])
        for p in source.parent.iterdir():
            if p.is_file() and p.suffix.lower() not in CHECKPOINT_EXTENSIONS:
                copy_file(p,CURRENT/'models'/m['id']/p.name)
    copy_tree(ROOT/'results/reliability_diagnostic_20261015',CURRENT/'reliability_diagnostic')
    copy_tree(ROOT/'results/reliability_closed_loop_precheck_20261019',CURRENT/'reliability_precheck')
    for dirname in ('paper_refresh_independent_ema_20261031','paper_refresh_all_independent_ema_20261031'):
        copy_tree(ROOT/'docs/figures'/dirname,CURRENT/'statistics'/dirname)
    plans = ['results/legacy_seed_block_0_19_20261026/frozen_plan.json',
             'results/baselines_seed_block_0_19_20261026/frozen_plan.json',
             'results/dqn_edge_independent_ema_0_19_20261029/frozen_plan.json',
             'results/dqn_edge_independent_ema_extra_20261031/frozen_plan.json',
             'results/ablation_sm74_new_eval_20261015/frozen_plan.json',
             'results/ablation_sm74_new_eval_20261015/validation_selection.json',
             'results/alibaba_sl111_100wf_20261015/evaluation_plan.json',
             'results/workflow_count_sm74_20261017/frozen_plan.json',
             'results/arrival_rate_sm74_20261017/frozen_plan.json',
             'results/legacy_train_new_eval_20261009/PROTOCOL.md']
    for filename in plans:copy_file(ROOT/filename,CURRENT/'provenance'/filename)
    for m in model_entries:
        d=read(ROOT/m['plan'])
        # Preserve the recorded config/source snapshots without rewriting historical manifests.
        for filename in d.get('source_sha256',{}):
            source=ROOT/filename
            if source.is_file() and source.suffix.lower() not in CHECKPOINT_EXTENSIONS and not filename.replace('\\','/').startswith('tests/'):
                if source.suffix.lower() in ('.json','.py','.xml','.md') and source.stat().st_size<5_000_000:
                    copy_file(source,CURRENT/'provenance/source_snapshots'/source.relative_to(ROOT))
    checkpoints=[]; archives=[]
    for entry in (ROOT/'results').iterdir():
        if entry==CURRENT:continue
        archives.append(dict(source=str(entry.relative_to(ROOT)),destination=str((ARCHIVE/'results'/entry.name).relative_to(ROOT))))
        paths=entry.rglob('*') if entry.is_dir() else [entry]
        for p in paths:
            if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS:
                checkpoints.append(dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,
                    modified_utc_ticks=p.stat().st_mtime_ns,selected_source_copy=any(p==ROOT/m['source_checkpoint'] for m in model_entries)))
    manifest=dict(version=1,date='2026-10-31',datasets=datasets,models=model_entries,files=copies,
        main_seeds=list(range(20)),supplementary_seeds=list(range(742000,742020)),
        dqn_edge_protocol='Restore checkpoint EMA before every realization; update within realization',
        main_records=2160,supplementary_records=1040,ablation_paper_records=300,
        missing_evidence=['D3QN-RFR: manuscript values have no identified matching raw run; not represented as verified data'],
        historical_archive='old/paper_cleanup_20261031/results',
        historical_paths='Original paths inside frozen records remain unchanged; use file source mapping in this manifest.')
    (CURRENT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plan=dict(root=str(ROOT),current=str(CURRENT),archive=str(ARCHIVE),copies=copies,models=model_entries,
        archive_moves=archives,delete_checkpoints=checkpoints,delete_tests='tests',
        test_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),size=p.stat().st_size) for p in (ROOT/'tests').rglob('*') if p.is_file()],
        expected_checkpoint_count=len(checkpoints),checkpoint_bytes=sum(p['size'] for p in checkpoints),
        manifest_sha256=sha(CURRENT/'manifest.json'),prepared_only=True)
    (ARCHIVE/'cleanup_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(staged_files=len(copies),models=len(model_entries),datasets=len(datasets),
        delete_checkpoint_count=len(checkpoints),delete_checkpoint_GiB=plan['checkpoint_bytes']/2**30,
        tests=len(plan['test_files']),plan=str(ARCHIVE/'cleanup_plan.json')),ensure_ascii=False))


if __name__=='__main__':main()
