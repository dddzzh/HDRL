"""Read-only CSV audit and constrained TeX patch; never start evaluations.

TeX changes are emitted as an apply_patch patch, not written directly.
"""
import argparse
import csv
import difflib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.paper_csv_data import main_panel, read_csv, stats, sha, PALETTE, MODES, SCALES, DEADLINES

OUT = ROOT / 'docs/figures/paper_refresh_20261027'
TEX = ROOT / 'paper/manuscript/main2.tex'


def replacements(original):
    _, _, cells = main_panel()
    changes = []
    def replace(old, new):
        assert original.count(old) == 1, (old[:90], original.count(old))
        changes.append((old, new))
    def cell(sc, d, m):
        return cells[{'S':'ss','M':'sm','L':'sl'}[sc[1]],
                     {'S':'small','M':'medium','L':'large'}[sc[0]],d,m]
    scenarios = ('SS','SM','SL','MS','MM','ML','LS','LM','LL')
    reductions=[]
    for sc in scenarios:
        for d in DEADLINES:
            ours=cell(sc,d,'d3qn_sac')['energy_mean_J']
            best=min(cell(sc,d,m)['energy_mean_J'] for m in MODES[1:])
            assert ours<best
            reductions.append(100*(1-ours/best))
    span=f'{min(reductions):.2f}--{max(reductions):.2f}'
    for anchor in ('reduces mean computation energy by ', 'conditions, reducing energy by '):
        replace(anchor+'10.92--57.02',anchor+span)
    replace('energy by 10.92--57.02\\% relative to the lowest-energy adapted\nbaseline',
            'energy by '+span+'\\% relative to the lowest-energy adapted\nbaseline')
    replace('workload test. On the fixed scientific-workflow test sets, CAHDRL',
            'workload test. On an outcome-selected scientific-workflow seed block, CAHDRL')
    replace('$9\\times3\\times4\\times20=2160$ runs. For realization $e$,',
            '$9\\times3\\times4\\times20=2160$ runs. The main comparison uses\n'
            'the contiguous workflow-seed block 0--19, adopted after inspecting\n'
            "CAHDRL's preliminary outcomes; the three baselines then use this\n"
            'same complete block with no failed observations removed. Thus,\n'
            'these are conditional, exploratory paired results, not an\n'
            'unconditional estimate from an outcome-independent test sample.\n'
            'The earlier blocks and all raw records are retained. For realization $e$,')
    replace('energy per realization. Text and ablations use kJ with sample\n    standard deviations; the main table and boxplots use J.',
            'energy per realization, reported in J throughout. Where shown,\n    dispersion is the sample standard deviation across realizations.')
    q=cell('SS','loose','d3qn_sac'); b=cell('SS','loose','dream_dqn')
    t=cell('LL','tight','d3qn_sac'); u=cell('LL','tight','oo_actor_critic')
    start=original.index('lowest-energy adapted baseline in each condition. In SS--Loose, it')
    end=original.index('Fig.~\\ref{fig:generalization_energy} summarize the results.',start)+len('Fig.~\\ref{fig:generalization_energy} summarize the results.')
    replace(original[start:end],
        'lowest-energy adapted baseline in each condition. In SS--Loose, it\n'
        f"consumes ${q['energy_mean_J']:.2f}\\pm{q['energy_sample_sd_J']:.2f}$ J against\n"
        f"${b['energy_mean_J']:.2f}\\pm{b['energy_sample_sd_J']:.2f}$ J for DReaM-DQN,\n"
        f'a reduction of {max(reductions):.2f}\\%; in LL--Tight, it consumes\n'
        f"${t['energy_mean_J']:.2f}$ J against OOAC's ${u['energy_mean_J']:.2f}$ J,\n"
        f'a reduction of {min(reductions):.2f}\\%. CAHDRL\'s mean energy increases\n'
        'from Loose to Tight deadlines at every workload/resource scale.\n'
        'Table~\\ref{tab:constraint_results} and\n'
        'Fig.~\\ref{fig:generalization_energy} summarize the results.')
    for sc in scenarios:
        assert cell(sc,'loose','d3qn_sac')['energy_mean_J']<cell(sc,'medium','d3qn_sac')['energy_mean_J']<cell(sc,'tight','d3qn_sac')['energy_mean_J']
    replace('All methods attain 100\\% for both metrics in all 27 conditions, so\nthe comparison concerns energy at equal sample attainment.',
            'CAHDRL, DQN-Edge, and OOAC attain 100\\% for both metrics in all\n'
            '27 conditions. DReaM-DQN has lower attainment in eight conditions:\n'
            'ESP@90 ranges from 65\\% to 100\\%, and reliability attainment\n'
            'ranges from 35\\% to 100\\%. Consequently, energy comparisons\n'
            'must be read together with both constraint metrics rather than\n'
            'assuming equal sample feasibility for every method.')
    start=original.index('\\label{tab:constraint_results}')
    end=original.index('\\bottomrule',start)
    lines=[l for l in original[start:end].splitlines() if '\\textbf{' in l]
    assert len(lines)==27
    i=0
    for d in DEADLINES:
        for j,sc in enumerate(scenarios):
            values=[]
            for m in MODES:
                c=cell(sc,d,m); e=f"{c['energy_mean_J']:.2f}"
                if m=='d3qn_sac': e='\\textbf{'+e+'}'
                values.extend((e,f"{100*c['esp_at_90']:g}",f"{100*c['reliability_attainment']:g}"))
            new=(d.capitalize() if j==0 else '')+' & '+sc+' & '+' & '.join(values)+' \\\\'
            replace(lines[i],new); i+=1
    for old,new in (('pic/framework1.pdf','pic/scheduling_framework.pdf'),
                    ('pic/energy_boxplots_success_selected_separators.pdf','pic/main_experiment.pdf'),
                    ('pic/alibaba_energy_boxplot_separators.pdf','pic/alibaba_generalization.pdf'),
                    ('pic/energy_grouped_bars_J_refined_colors.pdf','pic/sensitivity_analysis.pdf')):
        replace(old,new)
    replace('the protocol of Section~\\ref{subsec:experimental_setup}:',
            'the protocol of Section~\\ref{subsec:experimental_setup}, retaining\n'
            'its original paired workload-seed block 742000--742019 rather\n'
            'than replacing it with the new main-comparison block:')
    replace('\\label{tab:constraint_results}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.3pt}\n\\renewcommand{\\arraystretch}{1.00}',
            '\\label{tab:constraint_results}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.3pt}\n\\renewcommand{\\arraystretch}{0.96}')
    nonmain_replacements(original,replace)
    replace('scientific-workflow test sets, CAHDRL reduced mean computation',
            'the outcome-selected scientific-workflow block, CAHDRL reduced mean computation')
    result=original
    for old,new in changes:
        assert result.count(old)==1,old[:80]
        result=result.replace(old,new,1)
    assert 'kJ' not in result
    return result


def nonmain_replacements(original,replace):
    """Only change units, recomputing numbers from the same original CSVs."""
    sources={'CAHDRL':ROOT/'results/legacy_train_new_eval_20261009/sm_74/comparison_rows.csv'}
    for name,folder in (('CAHDRL, $\\rho=1$','full_rho1'),('HEFT+SAC','heft_sac'),
                        ('HEFT, $\\rho=1$','heft_rho1'),('No deadline head, $\\rho=1$','no_deadline_rho1')):
        sources[name]=ROOT/'results/ablation_sm74_new_eval_20261015'/folder/'comparison_rows.csv'
    start=original.index('CAHDRL\n& $\\boldsymbol',original.index('\\label{tab:sm74_ablation}'))
    end=original.index('\\bottomrule',start)
    lines=[]
    for name,path in sources.items():
        rows=[r for r in read_csv(path) if r['scale']=='small']
        lines.append(name)
        for j,d in enumerate(DEADLINES):
            c=stats([r for r in rows if r['deadline_type']==d])
            value=f"{c['energy_mean_J']:.0f}\\pm{c['energy_sample_sd_J']:.0f}"
            if name=='CAHDRL': value='\\boldsymbol{'+value+'}'
            lines.append(f"& ${value}$ & {100*c['esp_at_90']:g} & {100*c['reliability_attainment']:g}"+(' \\\\' if j==2 else ''))
    replace(original[start:end],'\n'.join(lines)+'\n')
    old='& EC (kJ) & ESP (\\%) & Rel. (\\%)\n& EC (kJ) & ESP (\\%) & Rel. (\\%)\n& EC (kJ) & ESP (\\%) & Rel. (\\%) \\\\'
    replace(old,old.replace('(kJ)','(J)'))
    ali={}
    for m in MODES:
        rows=read_csv(ROOT/'results/alibaba_sl111_100wf_20261015'/m/'comparison_rows.csv')
        for d in DEADLINES: ali[m,d]=stats([r for r in rows if r['deadline_type']==d])
    def ms(c): return f"{c['energy_mean_J']:.2f}\\pm{c['energy_sample_sd_J']:.2f}"
    start=original.index('lowest mean energy under all three regimes: ')
    end=original.index('corresponding to mean reductions of ',start)
    replace(original[start:end],'lowest mean energy under all three regimes: $'+ms(ali['d3qn_sac','loose'])+'$,\n$'+ms(ali['d3qn_sac','medium'])+'$, and $'+ms(ali['d3qn_sac','tight'])+'$ J under Loose, Medium,\nand Tight deadlines. The lowest-energy feasible baseline is DQN-Edge\nunder Loose deadlines ($'+ms(ali['dqn_edge','loose'])+'$ J) and OOAC under Medium\nand Tight ($'+ms(ali['oo_actor_critic','medium'])+'$ and $'+ms(ali['oo_actor_critic','tight'])+'$ J),\n')
    counts=[stats(read_csv(ROOT/f'results/workflow_count_sm74_20261017/d3qn_sac/n{n}/comparison_rows.csv')) for n in (50,250)]
    rates=[stats(read_csv(ROOT/f'results/arrival_rate_sm74_20261017/d3qn_sac/{tag}/comparison_rows.csv')) for tag in ('rate_0p01','rate_0p20')]
    replace('from 378.35 to 1,871.50~kJ',f"from {counts[0]['energy_mean_J']:,.2f} to {counts[1]['energy_mean_J']:,.2f}~J")
    replace('from 7.57 to 7.49~kJ',f"from {counts[0]['energy_mean_J']/50:,.2f} to {counts[1]['energy_mean_J']/250:,.2f}~J")
    replace('from 372.95 to 453.44~kJ',f"from {rates[0]['energy_mean_J']:,.2f} to {rates[1]['energy_mean_J']:,.2f}~J")


def audit():
    _,sources,cells=main_panel()
    OUT.mkdir(parents=True,exist_ok=True)
    fields=['resource','scale','deadline_type','mode','n','energy_mean_J','energy_sample_sd_J','esp_at_90','reliability_attainment']
    with (OUT/'main_cells_from_original_csv.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for (r,s,d,m),values in cells.items(): w.writerow(dict(resource=r,scale=s,deadline_type=d,mode=m,**values))
    artifacts={str(Path(s['path']).relative_to(ROOT)):sha(s['path']) for s in sources}
    for source in sources:
        p=Path(source['path']).with_suffix('.jsonl')
        artifacts[str(p.relative_to(ROOT))]=sha(p)
    for name in ('alibaba_sl111_100wf_20261015','ablation_sm74_new_eval_20261015','workflow_count_sm74_20261017','arrival_rate_sm74_20261017'):
        for p in (ROOT/'results'/name).rglob('comparison_rows.*'):
            if p.suffix in ('.csv','.jsonl'): artifacts[str(p.relative_to(ROOT))]=sha(p)
    (OUT/'data_audit.json').write_text(json.dumps(dict(records=2160,cells=108,seeds=list(range(20)),unit='J',
        selection_is_conditional=True,palette=PALETTE,source_sha256=artifacts,sources=sources),indent=2)+'\n',encoding='utf-8')
    print('Audited 2160 original CSV records; exported 108 cells; source files unchanged.')


def recolor():
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ContentStream, FloatObject, NameObject
    path=ROOT/'paper/manuscript/backups/main_data_palette_20261027/pic/framework1.pdf'
    reader=PdfReader(path); page=reader.pages[0]
    original_text=page.extract_text()
    original_box=tuple(page.mediabox)
    stream=ContentStream(page.get_contents(),reader)
    before_ops=[(tuple(str(v) for v in operands),op) for operands,op in stream.operations]
    old_colors=[(.8784,.9176,.9647),(.6588,.7529,.8941),(.7333,.8667,.698),(.9608,.6667,.5137),(.7294,.8784,.8902)]
    hexes=['#4C78A8','#4C78A8','#83A783','#D99A5B','#A28AA7']
    whites=[.88,.35,.58,.40,.60]
    mapping=[]
    for old,h,w in zip(old_colors,hexes,whites):
        rgb=[int(h[i:i+2],16)/255 for i in (1,3,5)]
        mapping.append((old,tuple(w+(1-w)*v for v in rgb)))
    changed=0
    for operands,op in stream.operations:
        if op not in (b'rg',b'RG'): continue
        current=tuple(float(v) for v in operands)
        for old,new in mapping:
            if all(abs(a-b)<.0002 for a,b in zip(current,old)):
                operands[:]=[FloatObject(v) for v in new]; changed+=1; break
    assert changed>200
    page[NameObject('/Contents')]=stream
    writer=PdfWriter(); writer.add_page(page)
    target=ROOT/'output/pdf/framework1_blue.pdf'; target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f: writer.write(f)
    after=PdfReader(target).pages[0]
    assert original_box==tuple(after.mediabox)
    assert original_text==after.extract_text()
    after_ops=ContentStream(after.get_contents(),after.pdf).operations
    assert len(before_ops)==len(after_ops)
    for (before,op),(args,op2) in zip(before_ops,after_ops):
        assert op==op2
        if op not in (b'rg',b'RG'): assert before==tuple(str(v) for v in args)
    (OUT/'framework_palette.json').write_text(json.dumps(dict(source=str(path),source_sha256=sha(path),
        colored_operators_changed=changed,mapping=[dict(old=old,new=new) for old,new in mapping],
        note='Muted categorical palette; light framework fills preserve dark-text contrast; no geometry/text edits'),indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tex-patch',action='store_true'); p.add_argument('--recolor',action='store_true')
    args=p.parse_args()
    if args.tex_patch:
        original=TEX.read_text(encoding='utf-8'); revised=replacements(original)
        diff=list(difflib.unified_diff(original.splitlines(),revised.splitlines(),n=3,lineterm=''))[2:]
        print('*** Begin Patch\n*** Update File: paper/manuscript/main2.tex')
        for line in diff: print('@@' if line.startswith('@@') else line)
        print('*** End Patch')
    elif args.recolor: recolor()
    else: audit()
