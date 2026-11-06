"""Verify the curated paper evidence without training or changing data."""
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from plot.paper_csv_data import CURRENT,current_dataset,main_panel,sha,stats


def main():
    manifest=json.loads((CURRENT/'manifest.json').read_text(encoding='utf-8'))
    for relative,entry in manifest['files'].items():
        assert sha(ROOT/relative)==entry['sha256'],relative
    for model in manifest['models']:
        assert sha(CURRENT/model['checkpoint'])==model['sha256'],model['id']
    assert len(manifest['models'])==14
    actual=list((CURRENT/'models').rglob('*.pt'))
    assert len(actual)==14
    datasets=0
    for entry in manifest['datasets']:
        rows,_=current_dataset(entry['experiment'],entry['mode'],entry['resource'])
        assert len(rows)==entry['records']
        datasets+=len(rows)
    assert datasets==3620
    _,sources,cells=main_panel(independent_ema=True)
    assert sum(s['rows'] for s in sources)==2160 and len(cells)==108
    from plot.plot_alibaba_energy_only import load_data
    from plot.plot_workload_sensitivity_options import load_verified
    ali,_,_=load_data(csv_source=True,independent_ema=True)
    sensitivity,_=load_verified(csv_source=True,independent_ema=True)
    assert len(ali)==12 and len(sensitivity)==40
    assert sensitivity['arrival','dqn_edge',.2]['esp_at_90']==.95
    before=ROOT/'paper/manuscript/main2.tex'
    text=before.read_text(encoding='utf-8')
    part=text.split(r'\label{tab:constraint_results}',1)[1].split(r'\bottomrule',1)[0]
    dd=None;checked=0;mismatches=[]
    resources={'S':'ss','M':'sm','L':'sl'};scales={'S':'small','M':'medium','L':'large'}
    for line in part.splitlines():
        m=re.match(r'^(Loose|Medium|Tight)? & (SS|SM|SL|MS|MM|ML|LS|LM|LL) & ',line)
        if not m:continue
        dd=m[1].lower() if m[1] else dd
        values=line.split(' & ')[2:]
        for i,mode in enumerate(('dqn_edge','dream_dqn','oo_actor_critic','d3qn_sac')):
            c=cells[resources[m[2][1]],scales[m[2][0]],dd,mode]
            value=re.sub(r'\\textbf\{([^}]+)\}',r'\1',values[3*i])
            if value!=f"{c['energy_mean_J']:.2f}":
                mismatches.append((m[2],dd,mode,value,f"{c['energy_mean_J']:.2f}"))
            checked+=1
    assert checked==108
    print(f'PASS: {len(manifest["files"])} copied source artifacts; 14 selected models; 3620 stored raw rows (3500 paper rows); current plot readers.')
    print(f'Manuscript check: {checked-len(mismatches)}/108 mean-energy cells match the curated current data.')
    if mismatches:
        print('WARNING: manuscript has been independently changed/restored; cleanup does not overwrite it.')
        print('Mismatching cells:',mismatches)
    print('RFR remains unverified: no raw experiment was invented or added.')


if __name__=='__main__':main()
