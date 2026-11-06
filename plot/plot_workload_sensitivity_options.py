"""Three layout previews from audited, unfiltered sensitivity results; no evaluation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_sm74_workflow_count as count
from tools import eval_sm74_arrival_rate as arrival
from tools.eval_sm74_ablation import read, rows, sha

OUT = ROOT/'docs/figures/workload_sensitivity_options'
MODES = ['dqn_edge', 'dream_dqn', 'oo_actor_critic', 'd3qn_sac']
COLORS = dict(zip(MODES, ['#F5AA83', '#BBDDB2', '#BAE0E3', '#A8C0E4']))
LABELS = {**count.LABELS, 'd3qn_sac':'D3QN+SAC (ours)'}
STYLES = dict(zip(MODES, ['--', '-.', (0, (5, 2, 1, 2)), '-']))
GRIDS = {'count':count.COUNTS, 'arrival':sorted(arrival.RATES)}
XLABELS = {'count':'Number of workflows', 'arrival':r'Arrival rate (s$^{-1}$)'}
TITLES = {'count':'Workflow count', 'arrival':'Arrival rate'}


def load_verified(csv_source=False, independent_ema=False):
    from plot.paper_csv_data import CURRENT, current_dataset
    if (CURRENT/'manifest.json').exists():
        data,sources={},{}
        for kind,module,experiment in [('count',count,'workflow_count'),('arrival',arrival,'arrival_rate')]:
            dimension='num_workflows' if kind=='count' else 'arrival_rate'
            for mode in MODES:
                records,files=current_dataset(experiment,mode)
                for source in files:
                    sources[str(Path(source['path']).relative_to(ROOT))]=source['sha256']
                for value in GRIDS[kind]:
                    sample=[r for r in records if r[dimension]==value]
                    assert sorted(r['eval_seed'] for r in sample)==list(range(742000,742020))
                    data[kind,mode,value]=dict(mode=mode,**{dimension:value},**module.summarize(sample))
        assert len(data)==40
        for mode in MODES:
            for field in ('energy_mean_J','energy_sample_sd_J','esp_at_90','reliability_attainment'):
                assert math.isclose(data['count',mode,50][field],data['arrival',mode,.03][field],rel_tol=1e-12)
        assert data['arrival','dqn_edge',.2]['esp_at_90']==.95
        assert data['arrival','dream_dqn',.2]['esp_at_90']==.9
        return data,sources
    data, sources = {}, {}
    for kind, module, base in [('count', count, count.OUT), ('arrival', arrival, arrival.OUT)]:
        plan = read(base/'frozen_plan.json')
        module.check_sources(plan)
        summary = read(base/'summary.json')
        assert summary['all_complete'] and len(summary['cells']) == 20
        dimension = 'num_workflows' if kind == 'count' else 'arrival_rate'
        seen = set()
        for cell in summary['cells']:
            mode, value = cell['mode'], cell[dimension]
            key = kind, mode, value
            assert mode in MODES and value in GRIDS[kind] and key not in seen
            seen.add(key)
            if independent_ema and mode == 'dqn_edge':
                from plot.paper_csv_data import independent_extra_rows, stats
                group = 'workflow_count' if kind == 'count' else 'arrival_rate'
                # One verified source file contains all five settings.
                if not any(k[0] == kind and k[1] == mode for k in data):
                    new_raw, new_source = independent_extra_rows(group)
                raw = [r for r in new_raw if r[dimension] == value]
                assert [r['eval_seed'] for r in raw] == list(range(742000,742020))
                sample_key = str(value) if kind == 'count' else arrival.tag(value)
                for row in raw:
                    sample = plan['samples'][sample_key][str(row['eval_seed'])]
                    for field in ('physical_sha256','deadlines_sha256','tasks','edges'):
                        if field in sample:
                            assert row[field] == sample[field]
                actual = module.summarize(raw)
                data[key] = {**cell, **actual}
                sources[str(Path(new_source['path']).relative_to(ROOT))] = new_source['sha256']
                continue
            folder = base/mode/(f'n{value}' if kind == 'count' else arrival.tag(value))
            path = folder/'comparison_rows.jsonl'
            raw = rows(path)
            if csv_source:
                from plot.paper_csv_data import read_csv
                path = path.with_suffix('.csv')
                raw = read_csv(path)
            module.validate(raw, mode, value, plan, complete=True)
            assert (folder/'completion.json').exists()
            actual = module.summarize(raw)
            for field, expected in actual.items():
                assert math.isclose(cell[field], expected, rel_tol=1e-12, abs_tol=1e-12), (key, field)
            assert cell['n'] == 20
            data[key] = {**cell, **actual}
            sources[str(path.relative_to(ROOT))] = sha(path)
        for path in (base/'summary.json', base/'frozen_plan.json'):
            sources[str(path.relative_to(ROOT))] = sha(path)
    assert len(data) == 40
    for mode in MODES:
        for field in ('energy_mean_J', 'energy_sample_sd_J', 'esp_at_90', 'reliability_attainment'):
            assert math.isclose(data['count',mode,50][field], data['arrival',mode,.03][field], rel_tol=1e-12)
    # The compact displays are valid only while these exact attainment patterns hold.
    for (kind, mode, value), cell in data.items():
        expected = .9 if kind == 'arrival' and value == .2 and mode in MODES[:2] else 1.
        if independent_ema and kind == 'arrival' and value == .2 and mode == 'dqn_edge':
            expected = .95
        assert math.isclose(cell['esp_at_90'], expected)
        assert cell['reliability_attainment'] == 1.
    return data, sources


def style():
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':11, 'axes.labelsize':12,
        'axes.titlesize':12, 'xtick.labelsize':10, 'ytick.labelsize':10,
        'legend.fontsize':11, 'axes.linewidth':.8, 'svg.fonttype':'none',
        'savefig.facecolor':'white', 'figure.facecolor':'white', 'axes.titlepad':12})


def clean_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out', length=3, width=.8)
    ax.grid(axis='y', color='#DFE3E8', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def xaxis(ax, kind):
    ax.set_xticks(GRIDS[kind])
    if kind == 'arrival':
        ax.set_xticklabels([f'{x:.2f}' for x in GRIDS[kind]])
        ax.set_xlim(-.002,.212)
    else:
        ax.set_xlim(38,262)
    ax.set_xlabel(XLABELS[kind])


def energy(ax, data, kind, letter, joules=False):
    divisor = 1 if joules else 1000
    for mode in MODES:
        points = [data[kind,mode,x] for x in GRIDS[kind]]
        values = [p['energy_mean_J']/divisor for p in points]
        sd = [p['energy_sample_sd_J']/divisor for p in points]
        ax.errorbar(GRIDS[kind], values, yerr=sd, fmt='none', ecolor='#555555',
                    elinewidth=.8, capsize=2.5, capthick=.8, alpha=.7, zorder=2)
        line, = ax.plot(GRIDS[kind], values, color=COLORS[mode], linestyle=STYLES[mode],
                        linewidth=2.5, zorder=3)
        line.set_path_effects([pe.Stroke(linewidth=3.15, foreground='#666666'), pe.Normal()])
    clean_axes(ax)
    xaxis(ax, kind)
    ax.set_ylabel('Total computation energy (J)' if joules else 'Total energy (kJ)')
    scale = 1000 if joules else 1
    ax.set_ylim(np.array((0,3600) if kind == 'count' else (300,850))*scale)
    ax.set_yticks(np.array([0,1000,2000,3000] if kind == 'count' else [300,400,500,600,700,800])*scale)
    if joules:
        power = 6 if kind == 'count' else 5
        ax.ticklabel_format(axis='y',style='sci',scilimits=(power,power),useMathText=True)
    ax.set_title(f'({letter}) {TITLES[kind]}', loc='left')


def legend(fig):
    handles = [Line2D([],[],color=COLORS[m],linewidth=3,linestyle=STYLES[m],
                     path_effects=[pe.Stroke(linewidth=3.65,foreground='#666666'),pe.Normal()],
                     label=LABELS[m]) for m in MODES]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.018), ncol=4,
               frameon=False, handlelength=3.1, columnspacing=2.4)


def save(fig, name):
    fig.canvas.draw()
    # Text must stay on the canvas; visible layout is additionally reviewed as PNG.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.width+1 and box.y1 <= bounds.height+1, item.get_text()
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'{name}.{suffix}', dpi=220)
    plt.close(fig)


def option_a(data, joules=False):
    fig = plt.figure(figsize=(12,7.1))
    grid = fig.add_gridspec(2,2, height_ratios=[3.1,1.55], left=.073, right=.977,
                           bottom=.12, top=.93, wspace=.25, hspace=.46)
    for i, kind in enumerate(GRIDS):
        main_ax = fig.add_subplot(grid[0,i])
        energy(main_ax, data, kind, 'ab'[i], joules=joules)
        if joules and kind == 'count':
            inset = main_ax.inset_axes([.08,.62,.45,.29])
            inset.set_facecolor('white')
            for mode in MODES:
                points = [data[kind,mode,x] for x in GRIDS[kind]]
                values = [p['energy_mean_J']/x for p,x in zip(points,GRIDS[kind])]
                sd = [p['energy_sample_sd_J']/x for p,x in zip(points,GRIDS[kind])]
                for p,value in zip(points,values):
                    assert math.isclose(value,p['energy_per_workflow_mean_J'],rel_tol=1e-12)
                inset.errorbar(GRIDS[kind],values,yerr=sd,fmt='none',ecolor='#777777',
                               elinewidth=.55,capsize=1.5,zorder=2)
                inset.plot(GRIDS[kind],values,color=COLORS[mode],linestyle=STYLES[mode],
                           linewidth=1.6,path_effects=[pe.Stroke(linewidth=2.1,foreground='#666666'),pe.Normal()])
            clean_axes(inset)
            inset.set_title('Energy per workflow (J/workflow)',fontsize=8.5,pad=5)
            inset.set_xticks([50,150,250])
            inset.set_yticks([8000,12000,16000])
            inset.set_ylim(6500,16500)
            inset.set_xlim(38,262)
            inset.ticklabel_format(axis='y',style='sci',scilimits=(3,3),useMathText=True)
            inset.tick_params(labelsize=7.5,length=2,pad=2)
            inset.yaxis.get_offset_text().set_fontsize(7.5)
    ax = fig.add_subplot(grid[1,:])
    ax.set_axis_off()
    ax.set_title('(c) Constraint attainment', loc='left', pad=7)
    headers = ['Method', 'ESP@90 (%)\nCount: 50-250', 'ESP@90 (%)\nRate: 0.01-0.10',
               'ESP@90 (%)\nRate: 0.20', 'Rel. (%)\nAll tested points']
    body = [[LABELS[m], '100', '100', f"{100*data['arrival',m,.2]['esp_at_90']:.0f}", '100'] for m in MODES]
    table = ax.table(cellText=body, colLabels=headers, colWidths=[.25,.19,.20,.17,.19],
                     cellLoc='center', bbox=[0,.02,1,.88])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('#DBDFE5')
        cell.set_linewidth(.7)
        if r == 0:
            cell.set_facecolor('#E0EAF6')
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor('#F7F9FC' if r == 4 else 'white')
            if c == 0:
                cell.set_facecolor(COLORS[MODES[r-1]])
            if c == 3 and r in (1,2):
                cell.set_facecolor('#FBE2D4')
            if r == 4:
                cell.get_text().set_fontweight('bold')
    legend(fig)
    save(fig, 'option_A_joules_with_unit_energy' if joules else 'option_A_energy_and_table')


def option_b(data):
    fig = plt.figure(figsize=(12,7.7))
    grid = fig.add_gridspec(2,2, height_ratios=[3,1.8], left=.125, right=.95,
                           bottom=.13, top=.94, wspace=.33, hspace=.60)
    for i, kind in enumerate(GRIDS):
        energy(fig.add_subplot(grid[0,i]), data, kind, 'ab'[i])
        ax = fig.add_subplot(grid[1,i])
        vals = np.array([[100*data[kind,m,x]['esp_at_90'] for x in GRIDS[kind]] for m in MODES])
        ax.imshow(vals, cmap=ListedColormap(['#F5AA83','#E0EAF6']),
                  norm=BoundaryNorm([85,95,105],2), aspect='auto', interpolation='nearest')
        ax.set_xticks(range(5), [str(x) if kind == 'count' else f'{x:.2f}' for x in GRIDS[kind]])
        ax.set_yticks(range(4), [LABELS[m] for m in MODES])
        ax.tick_params(length=0, pad=7, labelsize=9.7)
        for j in range(4):
            for k in range(5):
                ax.text(k,j,f'{vals[j,k]:.0f}%',ha='center',va='center',fontsize=11,
                        fontweight='bold' if j == 3 else 'normal',color='#252525')
        ax.set_xticks(np.arange(-.5,5,1), minor=True)
        ax.set_yticks(np.arange(-.5,4,1), minor=True)
        ax.grid(which='minor',color='white',linewidth=2)
        ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(XLABELS[kind],labelpad=8)
        ax.set_title(f'({"cd"[i]}) ESP@90 (%)',loc='left',pad=10)
        ax.text(1,1.10,'Rel. = 100% at all points',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=9.5,color='#555555')
    legend(fig)
    save(fig,'option_B_energy_and_heatmaps')


def option_c(data):
    fig, axes = plt.subplots(2,3,figsize=(13.2,7.5))
    fig.subplots_adjust(left=.065,right=.98,bottom=.15,top=.94,wspace=.32,hspace=.56)
    for row,kind in enumerate(GRIDS):
        energy(axes[row,0],data,kind,'ad'[row])
        for col,metric in enumerate(['esp_at_90','reliability_attainment'],start=1):
            ax = axes[row,col]
            for mode in MODES:
                ax.plot(GRIDS[kind],[100*data[kind,mode,x][metric] for x in GRIDS[kind]],
                        color=COLORS[mode],linestyle=STYLES[mode],linewidth=2.5,
                        path_effects=[pe.Stroke(linewidth=3.15,foreground='#666666'),pe.Normal()])
            clean_axes(ax)
            xaxis(ax,kind)
            ax.set_ylim(84,103)
            ax.set_yticks([85,90,95,100])
            ax.set_ylabel('ESP@90 (%)' if col == 1 else 'Rel. attainment (%)')
            ax.set_title(f'({"abcdef"[row*3+col]}) {TITLES[kind]}',loc='left')
            if metric == 'reliability_attainment' or kind == 'count':
                ax.text(.50,.59,'All methods: 100%',transform=ax.transAxes,ha='center',
                        color='#555555',fontsize=10.5)
            else:
                ax.annotate('DQN-Edge / DReaM: 90%',xy=(.2,90),xytext=(.17,92.7),
                            ha='right',fontsize=8.5,color='#555555')
    legend(fig)
    save(fig,'option_C_all_metrics_six_panels')


def bars_only(data, refined=False, paper_blue=False, single_column=False, independent_ema=False):
    colors = dict(zip(MODES,['#D9A182','#93B18A','#88B9BF','#5F86B5'])) if refined else COLORS
    if paper_blue:
        from plot.paper_csv_data import PALETTE
        colors = PALETTE
    labels = {**LABELS,'d3qn_sac':'CAHDRL'}
    if single_column:
        fig,axes = plt.subplots(1,2,figsize=(3.5,2.85))
        fig.subplots_adjust(left=.13,right=.99,bottom=.47,top=.92,wspace=.45)
        fig.text(.015,.695,'Total computation energy (J)',rotation=90,
                 ha='left',va='center',fontsize=8.5)
    else:
        fig,axes = plt.subplots(1,2,figsize=(12,5.2))
        fig.subplots_adjust(left=.075,right=.98,bottom=.32,top=.94,wspace=.27)
    positions = np.arange(5)
    width = .18
    panel_labels = []
    for i,kind in enumerate(GRIDS):
        ax = axes[i]
        for j,mode in enumerate(MODES):
            points = [data[kind,mode,x] for x in GRIDS[kind]]
            means = [p['energy_mean_J'] for p in points]
            sd = [p['energy_sample_sd_J'] for p in points]
            bars = ax.bar(positions+(j-1.5)*width,means,width=width,
                          yerr=sd,color=colors[mode],edgecolor='#505050',linewidth=.65,
                          error_kw=dict(ecolor='#555555',elinewidth=.85,capsize=1.3 if single_column else 2,capthick=.85),zorder=3)
            assert [b.get_height() for b in bars] == means
        clean_axes(ax)
        ax.set_xticks(positions,[str(x) if kind == 'count' else f'{x:.2f}' for x in GRIDS[kind]])
        ax.set_xlim(-.62,4.62)
        ax.set_xlabel(XLABELS[kind],labelpad=5 if single_column else 8,
                      fontsize=8.5 if single_column else 12)
        ax.set_ylabel('' if single_column else 'Total computation energy (J)',fontsize=12)
        if single_column:
            ax.tick_params(labelsize=8)
            ax.yaxis.get_offset_text().set_fontsize(8)
        ax.set_ylim(0,3.6e6 if kind == 'count' else 9e5)
        ax.set_yticks(np.array([0,1,2,3])*1e6 if kind == 'count' else np.array([0,2,4,6,8])*1e5)
        power = 6 if kind == 'count' else 5
        ax.ticklabel_format(axis='y',style='sci',scilimits=(power,power),useMathText=True)
        panel_labels.append(ax.text(.5,-.46 if single_column else -.24,f'({"ab"[i]}) {TITLES[kind]}',transform=ax.transAxes,
                                    ha='center',va='top',fontsize=8.5 if single_column else 12))
    handles = [Patch(facecolor=colors[m],edgecolor='#505050',linewidth=.65,label=labels[m]) for m in MODES]
    lg = fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=2 if single_column else 4,
                    frameon=False,handlelength=1.5 if single_column else 2,
                    columnspacing=1.6 if single_column else 2.6,
                    fontsize=8 if single_column else 11)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clearance = min(t.get_window_extent(renderer).y0 for t in panel_labels)-lg.get_window_extent(renderer).y1
    assert clearance/fig.dpi*72 >= 12, 'Legend too close to panel captions'
    name = 'energy_grouped_bars_J_blue' if paper_blue else ('energy_grouped_bars_J_refined_colors' if refined else 'energy_grouped_bars_J')
    if single_column:
        name = 'sensitivity_analysis_single_column'
    if independent_ema:
        name += '_independent_ema'
    if paper_blue:
        pdf = ROOT/'output/pdf'/f'{name}.pdf'
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf)
    save(fig,name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--joules',action='store_true',help='Only render revised A in J with a unit-energy inset')
    group.add_argument('--bars',action='store_true',help='Energy-only grouped bars in J; no table or inset')
    parser.add_argument('--paper-blue',action='store_true',help='Read original CSV; only render blue energy bars in J')
    parser.add_argument('--single-column',action='store_true',help='Keep horizontal panels at 3.5-inch column width with readable type')
    parser.add_argument('--independent-ema',action='store_true',help='Use completed per-rollout EMA-reset DQN-Edge CSVs')
    args = parser.parse_args()
    if args.independent_ema:
        args.paper_blue = True
    if args.single_column and not args.paper_blue:
        parser.error('--single-column requires --paper-blue to retain the current CSV sources and palette')
    data,sources = load_verified(csv_source=args.paper_blue,independent_ema=args.independent_ema)
    OUT.mkdir(parents=True,exist_ok=True)
    style()
    if args.paper_blue:
        bars_only(data,paper_blue=True,single_column=args.single_column,independent_ema=args.independent_ema)
    elif args.bars:
        bars_only(data)
        bars_only(data,refined=True)
    else:
        option_a(data,joules=args.joules)
        if not args.joules:
            option_b(data)
            option_c(data)
    for path,expected in sources.items():
        assert sha(ROOT/path) == expected, 'Source data changed during plotting'
    provenance = dict(source_sha256=sources, script_sha256=sha(Path(__file__)),
        complete_cells=40, records_per_cell=20, rendered_rollout_records=800,
        shared_reproduced_baseline_records=80, unfiltered=True, methods=MODES,
        colors=COLORS, error_bars='sample standard deviation of total energy across workload seeds',
        count_deadlines='recomputed per batch', arrival_deadlines='fixed relative budget from rate 0.03',
        original_sources_modified=False, manuscript_modified=False,
        constraint_axis_C='84 to 103 percent, explicitly labeled; no fake offsets',
        heatmap_B='two categorical observed levels 90 and 100; each cell annotated')
    if args.joules:
        provenance.update(units='J; inset J/workflow',count_axis_power=6,arrival_axis_power=5,
                          interpolation='straight segments between observed means; no fitted or smoothed curve',
                          unit_energy_inset='four methods; energy mean and sample SD divided by workflow count')
    if args.bars:
        provenance.update(units='J',count_axis_power=6,arrival_axis_power=5,
                          chart='grouped bars with zero-baseline energy axes',
                          xaxis='equally spaced discrete experimental settings; no interpolation',
                          table_in_image=False,inset=False,
                          display_method_name='CAHDRL; internal model key remains d3qn_sac',
                          panel_labels='centered below each subplot; >=12pt clearance above legend',
                          refined_color_preview=dict(zip(MODES,['#D9A182','#93B18A','#88B9BF','#5F86B5'])))
        provenance.pop('constraint_axis_C')
        provenance.pop('heatmap_B')
    provenance_name = 'provenance_bars.json' if args.bars else ('provenance_joules.json' if args.joules else 'provenance.json')
    if args.paper_blue:
        from plot.paper_csv_data import PALETTE
        provenance_name = 'provenance_blue.json'
        provenance.update(colors=PALETTE, units='J', actual_data_format='original CSV',
                          chart='grouped bars with zero-baseline energy axes',
                          xaxis='equally spaced discrete experimental settings; no interpolation')
    if args.single_column:
        provenance_name = 'provenance_single_column.json'
        provenance.update(layout='1 row, 2 columns',figure_size_inches=[3.5,2.85],
                          tick_font_pt=8,label_font_pt=8.5,legend_font_pt=8,
                          legend_columns=2,panel_caption_font_pt=8.5)
    if args.independent_ema:
        provenance_name = 'provenance_independent_ema.json'
        provenance['dqn_edge_ema_protocol'] = 'Restore checkpoint EMA before each realization; update within realization'
    (OUT/provenance_name).write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_cells=40, raw_rollout_records=800, output=str(OUT),
                         previews=['grouped bars in J'] if (args.bars or args.paper_blue) else (['A in J'] if args.joules else ['A','B','C']),
                         source_hashes_unchanged=True)))


if __name__ == '__main__':
    main()
