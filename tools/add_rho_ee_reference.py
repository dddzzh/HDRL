"""Add an analytic per-server rho_ee reference from existing frozen traces."""
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_reliability import OUT, SEEDS, read, rows, sha, relative, write, check_sources, describe, fixed_placement_scan

DEST = OUT / "rho_ee_reference"


def evaluate_saved_seed(data, servers):
    tasks = data["tasks"]
    for task in tasks:
        server = servers[task["server_id"]]
        if not math.isclose(task["rho_energy_efficient"], server["rho_energy_efficient"], abs_tol=1e-12):
            raise ValueError("Saved and reconstructed server floors differ")
        if float(server["rho_energy_efficient"]) < .2:
            raise ValueError("Reference must also respect the frozen SAC rho_min=0.2")
    requirements = {w["workflow_id"]:w["requirement"] for w in data["workflows"]}
    # Zero is a request to the existing floor-clamping calculation, not executed rho=0.
    result = fixed_placement_scan(tasks, servers, requirements, [0.0])[0]
    del result["requested_rho"]
    del result["floor_binding_fraction"]
    result.update(seed=data["seed"], policy="per_server_rho_ee", actual_rho=describe(
        servers[t["server_id"]]["rho_energy_efficient"] for t in tasks))
    # Every rho_ee in these saved tasks is <=0.5; the reference cannot be safer than rho=0.5.
    half = next(s for s in data["fixed_placement_scan"] if s["requested_rho"] == .5)
    old = {w["workflow_id"]:w["reliability"] for w in half["workflows"]}
    if result["actual_rho"]["max"] <= .5:
        if any(w["reliability"] > old[w["workflow_id"]]+1e-12 for w in result["workflows"]):
            raise ValueError("Frequency-risk monotonicity violated")
    return result


def make_figure(detail, ee):
    import matplotlib.pyplot as plt
    import numpy as np
    from plot.plot_reliability_diagnostic import style, save, BLUE, ORANGE, GREEN, CYAN, INK
    style()
    actual = [w["reliability"] for d in detail for w in d["workflows"]]
    frequency = [t["rho"] for d in detail for t in d["tasks"]]
    policies = ["Full\n(actual)", r"$\rho_{ee,k}$"+"\nper-server"] + [f"{v:.1f}" for v in [.5,.6,.7,.8,.9,1.]]
    values = [actual, [w["reliability"] for r in ee for w in r["workflows"]]]
    for v in [.5,.6,.7,.8,.9,1.]:
        values.append([w["reliability"] for d in detail for s in d["fixed_placement_scan"]
                       if s["requested_rho"] == v for w in s["workflows"]])
    colors = [BLUE, ORANGE] + [GREEN]*6
    fig, axes = plt.subplots(2,2,figsize=(10.5,6.5))
    ax=axes[0,0]
    ax.hist(frequency,bins=np.linspace(.65,1.,36),weights=np.full(len(frequency),100/len(frequency)),
            color=BLUE,edgecolor="white",lw=.4)
    ax.axvline(np.mean(frequency),color=INK,ls="--",lw=1)
    ax.set(xlabel=r"Executed frequency ratio $\rho$",ylabel="Tasks per bin (%)",xlim=(.65,1.005),
           title="(a) Actual Full frequency choices")
    ax.text(.02,.93,f"Mean: {np.mean(frequency):.4f}\nMinimum: {min(frequency):.4f}\nServer floors: 0.372-0.498",transform=ax.transAxes,va="top",fontsize=9)
    ax=axes[0,1]
    r=np.sort(actual)
    ax.step(r,np.arange(1,len(r)+1)/len(r),where="post",color="#6383B2",lw=1.5)
    ax.set(xlabel="Workflow reliability R",ylabel="Empirical cumulative fraction",xlim=(.99,1.0005),
           title="(b) Actual Full reliability (zoomed)")
    ax.text(.02,.96,f"Minimum R: {min(r):.5f}\nRequirement: 0.9\n(below displayed range)",transform=ax.transAxes,va="top",fontsize=9)
    ax=axes[1,0]
    boxes=ax.boxplot(values,patch_artist=True,showfliers=False,whis=(0,100),widths=.58,
                     medianprops=dict(color=INK,lw=1.1),whiskerprops=dict(color=INK,lw=.8),capprops=dict(color=INK,lw=.8))
    for box,color in zip(boxes["boxes"],colors):
        box.set(facecolor=color,edgecolor=INK,linewidth=.8)
    ax.axhline(.9,color=INK,ls="--",lw=.9)
    ax.set(xticks=range(1,len(policies)+1),xticklabels=policies,ylim=(-.02,1.05),ylabel="Workflow reliability R",
           xlabel="Full vs fixed-placement frequency references",title="(c) Reliability under each frequency setting")
    ax.text(.98,.915,r"$R_{req}=0.9$",transform=ax.get_yaxis_transform(),ha="right",va="bottom",fontsize=9)
    ax=axes[1,1]
    attainment=[100*np.mean(np.array(v)>=.9) for v in values]
    ax.bar(range(len(values)),attainment,color=colors,edgecolor=INK,linewidth=.6,width=.62)
    for i,v in enumerate(attainment):
        ax.text(i,v+2,f"{v:g}%",ha="center",fontsize=8)
    ax.set(xticks=range(len(policies)),xticklabels=policies,ylim=(0,112),ylabel=r"Workflows satisfying $R\geq0.9$ (%)",
           xlabel="Full vs fixed-placement frequency references",title="(d) Workflows meeting the reliability requirement")
    for ax in axes.ravel():
        ax.grid(axis="y",color="#E2E4E5",lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x",labelsize=8)
    fig.tight_layout(w_pad=2,h_pad=2)
    fig.text(.5,-.005,"74 servers; 20 original seeds; 1,000 workflows. Boxes: median/IQR, whiskers: min-max. Analytic references: fixed placement, no rescheduling or deadline claim.",ha="center",fontsize=8)
    save(fig,"reliability_with_rho_ee")


def main():
    plan=read(OUT/"frozen_plan.json")
    check_sources(plan)
    completion=read(OUT/"detailed_replay/completion.json")
    if completion["seeds"]!=20 or completion["workflows"]!=1000:
        raise ValueError("Detailed source is incomplete")
    paths=sorted((OUT/"detailed_replay").glob("seed_*.json"))
    before={relative(p):sha(p) for p in paths}
    detail=[read(p) for p in paths]
    if [d["seed"] for d in detail]!=SEEDS:
        raise ValueError("Saved seed order differs")
    from tools.hierarchical_common import build_resources
    with redirect_stdout(io.StringIO()):
        resources=build_resources(read(plan["run"]["config"]),42)
    servers={int(s["id"]):s for s in resources["servers"]}
    ee=[evaluate_saved_seed(d,servers) for d in detail]
    DEST.mkdir(exist_ok=True)
    write(DEST/"per_seed.json",ee)
    workflow=[w for r in ee for w in r["workflows"]]
    summary=dict(policy="per_server_rho_ee",seeds=20,workflows=len(workflow),
                 reliability=describe(w["reliability"] for w in workflow),
                 workflow_attainment=statistics.mean(w["reliability"]>=.9 for w in workflow),
                 actual_rho_min=min(r["actual_rho"]["min"] for r in ee),
                 actual_rho_max=max(r["actual_rho"]["max"] for r in ee),
                 compute_energy_mean_j=statistics.mean(r["total_compute_energy_j"] for r in ee),
                 scope="analytic_fixed_placement_not_new_rollout_or_ablation", source_sha256=before,
                 tool_sha256=sha(__file__),source_traces_unchanged=True)
    make_figure(detail,ee)
    check_sources(plan)
    if before!={relative(p):sha(p) for p in paths}:
        raise ValueError("Detailed traces changed")
    write(DEST/"summary.json",summary)
    lines=["# rho_ee参照补充（仅后处理，无训练或环境重跑）", "",
           "沿用原74台模型Large/Tight的20个未筛选种子，固定实际放置和工作量，每任务取其服务器的rho_ee。", "",
           f"- 实际rho范围：{summary['actual_rho_min']:.8f}–{summary['actual_rho_max']:.8f}。",
           f"- 工作流最低R：{summary['reliability']['min']:.8f}；中位R：{summary['reliability']['median']:.8f}。",
           f"- 工作流可靠性达标比例：{summary['workflow_attainment']:.2%}（R>=0.9）。", "",
           "该参照只回答能效下界是否自动保证可靠性，不评估Deadline，也不证明可靠性训练模块的独立收益。",
           "原始20份轨迹及冻结来源SHA核对不变。新图保存在../figures/reliability_with_rho_ee.png和同名PDF；旧图保留。"]
    (DEST/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print({k:v for k,v in summary.items() if k!='source_sha256'},flush=True)


if __name__=="__main__":
    main()
