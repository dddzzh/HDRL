"""Diagnostic plots, separate from the paper's original performance figures."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.diagnose_reliability import OUT, read, check_sources

BLUE, ORANGE, GREEN, CYAN = "#A8C0E4", "#F5AA83", "#BBDDB2", "#BAE0E3"
INK = "#303030"


def style():
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"],
                         "font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.edgecolor":INK, "axes.labelcolor":INK, "savefig.facecolor":"white",
                         "pdf.fonttype":42, "svg.fonttype":"none"})


def save(fig, name):
    directory=OUT/"figures"
    directory.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(directory/f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def training_plot(audit):
    groups=[t for t in audit["training"] if t["scope"]=="through_selected_checkpoint" and t["episodes"]]
    fig, axes=plt.subplots(len(groups), 2, figsize=(10, 2.25*len(groups)), squeeze=False)
    for (left,right), group in zip(axes, groups):
        trace=group["trace"]
        x=[r.get("global_episode", r["episode"]) for r in trace]
        left.plot(x, [r["reliability_success_rate"] for r in trace], color="#548580", lw=.9)
        left.axhline(1, color=INK, ls=":", lw=.8)
        left.set_ylim(-.03,1.06)
        left.set_ylabel("Workflow attainment")
        left.set_title(group["run"].replace("_d3qn_sac", "")+": training exploration", fontsize=11)
        right.plot(x, [r.get("low_reliability_lambda",np.nan) for r in trace], color="#6383B2", lw=1.2)
        right.set_ylabel(r"Lower reliability multiplier $\lambda_R$")
        other=right.twinx()
        other.plot(x, [r.get("low_reliability_lagrangian_penalty_mean",np.nan) for r in trace], color=ORANGE, alpha=.8, lw=.8)
        other.set_ylabel("SAC replay penalty", color="#B06C4C")
        other.spines["right"].set_visible(True)
        right.set_title("Multiplier (blue); replay penalty (orange)",fontsize=10)
        for ax in [left,right]:
            ax.set_xlabel("Global training episode")
            ax.grid(axis="y",color="#E2E4E5",lw=.6)
    fig.tight_layout(h_pad=1.5, w_pad=2.2)
    save(fig,"training_reliability_activation")


def detail_plot(detail):
    workflows=[w for row in detail for w in row["workflows"]]
    r=np.sort([w["reliability"] for w in workflows])
    fig, matrix=plt.subplots(2,2,figsize=(9.2,6.0))
    axes=matrix.ravel()
    axes[0].step(r,np.arange(1,len(r)+1)/len(r),where="post",color="#6383B2",lw=1.5)
    axes[0].axvline(.9,color=INK,ls="--",lw=.9,label=r"Requirement $R_{req}=0.9$")
    axes[0].set(xlabel="Workflow reliability R",ylabel="Empirical cumulative fraction",xlim=(.895,1.001),title=f"(a) Actual Full: {len(workflows)} workflows")
    axes[0].legend(frameon=False,loc="upper left",fontsize=8)
    scans=[row["fixed_placement_scan"] for row in detail]
    grid=np.array([s["requested_rho"] for s in scans[0]])
    energy=np.array([[s["total_compute_energy_j"]/1000 for s in one] for one in scans])
    minimum=np.array([[s["reliability"]["min"] for s in one] for one in scans])
    attainment=np.array([[s["workflow_attainment"] for s in one] for one in scans])
    axes[1].plot(grid,energy.mean(axis=0),"s-",color="#BA7956",mfc=ORANGE,ms=4,lw=1)
    axes[1].fill_between(grid,energy.min(axis=0),energy.max(axis=0),color=ORANGE,alpha=.22)
    axes[1].set(xlabel=r"Requested frequency ratio $\rho$",ylabel="Compute energy per batch (kJ)",title="(b) Fixed-placement frequency scan")
    axes[2].plot(grid,minimum.min(axis=0),"s-",color="#69965E",mfc=GREEN,lw=1,ms=4,label="Minimum workflow R")
    medians=np.array([[s["reliability"]["median"] for s in one] for one in scans])
    axes[2].plot(grid,medians.mean(axis=0),"^-",color="#548580",mfc=CYAN,lw=1,ms=4,label="Mean of within-batch medians")
    axes[2].axhline(.9,color=INK,ls="--",lw=.8,label="Reliability requirement")
    axes[2].set(xlabel=r"Requested frequency ratio $\rho$",ylabel="Workflow reliability R",title="(c) Reliability value",ylim=(-.03,1.03))
    axes[2].legend(frameon=False,fontsize=7,loc="lower right")
    axes[3].plot(grid,100*attainment.mean(axis=0),"s-",color="#6383B2",mfc=BLUE,lw=1,ms=4)
    axes[3].fill_between(grid,100*attainment.min(axis=0),100*attainment.max(axis=0),color=BLUE,alpha=.22)
    axes[3].set(xlabel=r"Requested frequency ratio $\rho$",ylabel="Workflows satisfying R >= 0.9 (%)",title="(d) Reliability attainment rate",ylim=(-3,103))
    for ax in axes:
        ax.grid(axis="y",color="#E2E4E5",lw=.6)
    for ax in axes[1:]:
        ax.set_xticks(sorted(grid))
    fig.tight_layout(w_pad=2, h_pad=1.8)
    fig.text(.5,-.015,"Original fixed seeds; original frequency floor retained. Shading: seed range, not CI. Fixed-placement scan: no rescheduling or deadline claim.",ha="center",fontsize=8)
    save(fig,"workflow_margin_and_fixed_placement_scan")


def frequency_plot(detail):
    rho=np.array([t["rho"] for d in detail for t in d["tasks"]])
    fig, ax=plt.subplots(figsize=(6.4,3.1))
    ax.hist(rho,bins=np.linspace(.65,1.,36),weights=np.full(len(rho),100/len(rho)),
            color=BLUE,edgecolor="white",linewidth=.5)
    ax.axvline(rho.mean(),color=INK,ls="--",lw=1,label=f"Task-weighted mean: {rho.mean():.4f}")
    ax.set(xlabel=r"Actual SAC frequency ratio $\rho$",ylabel="Tasks per bin (%)",
           title=f"Actual frequency distribution: {len(detail)} seeds, {len(rho):,} tasks",xlim=(.65,1.005))
    ax.legend(frameon=False,fontsize=9,loc="upper left")
    ax.text(.02,.74,f"Below 0.9: {np.mean(rho<.9):.2%}\nBelow 0.8: {np.mean(rho<.8):.2%}",transform=ax.transAxes,fontsize=9)
    ax.grid(axis="y",color="#E2E4E5",lw=.6)
    fig.tight_layout()
    fig.text(.5,-.02,"Descriptive task distribution, not independent training repeats; original unfiltered seeds.",ha="center",fontsize=8)
    save(fig,"actual_sac_frequency_distribution")


if __name__=="__main__":
    check_sources(read(OUT/"frozen_plan.json"))
    style()
    training_plot(read(OUT/"existing_evidence.json"))
    detail=[read(p) for p in sorted((OUT/"detailed_replay").glob("seed_*.json"))]
    if detail:
        detail_plot(detail)
        frequency_plot(detail)
    print(f"[reliability-figures] detailed_seeds={len(detail)}")
