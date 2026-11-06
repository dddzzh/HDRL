"""Code-aligned vector overview: decisions, networks, and credit assignment."""
from pathlib import Path
import json
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Ellipse, Arc
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_hex

ROOT = Path(__file__).resolve().parents[1]
INK = "#303030"
PALETTE = ("#E0EAF6", "#A8C0E4", "#BBDDB2", "#C7C5C6", "#F5AA83", "#BAE0E3")
PALE, BLUE, GREEN, GREY, ORANGE, CYAN = PALETTE


def check_implementation():
    for name in ("v3_sm_74_legacy_train.json", "v3_sl_111_legacy_train.json"):
        cfg = json.loads((ROOT/"configs"/name).read_text(encoding="utf-8"))
        agent = cfg["agent"]
        assert agent["high_algorithm"] == "d3qn" and agent["low_algorithm"] == "sac"
        assert not agent["physical_low_reward"] and not agent["dvfs_aware_high_state"]
        assert agent["enforce_config_lagrangian"]
        assert agent["low_dual_attribution"] == "workflow_counterfactual"
        assert agent["high"]["nonnegative_deadline_q"]
        assert agent["high"]["deadline_head_cost_mode"] == "dense_miss_plus_episode_failure_credit"
        assert cfg["simulation"]["dvfs_action_floor"] == "energy_efficient"
    high = (ROOT/"src/D3QN/reliability_aware_high.py").read_text(encoding="utf-8")
    low = (ROOT/"src/sac/hierarchical_low_sac.py").read_text(encoding="utf-8")
    dueling = (ROOT/"src/D3QN/D3QN.py").read_text(encoding="utf-8")
    assert "reliability_q_target" in high and "F.smooth_l1_loss" in high
    assert "advantage.mean" in dueling
    assert all(token in low for token in ("target_critic_1", "target_critic_2", "actor_loss", "log_std"))


def main():
    check_implementation()
    plt.rcParams.update({"font.family":"Times New Roman", "font.size":8,
                         "mathtext.fontset":"stix", "pdf.fonttype":42,
                         "svg.fonttype":"none"})
    fig, ax = plt.subplots(figsize=(7.4, 5.55), dpi=300)
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,25.5), ylim=(0,19.0))
    ax.axis("off")
    labels=[]

    def text(x,y,value,size=8,weight="normal",ha="center"):
        a=ax.text(x,y,value,fontsize=size,color=INK,weight=weight,ha=ha,
                  va="center",linespacing=1.25,zorder=12)
        labels.append(a)
        return a

    def line(points,lw=.6,z=3):
        ax.plot(*zip(*points),color=INK,lw=lw,zorder=z)

    def arrow(points,dashed=False,head=6,lw=.75):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=head,
                                    edgecolor=INK,facecolor=INK,lw=lw,
                                    linestyle=(0,(3,2)) if dashed else "-",zorder=8))

    def box(x,y,w,h,fill="white",dashed=False,lw=.65,z=1):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.12",
                                   edgecolor=INK,facecolor=fill,lw=lw,
                                   linestyle=(0,(4,3)) if dashed else "-",zorder=z))

    def panel(x,y,w,h,title,fill):
        box(x,y,w,h,dashed=True)
        ax.add_patch(Rectangle((x+.03,y+h-.67),w-.06,.64,
                               facecolor=fill,edgecolor="none",zorder=2))
        text(x+w/2,y+h-.34,title,9.0,"bold")

    def node(x,y,r=.105,fill=BLUE):
        ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=INK,lw=.5,zorder=10))

    def nn(x,y,w=1.65,h=.66,fill=BLUE):
        layers=[]
        for j,n in enumerate((2,3,2)):
            layers.append([(x+j*w/2,y+h*(k/(n-1)-.5)) for k in range(n)])
        for a,b in zip(layers,layers[1:]):
            for p in a:
                for q in b:
                    line([p,q],.35)
        for layer in layers:
            for p in layer:
                node(*p,min(.105,h/5.3),fill)

    def clock(x,y,r=.29):
        node(x,y,r,ORANGE)
        line([(x,y+r*.65),(x,y),(x+r*.5,y-r*.15)],.75,11)

    def shield(x,y,s=.64):
        pts=[(x-.44*s,y+.43*s),(x,y+.55*s),(x+.44*s,y+.43*s),
             (x+.36*s,y-.12*s),(x,y-.58*s),(x-.36*s,y-.12*s)]
        ax.add_patch(Polygon(pts,facecolor=CYAN,edgecolor=INK,lw=.65,zorder=5))
        line([(x-.2*s,y),(x-.04*s,y-.16*s),(x+.24*s,y+.22*s)],.75,9)

    def energy(x,y,s=.65):
        pts=[(x+.1*s,y+.6*s),(x-.3*s,y-.06*s),(x+.02*s,y-.06*s),
             (x-.1*s,y-.6*s),(x+.34*s,y+.12*s),(x+.03*s,y+.12*s)]
        ax.add_patch(Polygon(pts,facecolor=GREEN,edgecolor=INK,lw=.65,zorder=5))

    def server(x,y,w=.75,h=1.10,fill=BLUE):
        box(x,y,w,h,fill,z=4)
        for z in (.25,.5,.75):
            line([(x+.12*w,y+z*h),(x+.70*w,y+z*h)],.65,5)
            node(x+.85*w,y+z*h,.026,INK)

    def replay(x,y,w=1.32,h=1.12,fill=GREY):
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=fill,edgecolor=INK,lw=.65,zorder=4))
        ax.add_patch(Ellipse((x,y-h/2),w,.33,facecolor=fill,edgecolor=INK,lw=.65,zorder=5))
        ax.add_patch(Ellipse((x,y+h/2),w,.33,facecolor=fill,edgecolor=INK,lw=.65,zorder=5))

    def dag(x,y,s=1):
        pts=[(x,y),(x+.8*s,y+.52*s),(x+.8*s,y-.52*s),
             (x+1.6*s,y+.52*s),(x+1.6*s,y-.52*s),(x+2.4*s,y)]
        for a,b in ((0,1),(0,2),(1,3),(1,4),(2,4),(3,5),(4,5)):
            arrow([pts[a],pts[b]],head=4,lw=.45)
        for j,p in enumerate(pts):
            node(*p,.16*s,(GREY,GREEN,PALE)[min(j//2,2)])

    def timeline(x,y,full=False):
        for j in range(2):
            widths=(.35,.47,.31) if full else (.56,.72,.44)
            cursor=x
            line([(x-.05,y-j*.38),(x+2.0,y-j*.38)],.4)
            for k,ww in enumerate(widths):
                ax.add_patch(Rectangle((cursor,y-j*.38-.11),ww,.24,
                                       facecolor=(BLUE,ORANGE,CYAN)[k],edgecolor=INK,lw=.45,zorder=5))
                cursor+=ww+.08

    def dueling(x,y,symbol,fill):
        box(x,y-.67,5.12,1.35,PALE)
        node(x+.27,y+.21,.09,fill); node(x+.27,y-.21,.09,fill)
        for yy in (y-.25,y,y+.25):
            node(x+1.14,yy,.09,fill)
            for iy in (y-.21,y+.21):
                line([(x+.27,iy),(x+1.14,yy)],.35)
        for yy in (y-.25,y,y+.25):
            line([(x+1.14,yy),(x+1.79,y+.34)],.35)
            line([(x+1.14,yy),(x+1.79,y-.34)],.35)
        box(x+1.79,y+.10,.73,.49,CYAN)
        box(x+1.79,y-.59,.73,.49,ORANGE)
        text(x+2.155,y+.345,r"$V$",8)
        text(x+2.155,y-.345,r"$A$",8)
        arrow([(x+2.52,y+.34),(x+3.20,y+.34),(x+3.20,y)],head=4,lw=.5)
        arrow([(x+2.52,y-.34),(x+3.20,y-.34),(x+3.20,y)],head=4,lw=.5)
        node(x+3.20,y,.20,"white")
        text(x+3.20,y,"+",8)
        arrow([(x+3.42,y),(x+4.0,y)],head=4,lw=.5)
        text(x+4.48,y,rf"$Q_{symbol}$",10)

    # Left: actual scheduling environment, not a learned task-ordering agent.
    panel(.35,5.92,5.50,12.18,"Workflow environment",PALE)
    text(3.10,16.90,"Arriving DAGs",8.3,"bold")
    dag(1.72,15.94,1.13)
    for cx,label,fill in ((1.25,"Done",GREY),(3.0,"Ready",GREEN),(4.84,"Waiting",PALE)):
        node(cx-.42,14.88,.10,fill)
        text(cx+.13,14.88,label,6.9)
    arrow([(3.1,14.58),(3.1,14.22)],head=5)
    box(.78,13.35,4.64,.87,GREEN)
    text(3.10,13.79,"Shared ready-task selection",7.7)
    arrow([(3.10,13.35),(3.10,12.94)],head=5)
    box(.78,11.69,4.64,1.25,PALE)
    text(3.10,12.61,"Scheduling state",8,"bold")
    text(3.10,12.12,"Task · queues · urgency · reliability",7.0)
    arrow([(5.42,12.34),(6.65,12.34)],head=6)
    text(6.05,12.66,r"$s_t^H$",9)
    text(3.10,11.16,"Heterogeneous server pool",8,"bold")
    # Cloud topology is illustrative; actual server counts remain experimental.
    ax.add_patch(Ellipse((3.80,9.68),2.40,1.46,facecolor=CYAN,edgecolor=INK,lw=.7,zorder=2))
    server(3.35,9.22,.62,.92,CYAN)
    server(4.20,9.22,.62,.92,CYAN)
    server(.95,9.10,.70,1.06,BLUE)
    server(1.95,9.10,.70,1.06,BLUE)
    line([(1.3,10.20),(1.3,10.63),(4.5,10.63),(4.5,10.20)],.6)
    text(1.81,8.73,"Edge",7.5)
    text(3.98,8.73,"Cloud",7.5)
    arrow([(.91,10.48),(.62,10.48),(.62,12.09),(.78,12.09)],head=5,lw=.6)
    box(.78,7.21,4.64,.85,CYAN)
    text(3.10,7.64,"Transfer + wait + execute",7.8)
    arrow([(3.10,7.21),(3.10,6.98)],head=5)
    energy(1.57,6.65,.46); clock(3.10,6.65,.21); shield(4.65,6.65,.43)
    text(1.98,6.65,r"$E_w$",8.4)
    text(3.60,6.65,r"$T_w$",8.4)
    text(5.05,6.65,r"$\mathcal{R}_w$",8.4)

    panel(.35,1.22,5.50,4.18,"Credit & dual adaptation",ORANGE)
    arrow([(3.1,5.92),(3.1,5.40)],True,head=5)
    text(3.10,4.40,"Workflow / service-failure credit",7.4)
    timeline(.93,3.88)
    timeline(.93,3.02,True)
    text(4.12,3.67,"Actual",7.6)
    text(4.12,2.81,"Full freq.",7.6)
    arrow([(3.11,3.71),(3.11,3.12)],True,head=5,lw=.55)
    text(3.1,2.23,"DVFS delay + queue externality",7.3)
    text(3.1,1.65,r"$\mu_D^H,\ \mu_R^H\qquad\mu_D^L,\ \mu_R^L$",10)

    # Middle: deployed upper/lower policies and their online decision relation.
    panel(6.65,8.75,6.10,9.35,"Upper D3QN",BLUE)
    text(9.70,16.99,"Three independent dueling networks",7.6)
    for yy,sym,fill in ((15.97,"E",GREEN),(14.37,"D",ORANGE),(12.77,"R",CYAN)):
        dueling(7.14,yy,sym,fill)
    text(9.70,11.69,r"$Q_k=V_k+A_k-\overline{A}_k$",9.5)
    box(7.12,9.87,5.16,1.24,GREEN)
    text(9.70,10.68,"Constrained action score",7.5,"bold")
    text(9.70,10.22,r"$Q_E-\mu_D^H[Q_D]_{+}\;-\;\mu_R^H Q_R$",8.7)
    arrow([(9.70,11.38),(9.70,11.11)],head=5)
    text(9.70,9.30,r"$m_i=\mathrm{arg\,max}_m\,S_H(s_t^H,m)$",8.5)
    arrow([(9.70,9.87),(9.70,9.58)],head=5)

    panel(6.65,1.22,6.10,6.85,"Lower SAC",GREEN)
    arrow([(9.70,8.75),(9.70,8.07)],head=6)
    text(10.35,8.43,r"$m_i$",9)
    box(7.14,6.76,5.12,.56,PALE)
    text(9.70,7.04,r"Selected-server state $s_t^L$",7.9)
    arrow([(9.70,6.76),(9.70,6.48)],head=5)
    box(7.14,4.67,5.12,1.81,GREEN)
    text(9.70,6.28,r"Stochastic actor $\pi_\theta$",8.1,"bold")
    nn(7.60,5.46,1.67,.63,GREEN)
    arrow([(9.46,5.63),(10.08,5.85)],head=4,lw=.5)
    arrow([(9.46,5.29),(10.08,5.10)],head=4,lw=.5)
    text(10.34,5.84,r"$\mu$",9)
    text(10.34,5.09,r"$\sigma$",9)
    arrow([(10.68,5.84),(11.10,5.84),(11.10,5.46)],head=4,lw=.5)
    arrow([(10.68,5.09),(11.10,5.09),(11.10,5.46)],head=4,lw=.5)
    text(11.63,5.46,r"$\rho_i^{\mathrm{prop}}$",9)
    box(7.14,3.16,5.12,1.05,ORANGE)
    text(9.70,3.96,"Energy-efficient action floor",7.2)
    text(9.70,3.42,r"$\rho_i\in[\rho_{m_i}^{\mathrm{ee}},1]$",9)
    arrow([(9.70,4.67),(9.70,4.21)],head=5)
    box(7.14,1.79,5.12,.89,CYAN)
    text(9.70,2.24,r"$a_t=(m_i,\rho_i),\quad f_i=\rho_iF_{m_i}$",9.4)
    arrow([(9.70,3.16),(9.70,2.68)],head=5)
    arrow([(7.14,2.24),(6.25,2.24),(6.25,7.64),(5.42,7.64)],head=6)
    # Shared feedback updates the state; it does not select the next ready task.
    arrow([(.78,7.64),(.17,7.64),(.17,18.46),(3.1,18.46),(3.1,18.10)],lw=.65)
    text(9.7,18.56,"Online scheduling and off-policy learning",10,"bold")

    # Right upper: independent target estimators, replay, and Double-Q learning.
    panel(13.55,8.75,11.55,9.35,"Upper-level learning",BLUE)
    box(14.04,11.26,4.58,5.53,PALE)
    text(16.33,16.44,"Target estimators",8.1,"bold")
    for yy,sym,fill in ((15.46,"E",GREEN),(13.91,"D",ORANGE),(12.36,"R",CYAN)):
        nn(14.52,yy,1.76,.68,fill)
        text(17.33,yy,rf"$\bar Q_{sym}$",10)
    arrow([(12.75,15.97),(14.04,15.97)],True,head=5)
    text(13.38,16.32,r"$\tau$",8.5)
    box(20.05,14.25,4.46,1.34,ORANGE)
    text(22.28,15.20,"Huber TD losses",8.1,"bold")
    text(22.28,14.63,r"$\mathcal{L}_E,\ \mathcal{L}_D,\ \mathcal{L}_R$",10)
    arrow([(12.75,17.20),(22.28,17.20),(22.28,15.59)],True,head=5)
    text(21.30,16.91,r"$Q_k(s_t^H,m_i)$",8.4)
    arrow([(20.05,14.92),(19.13,14.92),(19.13,17.07),(12.50,17.07),(12.50,15.97),(12.26,15.97)],True,head=5)
    text(20.14,16.18,r"Update $\theta_k$",7.2)
    box(20.05,11.48,4.46,1.59,CYAN)
    text(22.28,12.68,"Double-Q targets",8.0,"bold")
    text(22.28,12.15,"Online selection",7.3)
    text(22.28,11.75,"Target evaluation",7.3)
    arrow([(18.62,12.36),(20.05,12.36)],True,head=5)
    arrow([(22.28,13.07),(22.28,14.25)],True,head=5)
    arrow([(12.75,10.58),(19.48,10.58),(19.48,12.73),(20.05,12.73)],True,head=5)
    replay(16.33,9.94,1.53,1.09,BLUE)
    text(16.33,9.08,"Upper replay",7.5)
    text(21.86,9.57,r"$(s_t^H,m_i,r_E,c_D,c_R,s_{t+1}^H)$",8.0)
    arrow([(17.21,9.94),(22.28,9.94),(22.28,11.48)],True,head=5)
    arrow([(16.33,10.64),(16.33,11.26)],True,head=5)

    # Right lower: two ordinary soft-Q networks, not constraint-specific critics.
    panel(13.55,1.22,11.55,6.85,"Lower-level learning",GREEN)
    box(14.04,4.66,4.58,2.13,GREEN)
    text(16.33,6.50,"Twin soft-Q critics",8.2,"bold")
    for yy,j in ((5.97,1),(5.16,2)):
        nn(14.47,yy,1.72,.47,GREEN)
        text(17.27,yy,rf"$Q_{{\phi_{j}}}$",9.5)
    box(14.04,1.79,4.58,2.13,CYAN)
    text(16.33,3.57,"Target twin critics",7.9,"bold")
    for yy,j in ((2.99,1),(2.28,2)):
        nn(14.47,yy,1.72,.38,CYAN)
        text(17.27,yy,rf"$Q_{{\bar\phi_{j}}}$",9.5)
    arrow([(16.33,4.66),(16.33,3.92)],True,head=5)
    text(17.79,4.28,"Soft update",6.9)
    replay(22.28,6.13,1.68,1.01,GREEN)
    text(22.28,6.13,"Replay",7.5,"bold")
    text(22.28,5.23,r"$(s_t^L,\rho_i,r_L,s_{t+1}^L)$",8.5)
    arrow([(21.36,6.13),(18.62,6.13)],True,head=5)
    box(20.05,1.80,4.46,2.63,ORANGE)
    text(22.28,4.06,"Soft Bellman target",8,"bold")
    text(22.28,3.61,r"$r_L+\gamma(1-d_t)\,[\,$",8.6)
    text(22.28,3.04,r"$\min_j Q_{\bar\phi_j}-\alpha\log\pi_\theta\,]$",8.4)
    text(22.28,2.53,"Twin-critic TD losses",7.7)
    text(22.28,2.04,r"$\mathcal{L}_{\phi_1},\ \mathcal{L}_{\phi_2}$",9.8)
    arrow([(22.28,5.00),(22.28,4.43)],True,head=5)
    arrow([(18.62,2.99),(20.05,2.99)],True,head=5)
    arrow([(20.05,2.28),(19.35,2.28),(19.35,5.16),(18.62,5.16)],True,head=5)
    arrow([(14.04,5.57),(12.28,5.57)],True,head=5)
    text(13.12,5.97,"Policy",6.8)
    text(13.12,5.18,"gradient",6.8)
    box(14.04,6.87,10.47,.43,PALE)
    text(19.275,7.09,"Entropy-regularized policy update",7.7)
    arrow([(12.75,7.05),(13.15,7.05),(13.15,7.09),(14.04,7.09)],True,head=5)

    # Credit streams are routed outside modules and away from the online path.
    arrow([(5.85,4.40),(6.06,4.40),(6.06,8.40),(16.33,8.40),(16.33,9.23)],True,head=5,lw=.65)
    arrow([(5.85,2.90),(5.99,2.90),(5.99,.96),(24.78,.96),(24.78,6.13),(23.20,6.13)],True,head=5,lw=.65)
    arrow([(5.42,1.65),(5.71,1.65),(5.71,.73),(25.27,.73),(25.27,3.62),(24.51,3.62)],True,head=5,lw=.6)
    # Multiplier adaptation enters action scoring and lower reward construction.
    arrow([(5.71,1.65),(5.71,8.25),(7.02,8.25),(7.02,9.97),(7.12,9.97)],True,head=4,lw=.6)
    arrow([(12.26,6.36),(13.25,6.36),(13.25,6.15),(14.04,6.15)],head=4,lw=.6)
    arrow([(7.3,.38),(8.15,.38)],head=5)
    text(8.39,.38,"Online flow",7.4,ha="left")
    arrow([(14.12,.38),(14.97,.38)],True,head=5)
    text(15.22,.38,"Replay / parameter updates",7.4,ha="left")

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    for a in labels:
        b=a.get_window_extent(renderer)
        assert canvas.contains(b.x0,b.y0) and canvas.contains(b.x1,b.y1),a.get_text()
    for i,a in enumerate(labels):
        aa=a.get_window_extent(renderer)
        for b in labels[i+1:]:
            bb=b.get_window_extent(renderer)
            assert min(aa.x1,bb.x1)-max(aa.x0,bb.x0)<=1 or min(aa.y1,bb.y1)-max(aa.y0,bb.y0)<=1,(a.get_text(),b.get_text())
    allowed={c.lower() for c in PALETTE}|{"#ffffff","#303030","#000000"}
    for p in ax.patches:
        if p.get_fill() and p.get_facecolor()[3]>0:
            assert to_hex(p.get_facecolor()).lower() in allowed
    asset=ROOT/"docs/figures/method_framework"
    pdf=ROOT/"output/pdf"
    backup=asset/"previous_icon_layout"
    backup.mkdir(parents=True,exist_ok=True)
    for p in [pdf/"d3qn_sac_framework.pdf"]+[asset/f"d3qn_sac_framework.{s}" for s in ("svg","png")]:
        if p.exists() and not (backup/p.name).exists():
            shutil.copy2(p,backup/p.name)
    fig.savefig(pdf/"d3qn_sac_framework.pdf",facecolor="white")
    for s in ("svg","png"):
        fig.savefig(asset/f"d3qn_sac_framework.{s}",dpi=400,facecolor="white")
    plt.close(fig)
    print("Detailed framework: implementation, palette, text bounds and overlap checks passed.")


if __name__=="__main__":
    main()
