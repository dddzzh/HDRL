"""New vector system overview; independent of the prior paper's artwork."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, PathPatch, Arc
from matplotlib.colors import to_hex
from draw_method_framework import PALETTE, INK, PALE, BLUE, GREEN, GREY, ORANGE, CYAN, check_implementation

ROOT=Path(__file__).resolve().parents[1]


def main():
    check_implementation()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.3,
                         'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.65,3.65),dpi=300)
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,25.2),ylim=(0,12))
    ax.axis('off')
    labels=[]

    def text(x,y,s,size=8,weight='normal',ha='center'):
        item=ax.text(x,y,s,ha=ha,va='center',fontsize=size,weight=weight,color=INK,zorder=15)
        labels.append(item)
        return item

    def line(points,color=INK,lw=.7,dashed=False,z=4):
        ax.plot(*zip(*points),color=color,lw=lw,linestyle=(0,(3.5,2.8)) if dashed else '-',zorder=z)

    def arrow(points,dashed=False,both=False,color=INK,lw=.85):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='<->' if both else '-|>',
                     edgecolor=color,facecolor=color,mutation_scale=7,lw=lw,
                     linestyle=(0,(3.5,2.8)) if dashed else '-',zorder=7))

    def box(x,y,w,h,fill='white',edge=GREY,lw=.7,rounding=.16,z=2):
        item=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',
                           facecolor=fill,edgecolor=edge,lw=lw,zorder=z)
        ax.add_patch(item)
        return item

    def node(x,y,r=.12,fill=BLUE):
        ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=INK,lw=.65,zorder=10))

    def device(x,y):
        box(x,y,1.17,.84,PALE,INK,.7,.07)
        line([(x+.58,y),(x+.58,y-.23)],lw=.8)
        line([(x+.24,y-.23),(x+.93,y-.23)],lw=.8)
        box(x+1.39,y-.18,.45,.87,'white',INK,.7,.07)
        ax.add_patch(Rectangle((x+1.46,y-.02),.31,.56,facecolor=CYAN,edgecolor='none',zorder=4))
        node(x+1.62,y-.09,.025,INK)
        for radius in (.42,.68):
            ax.add_patch(Arc((x+2.05,y+.5),radius,radius,theta1=-40,theta2=40,color=GREY,lw=.65,zorder=6))

    def dag(x,y,s=1):
        points=[(x,y),(x+.65*s,y+.40*s),(x+.65*s,y-.40*s),
                (x+1.4*s,y+.40*s),(x+1.4*s,y-.40*s),(x+2.05*s,y)]
        for a,b in ((0,1),(0,2),(1,3),(1,4),(2,4),(3,5),(4,5)):
            arrow([points[a],points[b]],lw=.5)
        for j,p in enumerate(points):
            node(*p,.105*s,(BLUE,CYAN,GREEN)[j%3])

    def rack(x,y,w=.84,h=1.5,fill=GREEN):
        depth=.21
        ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                            facecolor=PALE,edgecolor=INK,lw=.55,zorder=7))
        ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                            facecolor=GREY,edgecolor=INK,lw=.55,zorder=7))
        box(x,y,w,h,fill,INK,.6,.05,8)
        for j in range(3):
            yy=y+.27+j*(h-.42)/3
            line([(x+.13,yy),(x+w-.27,yy)],lw=.65,z=9)
            node(x+w-.14,yy,.025,INK)

    def cloud(x,y):
        pts=[(x,y),(x-.65,y+.1),(x-.65,y+1.1),(x+.05,y+1.15),
             (x+.02,y+2),(x+1.25,y+2.35),(x+1.7,y+1.75),
             (x+2.25,y+2.25),(x+3.4,y+1.9),(x+3.2,y+1.12),
             (x+4.2,y+1.1),(x+4.25,y+.1),(x+3.55,y), (x,y)]
        codes=[MplPath.MOVETO]+[MplPath.CURVE4]*12+[MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(pts,codes),facecolor=PALE,edgecolor=BLUE,lw=.9,zorder=2))

    def clock(x,y):
        node(x,y,.20,ORANGE)
        line([(x,y+.12),(x,y),(x+.09,y-.05)],lw=.6,z=12)

    def shield(x,y):
        ax.add_patch(Polygon([(x-.18,y+.15),(x,y+.21),(x+.18,y+.15),
                            (x+.14,y-.10),(x,y-.23),(x-.14,y-.10)],
                            facecolor=CYAN,edgecolor=INK,lw=.6,zorder=8))
        line([(x-.09,y),(x-.01,y-.08),(x+.10,y+.07)],lw=.65,z=12)

    def dial(x,y):
        ax.add_patch(Arc((x,y),1.05,1.05,theta1=20,theta2=160,color=INK,lw=.8,zorder=9))
        for angle in (25,65,105,155):
            rad=math.radians(angle)
            line([(x+.43*math.cos(rad),y+.43*math.sin(rad)),
                  (x+.52*math.cos(rad),y+.52*math.sin(rad))],lw=.55,z=10)
        line([(x,y),(x+.31,y+.24)],lw=1,z=10)
        node(x,y,.045,INK)

    # Three editorial zones, without the previous stacked dashed containers.
    text(.58,11.33,'Dynamic workflows',9.7,'bold',ha='left')
    line([(.58,10.88),(5.82,10.88)],BLUE,1.5)
    text(7.22,11.33,'Heterogeneous edge-cloud execution',9.7,'bold',ha='left')
    line([(7.22,10.88),(24.54,10.88)],CYAN,1.5)
    line([(6.42,.75),(6.42,11.2)],GREY,.55)

    # Workflow input: devices, arriving graph cards and two workflow constraints.
    device(1.78,9.3)
    text(3.02,8.76,'Workflow sources',7.8)
    arrow([(3.05,8.37),(3.05,7.91)],lw=.8)
    box(1.30,5.98,3.83,1.68,PALE,BLUE,.65)
    box(1.04,5.74,3.83,1.68,'white',BLUE,.65)
    box(.78,5.50,3.83,1.68,'white',INK,.75)
    dag(1.25,6.36,1.42)
    text(3.0,4.98,r'$G_k=(\mathcal{N}_k,\mathcal{E}_k,D_k,\Theta_k)$',9)
    clock(.93,4.20)
    text(1.40,4.20,r'Deadline $D_k$',7.8,ha='left')
    shield(3.60,4.20)
    text(4.04,4.20,r'Reliability $\Theta_k$',7.8,ha='left')
    text(3.01,3.51,r'Online arrivals at $a_k$',8.0)
    # The same graph stream feeds the actual shared dispatcher.
    arrow([(3.0,2.99),(3.0,2.27),(7.24,2.27)],lw=.9)
    text(4.84,2.62,'Ready tasks',7.7)

    # Edge plane and server-local waiting queues.
    text(11.40,10.10,'Edge servers',8.7,'bold')
    ax.add_patch(Polygon([(7.5,7.2),(14.83,7.2),(16.7,8.15),(9.37,8.15)],
                        facecolor=PALE,edgecolor=BLUE,lw=.65,zorder=1))
    server_positions=[(8.56,8.0,1.25),(11.13,8.0,1.63),(13.70,8.0,1.40)]
    for j,(x,y,h) in enumerate(server_positions):
        rack(x,y,h=h,fill=(GREEN,CYAN,GREEN)[j])
        for q in range(3):
            ax.add_patch(Rectangle((x-.04+q*.31,7.47),.25,.22,
                         facecolor=(ORANGE,BLUE,GREY)[(q+j)%3],edgecolor=INK,lw=.4,zorder=9))
    text(11.8,6.70,'Local queues and DVFS-enabled processors',7.6)

    # Cloud resource class is visually different but uses the same execution model.
    text(21.56,10.10,'Cloud servers',8.7,'bold')
    cloud(19.86,7.43)
    rack(20.18,7.83,w=.91,h=1.29,fill=BLUE)
    rack(22.12,7.83,w=.91,h=1.52,fill=BLUE)
    arrow([(16.15,8.25),(19.18,8.25)],both=True,color=INK,lw=.9)
    text(17.67,8.77,'Dependency',7.2)
    text(17.67,8.38,'data',7.2)

    # One compact online controller; network optimization remains in the method figure.
    text(7.23,5.35,'Hierarchical scheduling',9.5,'bold',ha='left')
    line([(7.23,4.94),(24.52,4.94)],GREY,.55)
    box(7.25,1.30,3.89,2.64,PALE,BLUE,.8)
    text(9.195,3.48,'Shared dispatcher',8.3,'bold')
    for j in range(4):
        ax.add_patch(Rectangle((7.98+j*.58,2.58),.43,.44,facecolor=(BLUE,CYAN,GREEN,GREY)[j],
                              edgecolor=INK,lw=.5,zorder=6))
    text(9.195,1.88,'Smallest workload first',7.2)
    box(12.05,1.30,5.15,2.64,'white',BLUE,.95)
    ax.add_patch(Rectangle((12.06,3.23),5.13,.69,facecolor=BLUE,edgecolor='none',zorder=3))
    text(14.625,3.55,'Upper D3QN',9,'bold')
    for j,(symbol,color) in enumerate(((r'$Q_E$',GREEN),(r'$Q_D$',ORANGE),(r'$Q_R$',CYAN))):
        box(12.51+j*1.42,2.33,1.12,.48,color,INK,.45,.07,5)
        text(13.07+j*1.42,2.57,symbol,8.0)
    text(14.625,1.79,r'Server selection $m_i$',8.2)
    box(18.10,1.30,5.15,2.64,'white',GREEN,.95)
    ax.add_patch(Rectangle((18.11,3.23),5.13,.69,facecolor=GREEN,edgecolor='none',zorder=3))
    text(20.675,3.55,'Lower SAC',9,'bold')
    dial(19.21,2.24)
    text(21.36,2.50,r'DVFS ratio $\rho_i$',8.0)
    text(21.36,1.86,'Frequency control',7.5)
    arrow([(11.14,2.60),(12.05,2.60)],lw=.9)
    arrow([(17.20,2.60),(18.10,2.60)],lw=.9)
    # Control output and resource-state feedback use separate short outer routes.
    arrow([(23.25,2.60),(24.52,2.60),(24.52,6.13),(21.68,6.13),(21.68,7.39)],lw=.9)
    arrow([(21.68,6.13),(16.40,6.13),(16.40,7.14)],lw=.9)
    text(23.40,4.42,r'$(m_i,\rho_i)$',8.9)
    arrow([(7.67,7.18),(7.67,5.96),(14.62,5.96),(14.62,3.96)],dashed=True,lw=.7)
    text(12.48,6.24,'State and workflow feedback',7.4)

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for item in labels:
        b=item.get_window_extent(renderer)
        assert fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1),item.get_text()
    for i,a in enumerate(labels):
        aa=a.get_window_extent(renderer)
        for b in labels[i+1:]:
            bb=b.get_window_extent(renderer)
            assert min(aa.x1,bb.x1)-max(aa.x0,bb.x0)<=1 or min(aa.y1,bb.y1)-max(aa.y0,bb.y0)<=1,(a.get_text(),b.get_text())
    allowed={c.lower() for c in PALETTE}|{'#ffffff','#303030'}
    for item in ax.patches:
        if item.get_fill() and item.get_facecolor()[3] > 0:
            assert to_hex(item.get_facecolor()).lower() in allowed
    out=ROOT/'docs/figures/system_overview_clean'
    out.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(out/f'system_overview.{extension}',dpi=600,facecolor='white')
    pdf=ROOT/'output/pdf/system_overview.pdf'
    fig.savefig(pdf,facecolor='white')
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(palette=PALETTE,original_icons_reused=False,
        task_order='smallest_workload',upper='three independent Q estimators for server selection',
        lower='ordinary SAC with continuous frequency',source_pdf_unchanged=True,
        wireless_scheduling_claimed=False,backup_claimed=False,text_overlap_checked=True),indent=2)+'\n',encoding='utf-8')
    print('New system overview: all-vector, original geometry, code/palette/text checks passed')


if __name__=='__main__':
    main()
