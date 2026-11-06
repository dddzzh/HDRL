"""Vector system overview, reusing the author's older PDF icon paths."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import to_hex, to_rgb
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle, Polygon
from fontTools.svgLib.path import parse_path
from fontTools.pens.basePen import BasePen

from draw_method_framework import PALETTE, INK, PALE, BLUE, GREEN, GREY, ORANGE, CYAN, check_implementation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/Administrator/Desktop/edge_cloud_network_v4.pdf')
BUNDLED_PYTHON = Path('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')


class VectorPen(BasePen):
    def __init__(self):
        super().__init__(None)
        self.vertices, self.codes = [], []

    def _moveTo(self, p):
        self.vertices.append(p)
        self.codes.append(MplPath.MOVETO)

    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(MplPath.LINETO)

    def _curveToOne(self, a, b, c):
        self.vertices.extend((a,b,c))
        self.codes.extend([MplPath.CURVE4]*3)

    def _qCurveToOne(self, a, b):
        self.vertices.extend((a,b))
        self.codes.extend([MplPath.CURVE3]*2)

    def _closePath(self):
        self.vertices.append((0,0))
        self.codes.append(MplPath.CLOSEPOLY)

    def _endPath(self):
        pass


def original_icons():
    code = """from pypdf import PdfReader
import urllib.parse,xml.etree.ElementTree as E,json,base64,sys
p=PdfReader(sys.argv[1]); t=E.fromstring(urllib.parse.unquote(p.metadata['/Subject']))
ids={'9','10','46','47','69','70'}
print(json.dumps({x.attrib['id']:base64.b64decode(x.attrib['style'].split('image=data:image/svg+xml,')[1].split(';')[0]).decode() for x in t.iter('mxCell') if x.attrib.get('id') in ids}))
"""
    result = subprocess.run([str(BUNDLED_PYTHON), '-c', code, str(SOURCE)],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    icons = json.loads(result.stdout)
    assert len(icons) == 6
    return icons


def mapped_color(value, accent):
    if value == 'none':
        return 'none'
    r,g,b = to_rgb(value)
    brightness = .2126*r+.7152*g+.0722*b
    if brightness < .29:
        return GREEN if accent == GREEN else INK
    if brightness < .60:
        return CYAN if accent == GREEN else accent
    if brightness < .77:
        return CYAN if accent == BLUE else GREY
    if brightness < .94:
        return PALE
    return 'white'


def main():
    check_implementation()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    icons = original_icons()
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':8,
                         'mathtext.fontset':'stix', 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(7.45,5.9), dpi=300)
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,18), ylim=(0,14.25))
    ax.axis('off')
    labels = []

    def text(x,y,s,size=8,weight='normal',ha='center',z=12):
        item=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=INK,weight=weight,
                     linespacing=1.15,zorder=z)
        labels.append(item)
        return item

    def line(points, lw=.65, dashed=False, z=5):
        ax.plot(*zip(*points),color=INK,lw=lw,linestyle=(0,(3,2)) if dashed else '-',zorder=z)

    def arrow(points, dashed=False, both=False, color=INK, lw=.85):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='<->' if both else '-|>',
                     mutation_scale=8,edgecolor=color,facecolor=color,lw=lw,
                     linestyle=(0,(4,2)) if dashed else '-',zorder=8))

    def box(x,y,w,h,fill='white',dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.16',
                     facecolor=fill,edgecolor=INK,lw=.8,linestyle=(0,(4,3)) if dashed else '-',zorder=1))

    def icon(which,x,y,w,h,accent=BLUE):
        root=ET.fromstring(icons[which])
        vx,vy,vw,vh=map(float,root.attrib['viewBox'].split())
        scale=min(w/vw,h/vh)
        transform=(transforms.Affine2D().translate(-vx,-vy).scale(scale,-scale)
                   .translate(x+(w-vw*scale)/2,y+(h+vh*scale)/2)+ax.transData)
        for element in root.iter():
            if element.tag.split('}')[-1] != 'path':
                continue
            assert not element.attrib.get('transform'), 'Unexpected icon transform'
            pen=VectorPen()
            parse_path(element.attrib['d'],pen)
            color=mapped_color(element.attrib.get('fill','#000000'),accent)
            ax.add_patch(PathPatch(MplPath(pen.vertices,pen.codes),transform=transform,
                                   facecolor=color,edgecolor='none',zorder=6))

    def dag(x,y,s=.8):
        points=[(x,y+.5*s),(x,y-.5*s),(x+.95*s,y),(x+1.9*s,y+.5*s),(x+1.9*s,y-.5*s),(x+2.85*s,y)]
        for a,b in ((0,2),(1,2),(2,3),(2,4),(3,5),(4,5)):
            arrow([points[a],points[b]],lw=.6)
        for i,p in enumerate(points):
            ax.add_patch(Circle(p,.22*s,facecolor=(BLUE,CYAN,GREEN)[i%3],edgecolor=INK,lw=.65,zorder=9))
            text(*p,rf'$t_{i+1}$',size=7)

    def mini_network(x,y,color):
        layers=[[(x,y-.24),(x,y+.24)],[(x+.47,y-.32),(x+.47,y),(x+.47,y+.32)],[(x+.94,y)]]
        for a,b in zip(layers,layers[1:]):
            for p in a:
                for q in b:
                    line([p,q],.45)
        for group in layers:
            for p in group:
                ax.add_patch(Circle(p,.09,facecolor=color,edgecolor=INK,lw=.5,zorder=8))

    # Physical topology; server icons depict heterogeneous resource classes, not counts.
    box(.35,10.32,16.25,3.50,dashed=True)
    text(.65,13.48,'Cloud layer',9.8,'bold',ha='left')
    text(16.25,13.48,'Heterogeneous cloud servers',8,ha='right')
    for x,label in ((5.20,r'$m_1^{\mathrm{c}}$'),(11.15,r'$m_{|\mathcal{M}^{\mathrm{c}}|}^{\mathrm{c}}$')):
        ax.add_patch(Ellipse((x+.68,11.16),2.85,.48,facecolor=PALE,edgecolor='none',zorder=2))
        icon('69',x,11.10,1.35,1.78)
        icon('70',x-.65,10.98,1.12,1.04)
        text(x+.68,10.65,'Cloud server '+label,8.4,'bold')
    text(8.48,11.85,r'$\cdots$',18)
    line([(5.20,11.65),(2.45,11.65),(2.45,10.35)],dashed=False)
    line([(12.1,11.65),(14.75,11.65),(14.75,10.35)],dashed=False)

    # The inter-tier band carries dependency transfers, not unmodeled device upload.
    arrow([(2.45,10.35),(2.45,9.43)],both=True,color=INK,lw=1.05)
    arrow([(14.75,10.35),(14.75,9.43)],both=True,color=INK,lw=1.05)
    box(3.50,9.64,10.15,.48,CYAN)
    text(8.575,9.88,'Inter-server communication: DAG dependency data',8.0)

    box(.35,5.52,16.25,3.91,dashed=True)
    text(.65,9.06,'Edge layer',9.8,'bold',ha='left')
    text(16.25,9.06,'Heterogeneous DVFS-enabled edge servers',8,ha='right')
    for x,label in ((1.05,r'$m_1^{\mathrm{e}}$'),(4.9,r'$m_2^{\mathrm{e}}$'),
                    (8.75,r'$m_3^{\mathrm{e}}$'),(12.60,r'$m_{|\mathcal{M}^{\mathrm{e}}|}^{\mathrm{e}}$')):
        ax.add_patch(Ellipse((x+1.4,6.45),3.05,.52,facecolor=PALE,edgecolor='none',zorder=2))
        icon('10',x+.20,6.36,1.02,1.35,GREEN)
        icon('9',x+1.37,6.4,.92,1.98,INK)
        text(x+1.42,5.99,'Edge server '+label,8.15,'bold')
    # Horizontal topology is a schematic network connection.
    arrow([(3.35,7.32),(4.78,7.32)],both=True,lw=.65)
    arrow([(7.20,7.32),(8.63,7.32)],both=True,lw=.65)
    arrow([(11.05,7.32),(12.48,7.32)],both=True,lw=.65)

    # Bottom-left application arrivals and workflow-level constraints.
    box(.35,.48,6.28,4.58)
    text(.68,4.68,'Workflow arrivals',9.5,'bold',ha='left')
    icon('46',.65,3.10,1.15,1.2)
    icon('47',1.35,2.96,.64,.95)
    text(1.28,2.56,'Users / devices',7.7)
    arrow([(2.03,3.59),(2.58,3.59)])
    dag(2.92,3.62,.89)
    text(4.20,2.76,r'$G_k=(\mathcal{N}_k,\mathcal{E}_k,D_k,\Theta_k)$',9.0)
    line([(.7,2.21),(6.24,2.21)],.5)
    text(.72,1.77,r'Arrival time: $a_k$',8.3,ha='left')
    text(.72,1.25,r'Deadline: $D_k$',8.3,ha='left')
    text(3.55,1.25,r'Reliability: $\Theta_k$',8.3,ha='left')
    text(.72,.77,'Dynamic DAG workflow stream',8.0,ha='left')

    # Current two-stage learned action, with the shared task-ordering rule separate.
    box(7.75,.48,8.85,4.58)
    text(12.17,4.68,'Hierarchical scheduling controller',9.5,'bold')
    box(8.12,3.72,8.1,.54,PALE)
    for j in range(4):
        ax.add_patch(Rectangle((8.38+j*.25,3.87),.17,.23,facecolor=GREY,edgecolor=INK,lw=.4,zorder=4))
    text(12.93,3.99,'Ready-task queue: smallest workload first',7.8)
    box(8.12,2.61,3.69,.80,BLUE)
    mini_network(8.38,3.01,BLUE)
    text(10.54,3.12,'Upper D3QN',8.2,'bold')
    text(10.54,2.84,'Server selection',7.6)
    box(12.45,2.61,3.77,.80,GREEN)
    mini_network(12.71,3.01,GREEN)
    text(14.91,3.12,'Lower SAC',8.2,'bold')
    text(14.91,2.84,'Frequency control',7.6)
    arrow([(11.81,3.01),(12.45,3.01)])
    arrow([(12.17,3.72),(12.17,3.52),(9.95,3.52),(9.95,3.41)],lw=.7)
    box(8.12,1.54,8.1,.65,ORANGE)
    text(12.17,1.87,r'Scheduling action: server $m_i$ and frequency ratio $\rho_i$',8.4,'bold')
    arrow([(9.95,2.61),(9.95,2.19)],lw=.7)
    arrow([(14.33,2.61),(14.33,2.19)],lw=.7)
    text(12.17,.98,'State, energy, deadline and reliability feedback',7.6)

    arrow([(6.63,3.6),(7.75,3.6)])
    text(7.18,4.00,r'$G_k$',10)
    # Control and telemetry are distinct from physical dependency-data transfer.
    arrow([(16.60,1.88),(17.50,1.88),(17.50,12.2),(16.60,12.2)],color=INK)
    arrow([(17.50,6.8),(16.60,6.8)],color=INK)
    text(17.75,8.8,'Execution control',7.3).set_rotation(90)
    arrow([(16.6,11.2),(17.0,11.2),(17.0,.98),(16.6,.98)],dashed=True,lw=.7)
    arrow([(16.6,7.75),(17.0,7.75)],dashed=True,lw=.7)
    text(8.55,5.27,'Joint edge-cloud resource pool',8.0)

    # Validate text bounds, pairwise overlaps and exact semantic palette.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for a in labels:
        b=a.get_window_extent(renderer)
        assert fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1),a.get_text()
    for i,a in enumerate(labels):
        aa=a.get_window_extent(renderer)
        for b in labels[i+1:]:
            bb=b.get_window_extent(renderer)
            assert min(aa.x1,bb.x1)-max(aa.x0,bb.x0)<=1 or min(aa.y1,bb.y1)-max(aa.y0,bb.y0)<=1,(a.get_text(),b.get_text())
    allowed={c.lower() for c in PALETTE}|{'#ffffff','#303030'}
    for item in ax.patches:
        if item.get_fill() and item.get_facecolor()[3] > 0:
            assert to_hex(item.get_facecolor()).lower() in allowed
    asset=ROOT/'docs/figures/edge_cloud_system'
    asset.mkdir(parents=True,exist_ok=True)
    pdf=ROOT/'output/pdf/edge_cloud_system_d3qn_sac.pdf'
    fig.savefig(pdf,facecolor='white')
    for suffix in ('png','svg'):
        fig.savefig(asset/f'edge_cloud_system_d3qn_sac.{suffix}',dpi=600,facecolor='white')
    plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha
    (asset/'provenance.json').write_text(json.dumps(dict(source_pdf=str(SOURCE),source_sha256=source_sha,
        palette=PALETTE,icon_source='Embedded editable SVG paths from author-supplied prior diagram',
        code_anchor='tools/train_coupled_hierarchical_d3qn_sac.py; v3_sm_74_legacy_train.json',
        task_order='shared smallest_workload, not a learned task-selection policy',
        learned_actions=['server selection','continuous DVFS frequency'],
        no_backup_duplication=True,communication='DAG dependency data; upload/download energy outside model',
        text_overlap_checked=True),indent=2)+'\n',encoding='utf-8')
    print(f'Created vector system diagram; original PDF unchanged: {pdf}')


if __name__ == '__main__':
    main()
