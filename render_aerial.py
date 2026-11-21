"""45° 天空俯瞰 3D 渲染 —— 干净场景: 球场 + 球网 + 球轨迹 + 人 (云台销钉)

消费 trajectory3d_anchor.json (锚定拟合结果) + detections.jsonl (人脚位)。
纯工作系 3D 绘制, 不依赖相机参数。

用法:
  python render_aerial.py --results-dir results/<name> [--rally longest|N|all]
                          [--elev 45] [--azim -60] [--out aerial.png]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import trajectory3d as T3

_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK):
    from matplotlib import font_manager as _fm
    _fm.fontManager.addfont(_CJK)
    matplotlib.rcParams['font.family'] = _fm.FontProperties(fname=_CJK).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

CW, CL, NET_Y, NET_H = 6.1, 13.4, 6.7, 1.55
FPS = T3.FPS

TYPE_COLOR = {
    '杀球': '#d62728', '吊球': '#1f77b4', '高远球/挑球': '#ff7f0e',
    '平抽': '#2ca02c', '搓球/放网': '#9467bd',
}
COLOR_CYCLE = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']


def court_lines():
    """工作系 (x 0..6.1, y 0..13.4) 标准羽毛球线, 返回 [(x0,y0,x1,y1)]"""
    L = []
    L += [(0, 0, CW, 0), (0, CL, CW, CL), (0, 0, 0, CL), (CW, 0, CW, CL)]  # 外框
    L += [(0, NET_Y - 1.98, CW, NET_Y - 1.98), (0, NET_Y + 1.98, CW, NET_Y + 1.98)]  # 前发球线
    L += [(0, 0.76, CW, 0.76), (0, CL - 0.76, CW, CL - 0.76)]            # 双打后发球线
    L += [(0.46, 0, 0.46, CL), (CW - 0.46, 0, CW - 0.46, CL)]            # 单打边线
    L += [(CW / 2, NET_Y - 1.98, CW / 2, 0.76), (CW / 2, NET_Y + 1.98, CW / 2, CL - 0.76)]  # 中线
    return L


def draw_scene(ax, delta=0.0):
    # 地面 + 场地区域 (标准羽毛球场地绿); computed_zorder=False 时按 add 顺序绘制
    ground = Poly3DCollection([[(-1.2, -0.6, -0.06), (CW + 1.2, -0.6, -0.06),
                                (CW + 1.2, CL + 0.6, -0.06), (-1.2, CL + 0.6, -0.06)]],
                              facecolor='#f2f2ee', edgecolor='none', zorder=1)
    ax.add_collection3d(ground)
    court_bg = Poly3DCollection([[(0, 0, 0), (CW, 0, 0), (CW, CL, 0), (0, CL, 0)]],
                                facecolor='#4a8f5c', edgecolor='none', zorder=2)
    ax.add_collection3d(court_bg)
    # 球场线
    segs = [[(x0, y0, 0.04), (x1, y1, 0.04)] for x0, y0, x1, y1 in court_lines()]
    ax.add_collection3d(Line3DCollection(segs, colors='#ffffff', linewidths=2.4,
                                         zorder=3))
    # 球网: 柱 (立在边线外 delta) + 顶带 + 稀疏网线
    xs = (-delta, CW + delta)
    for x in xs:
        ax.plot([x, x], [NET_Y, NET_Y], [0, NET_H], color='#444444', lw=2.5)
    ax.plot([xs[0], xs[1]], [NET_Y, NET_Y], [NET_H, NET_H], color='#333333', lw=3,
            zorder=4)
    for x in np.linspace(xs[0], xs[1], 17)[1:-1]:
        ax.plot([x, x], [NET_Y, NET_Y], [0.05, NET_H - 0.05], color='#999999',
                lw=0.6, alpha=0.45, zorder=4)


def rally_split(rows):
    """按帧间隙分回合"""
    rallies, cur = [], []
    for r in sorted(rows, key=lambda r: r['frame0']):
        if cur and r['frame0'] > cur[-1]['frame1'] + 1.2 * FPS:
            rallies.append(cur)
            cur = []
        cur.append(r)
    if cur:
        rallies.append(cur)
    return rallies


def draw_rally(ax, rows, court_pos, delta, show_foot=True):
    n_bad = 0
    for i, r in enumerate(rows):
        if (r['anch']['rms'] > 0.4 or r['anch']['apex'] > 4.5
                or r['anch']['v_first'] > 45):
            n_bad += 1
            continue  # 坏拟合弧不入画
        p0 = np.array(r['p0']); v0 = np.array(r['v0'])
        tau = np.arange(0.0, 1.5, 0.01)
        pos, _, _ = T3.simulate(p0, v0, r['anch']['beta'], tau)
        pos = pos[pos[:, 2] > 0.0]
        keep = ((pos[:, 0] > -0.8) & (pos[:, 0] < CW + 0.8)
                & (pos[:, 1] > -0.8) & (pos[:, 1] < CL + 0.8))
        if keep.mean() < 0.6:
            n_bad += 1
            continue  # 弧主体在场外: 坏拟合
        pos = pos[keep]
        col = TYPE_COLOR.get(r['type'] or '', COLOR_CYCLE[i % len(COLOR_CYCLE)])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=col, lw=2.2, alpha=0.9,
                zorder=5)
        ax.scatter(*p0, color=col, s=26, depthshade=False, zorder=6)
    # 人: 回合最后一帧的双人云台销钉 + 脚步轨迹
    fr_last = max(r['frame1'] for r in rows)
    if not show_foot:
        return n_bad
    for side, col in (('upper', '#e377c2'), ('lower', '#7f7f7f')):
        pts = []
        for fr in sorted(court_pos):
            if rows[0]['frame0'] - 5 <= fr <= fr_last and side in court_pos[fr]:
                p = court_pos[fr][side]
                pts.append((p[0], T3.CL_W - p[1]))
        if not pts:
            continue
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], 0.06, color=col, lw=1.0, ls=':', alpha=0.55,
                zorder=4)
        p = court_pos.get(fr_last, {}).get(side)
        if p:
            x, y = p[0], T3.CL_W - p[1]
            ax.plot([x, x], [y, y], [0, 1.75], color=col, lw=3, zorder=7)
            ax.scatter([x], [y], [1.75], color=col, s=60, depthshade=False, zorder=8)


def main():
    ap = argparse.ArgumentParser(description='45° 俯瞰 3D 渲染')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--anchor-json', default='trajectory3d_anchor.json')
    ap.add_argument('--rally', default='longest')
    ap.add_argument('--elev', type=float, default=45)
    ap.add_argument('--azim', type=float, default=-60)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rd = args.results_dir
    with open(os.path.join(rd, args.anchor_json)) as fp:
        data = json.load(fp)
    rows = data['rows']
    with open(os.path.join(rd, 'calib3d_2plane.json')) as fp:
        delta = float(json.load(fp).get('post_offset_m', 0.3))
    court_pos = {}
    with open(os.path.join(rd, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            court_pos[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                     if p and p.get('court')}

    rallies = rally_split(rows)
    print(f'{len(rows)} 弧 / {len(rallies)} 回合, 弧数分布: '
          f'{sorted((len(r) for r in rallies), reverse=True)[:8]}...')
    if args.rally == 'longest':
        sel = max(rallies, key=len)
    elif args.rally == 'all':
        sel = rows
    else:
        sel = rallies[int(args.rally)]
    fr0, fr1 = sel[0]['frame0'], sel[-1]['frame1']
    ntype = sum(1 for r in sel if r['type'])
    print(f'渲染: 帧 {fr0}-{fr1}, {len(sel)} 弧 ({ntype} 个识别类型)')

    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False   # mplot3d 跨 collection 深度排序不稳, 手工分层
    draw_scene(ax, delta)
    n_bad = draw_rally(ax, sel, court_pos, delta, show_foot=(args.rally != 'all'))
    if n_bad:
        print(f'（过滤坏拟合弧 {n_bad} 条）')
    ax.set_xlim(-1.0, CW + 1.0)
    ax.set_ylim(-0.5, CL + 0.5)
    ax.set_zlim(0, 5.2)
    ax.set_box_aspect((CW + 2.0, CL + 1.0, 5.2 * 1.35))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.set_facecolor('white')
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=t)
               for t, c in TYPE_COLOR.items() if any(r['type'] == t for r in sel)]
    handles.append(plt.Line2D([], [], color='#e377c2', lw=3, label='upper 球员'))
    handles.append(plt.Line2D([], [], color='#7f7f7f', lw=3, label='lower 球员'))
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(f'Aerial 45° — frames {fr0}-{fr1} ({len(sel)} shots)', fontsize=11)
    out = args.out or os.path.join(rd, 'aerial_45.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()
