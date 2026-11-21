"""逐帧 45° 俯瞰动画, 与原视频时间轴对齐并排合成

消费 trajectory3d_anchor.json (锚定拟合) + detections.jsonl (球检出/人脚位)。
每帧:
  原视频帧 (左) | 45° 俯瞰 (右)
  俯瞰: 静态球场+球网; 当前弧实线彩弹道 (至当前帧); 检出中断/低置信弧 ->
        物理预测延续的灰虚线 (至落地, 最长 +1.2s); 球员销钉 (当前帧脚位);
        累计脚步轨迹; 球 3D 位置亮点。
无弧覆盖的帧: 上一弧的灰虚线预测继续 (落地后干净留白)。

用法:
  python render_aerial_video.py --results-dir results/<name> \
      [--t0 0] [--t1 -1] [--stride 1] [--out aerial_sync.mp4]
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import trajectory3d as T3
from render_aerial import (TYPE_COLOR, COLOR_CYCLE, court_lines, CW, CL,
                           NET_Y, NET_H, FPS)

_CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_CJK):
    from matplotlib import font_manager as _fm
    _fm.fontManager.addfont(_CJK)
    matplotlib.rcParams['font.family'] = _fm.FontProperties(fname=_CJK).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

GRAY = '#888888'
OUT_H = 540          # 合成帧高度
DT_MAX = 0.2         # 预测虚线最长延伸 (s), 之后留白
PAD = 8


def load_all(rd):
    with open(os.path.join(rd, 'trajectory3d_anchor.json')) as fp:
        rows = json.load(fp)['rows']
    with open(os.path.join(rd, 'calib3d_2plane.json')) as fp:
        delta = float(json.load(fp).get('post_offset_m', 0.3))
    ball, court_pos, video = {}, {}, None
    with open(os.path.join(rd, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            sc = d.get('shuttlecock') or {}
            if sc.get('image') is not None:
                ball[d['frame']] = sc['image']
            court_pos[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                     if p and p.get('court')}
    meta = json.load(open(os.path.join(rd, 'metadata.json')))
    return rows, delta, ball, court_pos, meta['video']['path']


def arc_paths(r):
    """弧的实线段 [0, τ_last] 与预测段 [τ_last, min(τ_land, τ_last+DT_MAX)]"""
    p0 = np.array(r['p0']); v0 = np.array(r['v0'])
    tau_all = np.arange(0.0, 2.2, 0.01)
    pos, _, _ = T3.simulate(p0, v0, r['anch']['beta'], tau_all)
    below = np.where(pos[:, 2] <= 0)[0]
    tau_end = float(tau_all[below[0]]) if len(below) else float(tau_all[-1])
    tau_last = (r['frame1'] - r['frame0']) / FPS
    good = (r['anch']['rms'] <= 0.4 and r['anch']['apex'] <= 4.5
            and r['anch']['v_first'] <= 45)
    return dict(tau_last=tau_last, tau_end=tau_end, good=good, pos=pos, tau=tau_all)


class SyncRenderer:
    def __init__(self, rd):
        self.rd = rd
        self.rows, self.delta, self.ball, self.court_pos, self.video = load_all(rd)
        for r in self.rows:
            r['path'] = arc_paths(r)
        self.fig = plt.figure(figsize=(6.6, 5.5), facecolor='white', dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.computed_zorder = False
        self._scene()
        self._artists()

    def _scene(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        ax = self.ax
        ax.add_collection3d(Poly3DCollection(
            [[(-1.2, -0.6, -0.06), (CW + 1.2, -0.6, -0.06),
              (CW + 1.2, CL + 0.6, -0.06), (-1.2, CL + 0.6, -0.06)]],
            facecolor='#f2f2ee', edgecolor='none', zorder=1))
        ax.add_collection3d(Poly3DCollection(
            [[(0, 0, 0), (CW, 0, 0), (CW, CL, 0), (0, CL, 0)]],
            facecolor='#4a8f5c', edgecolor='none', zorder=2))
        segs = [[(x0, y0, 0.04), (x1, y1, 0.04)] for x0, y0, x1, y1 in court_lines()]
        ax.add_collection3d(Line3DCollection(segs, colors='#ffffff', linewidths=2.4,
                                             zorder=3))
        xs = (-self.delta, CW + self.delta)
        for x in xs:
            ax.plot([x, x], [NET_Y, NET_Y], [0, NET_H], color='#444444', lw=2.5,
                    zorder=4)
        ax.plot([xs[0], xs[1]], [NET_Y, NET_Y], [NET_H, NET_H], color='#333333',
                lw=3, zorder=4)
        for x in np.linspace(xs[0], xs[1], 17)[1:-1]:
            ax.plot([x, x], [NET_Y, NET_Y], [0.05, NET_H - 0.05], color='#999999',
                    lw=0.6, alpha=0.45, zorder=4)
        ax.set_xlim(-1.0, CW + 1.0)
        ax.set_ylim(-0.5, CL + 0.5)
        ax.set_zlim(0, 5.2)
        ax.set_box_aspect((CW + 2.0, CL + 1.0, 5.2 * 1.35))
        ax.view_init(elev=45, azim=-60)
        ax.set_axis_off()

    def _artists(self):
        ax = self.ax
        self.ln_solid, = ax.plot([], [], [], lw=2.4, zorder=5)      # 当前弧
        self.ln_dash, = ax.plot([], [], [], lw=1.6, ls='--', color=GRAY,
                                alpha=0.85, zorder=5)               # 预测/低置信
        self.dot = ax.scatter([], [], [], s=42, depthshade=False, zorder=7,
                              edgecolors='k', linewidths=0.6)
        self.pins, self.feet = {}, {}
        for side, col in (('upper', '#e377c2'), ('lower', '#7f7f7f')):
            (p,), = [ax.plot([], [], [], color=col, lw=3.5, zorder=6)]
            self.pins[side] = (p, ax.scatter([], [], [], s=55, color=col,
                                             depthshade=False, zorder=8))
            self.feet[side], = ax.plot([], [], [], color=col, lw=0.9, ls=':',
                                       alpha=0.5, zorder=4)

    def _slice(self, path, t0, t1):
        m = (path['tau'] >= t0 - 1e-9) & (path['tau'] <= min(t1, path['tau_end']) + 1e-9)
        pos = path['pos'][m]
        return pos[:, 0], pos[:, 1], np.maximum(pos[:, 2], 0.0)

    def _active(self, fr):
        """当前帧的弧: 优先正在检出的, 否则最近一个已开始的"""
        best = None
        for r in self.rows:
            if r['frame0'] <= fr:
                if best is None or r['frame0'] > best['frame0']:
                    best = r
        return best

    def update(self, fr):
        r = self._active(fr)
        solid, dash, ball_pos = ([], [], []), ([], [], []), None
        label = ''
        if r is not None:
            t_now = (fr - r['frame0']) / FPS
            path = r['path']
            if t_now <= path['tau_last']:      # 检出段内
                solid = self._slice(path, 0.0, t_now)
                col = TYPE_COLOR.get(r['type'] or '', COLOR_CYCLE[0]) \
                    if path['good'] else GRAY
                self.ln_solid.set_color(col)
                self.ln_solid.set_linestyle('-' if path['good'] else '--')
                label = f"{r['type'] or '未识别'}"
            else:                              # 检出后: 预测虚线
                solid = ([], [], [])
                self.ln_solid.set_color(GRAY)
            if t_now > path['tau_last']:
                dash = self._slice(path, path['tau_last'],
                                   min(t_now, path['tau_last'] + DT_MAX))
            elif not path['good']:
                dash = solid
            if len(solid[0]):
                ball_pos = tuple(a[-1] for a in solid)
            elif len(dash[0]):
                ball_pos = tuple(a[-1] for a in dash)
        self.ln_solid.set_data_3d(*solid)
        self.ln_dash.set_data_3d(*dash)
        if ball_pos is not None:
            self.dot._offsets3d = ([ball_pos[0]], [ball_pos[1]], [ball_pos[2]])
            self.dot.set_visible(True)
        else:
            self.dot.set_visible(False)
        # 球员: 当前帧脚位 (就近 ±5 帧) + 累计脚步
        for side in ('upper', 'lower'):
            p = None
            for f2 in [fr] + [fr + k for k in range(1, 6) for f2 in (fr - k, fr + k)]:
                q = self.court_pos.get(f2, {}).get(side)
                if q is not None:
                    p = (q[0], T3.CL_W - q[1])
                    break
            line, sc = self.pins[side]
            if p is None:
                line.set_data_3d([], [], [])
                sc._offsets3d = ([], [], [])
            else:
                line.set_data_3d([p[0], p[0]], [p[1], p[1]], [0, 1.75])
                sc._offsets3d = ([p[0]], [p[1]], [1.75])
            fts = [(self.court_pos[f][side][0], T3.CL_W - self.court_pos[f][side][1])
                   for f in sorted(self.court_pos)
                   if f <= fr and side in self.court_pos[f]]
            if fts:
                a = np.array(fts)
                self.feet[side].set_data_3d(a[:, 0], a[:, 1], np.full(len(a), 0.06))
            else:
                self.feet[side].set_data_3d([], [], [])
        self.ax.set_title(f'45° 俯瞰  f{fr}  t={fr / FPS:.2f}s  {label}', fontsize=10)

    def render_frame(self, fr):
        self.update(fr)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description='45° 俯瞰逐帧动画 (与原视频并排)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--t0', type=int, default=0)
    ap.add_argument('--t1', type=int, default=-1)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    R = SyncRenderer(args.results_dir)
    cap = cv2.VideoCapture(R.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr0, fr1 = args.t0, (n_total - 1 if args.t1 < 0 else args.t1)
    out = args.out or os.path.join(args.results_dir, 'aerial_sync.mp4')
    for _ in range(fr0):
        cap.read()
    # 试渲染一帧定尺寸
    ok, frame = cap.read()
    if not ok:
        raise SystemExit('读不到帧')
    left = cv2.resize(frame, (int(frame.shape[1] * OUT_H / frame.shape[0]), OUT_H))
    right = cv2.resize(R.render_frame(fr0), (int(R.fig.get_size_inches()[0] * 100
                                                 * OUT_H / (R.fig.get_size_inches()[1] * 100)), OUT_H))
    W = left.shape[1] + right.shape[1] + PAD
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                             (W, OUT_H))
    fr = fr0
    n_written = 0
    while fr <= fr1:
        if fr > fr0:
            ok, frame = cap.read()
            if not ok:
                break
        left = cv2.resize(frame, (left.shape[1], OUT_H))
        right = cv2.resize(R.render_frame(fr), (right.shape[1], OUT_H))
        canvas = np.full((OUT_H, W, 3), 255, np.uint8)
        canvas[:, :left.shape[1]] = left
        canvas[:, left.shape[1] + PAD:] = right
        for _ in range(args.stride):
            writer.write(canvas)
        n_written += 1
        if n_written % 100 == 0:
            print(f'  {fr}/{fr1} ({100 * fr / max(fr1, 1):.0f}%)', flush=True)
        fr += 1
    writer.release()
    print(f'已保存: {out}  ({n_written} 帧渲染, {n_written * args.stride} 帧写出)')


if __name__ == '__main__':
    main()
