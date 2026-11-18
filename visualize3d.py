"""3D 结果可视化: 标定 + 弹道拟合结果画回视频

画面元素:
  灰线   场地底面轮廓           红线   网顶(高度1.55m)      橙线   网柱
  青杆   球员脚位竖直标杆1.75m   白点   球检测位置
  黄虚线 拟合 3D 弹道投影(随时间逐帧揭示, 前瞻 0.25s)
  绿圈   估计击球点(反推到击球者轴处, 标注出手速度 km/h)
  品红 X 拟合落点(米制)          青点   过网点

用法:
  python visualize3d.py --results-dir results/<name> [--start F] [--end F]
                        [--top N] [--only F0,F0,...] [--lookahead 0.25]
输出:
  results/<name>/calib3d_vis.mp4            (全片, 默认画全部可信弧)
  results/<name>/calib3d_vis_{s}_{e}.mp4    (指定窗口)
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

from trajectory3d import FPS, simulate

CL = 13.4
TYPE_EN = {'高远球/挑球': 'CLEAR/LIFT', '平抽': 'DRIVE', '杀球': 'SMASH',
           '吊球': 'DROP', '放网': 'NET', '未识别': ''}


def project(P, X):
    X = np.atleast_2d(X)
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    x = Xh @ P.T
    ok = x[:, 2] > 1e-6
    uv = np.full((len(X), 2), np.nan)
    uv[ok] = x[ok, :2] / x[ok, 2:3]
    return uv


def draw_dashed(vis, pts, color, thick=2, dash=7, gap=5):
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            continue
        seg = float(np.hypot(*(q - p)))
        if seg < 1:
            continue
        n = max(1, int(seg // (dash + gap)) + 1)
        for k in range(n):
            t0 = k * (dash + gap) / seg
            t1 = min((k * (dash + gap) + dash) / seg, 1.0)
            a = (p + t0 * (q - p)).astype(int)
            b = (p + t1 * (q - p)).astype(int)
            cv2.line(vis, tuple(a), tuple(b), color, thick, cv2.LINE_AA)


def label(vis, xy, text, color=(255, 255, 255)):
    x, y = int(xy[0]), int(xy[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = max(2, min(x, vis.shape[1] - tw - 8))
    y = max(th + 6, min(y, vis.shape[0] - 6))
    cv2.rectangle(vis, (x - 3, y - th - 5), (x + tw + 3, y + 4), (30, 30, 30), -1)
    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_static(vis, P):
    def seg(pts3, color, thick):
        uv = project(P, pts3)
        m = np.isfinite(uv).all(1)
        if m.sum() > 1:
            cv2.polylines(vis, [uv[m].astype(int)], False, color, thick, cv2.LINE_AA)

    seg([(0, 0, 0), (6.1, 0, 0), (6.1, CL, 0), (0, CL, 0), (0, 0, 0)], (185, 185, 185), 1)
    seg([(0, 6.7, 1.55), (6.1, 6.7, 1.55)], (0, 0, 255), 2)
    seg([(0, 6.7, 0), (0, 6.7, 1.55)], (0, 140, 255), 2)
    seg([(6.1, 6.7, 0), (6.1, 6.7, 1.55)], (0, 140, 255), 2)


def main():
    ap = argparse.ArgumentParser(description='3D 标定+弹道可视化回写视频')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--top', type=int, default=0, help='只画出手速度最快的 N 弧 (0=全部可信弧)')
    ap.add_argument('--only', default='', help='逗号分隔的 frame0 列表, 优先于 --top')
    ap.add_argument('--lookahead', type=float, default=0.25, help='弹道虚线前瞻秒数')
    ap.add_argument('--no-suspect', action='store_true', help='不画低置信弧(灰线)')
    args = ap.parse_args()

    with open(os.path.join(args.results_dir, 'calib3d.json')) as fp:
        c = json.load(fp)
    K, R, t = np.array(c['K']), np.array(c['R']), np.array(c['t'])
    P = K @ np.concatenate([R, t[:, None]], 1)

    with open(os.path.join(args.results_dir, 'metadata.json')) as fp:
        meta = json.load(fp)
    vid = meta['video']['path']
    if not os.path.exists(vid):
        vid = meta['outputs']['video']
    fps = meta['video']['fps'] or FPS
    W, H = meta['video']['width'], meta['video']['height']

    court, ballimg = {}, {}
    with open(os.path.join(args.results_dir, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            court[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                 if p and p.get('court')}
            sc = d.get('shuttlecock') or {}
            if sc.get('image') is not None:
                ballimg[d['frame']] = sc['image']

    with open(os.path.join(args.results_dir, 'trajectory3d.json')) as fp:
        results = json.load(fp)['results']
    if args.only:
        want = {int(x) for x in args.only.split(',')}
        arcs = [r for r in results if r['frame0'] in want]
    else:
        arcs = [r for r in results if not (r['suspect'] and args.no_suspect)]
        if args.top > 0:
            arcs = sorted(arcs, key=lambda r: -(r.get('v_owner') or 0))[:args.top]
    if not arcs:
        print('没有可画的弧 (先跑 trajectory3d.py)')
        return

    # 预演每条弧: 稠密采样 -> 投影 -> 落点/过网时刻
    prepared = []
    for r in arcs:
        p0, v0 = np.array(r['p0']), np.array(r['v0'])
        t_end = (r['frame1'] - r['frame0']) / fps + 0.8
        taus = np.arange(0.0, t_end, 0.01)
        pos, _, _ = simulate(p0, v0, r['beta'], taus)
        uv = project(P, pos)
        land_t = net_t = None
        below = np.where(pos[:, 2] <= 0)[0]
        if len(below) and below[0] > 0 and r.get('land'):
            land_t = float(taus[below[0]])
        cross = np.where((pos[:-1, 1] - 6.7) * (pos[1:, 1] - 6.7) <= 0)[0]
        if len(cross):
            i = int(cross[0])
            net_t = float(taus[i])
        prepared.append(dict(r=r, taus=taus, uv=uv, land_t=land_t, net_t=net_t,
                             t_end=float(taus[-1])))

    end = args.end if args.end is not None else meta['video']['total_frames']
    tag = f'_{args.start}_{end}' if (args.start or args.end is not None) else ''
    out = os.path.join(args.results_dir, f'calib3d_vis{tag}.mp4')
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    print(f'绘制弧 {len(prepared)} 条, 帧 [{args.start},{end}) -> {out}')

    for f in range(args.start, end):
        ret, frame = cap.read()
        if not ret:
            break
        draw_static(frame, P)
        for side, p in court.get(f, {}).items():
            pw = (float(p[0]), CL - float(p[1]))
            uv = project(P, [(pw[0], pw[1], 0), (pw[0], pw[1], 1.75)])
            if np.isfinite(uv).all():
                cv2.line(frame, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)),
                         (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(uv[0].astype(int)), 4, (255, 255, 0), -1)
        if f in ballimg:
            cv2.circle(frame, (int(ballimg[f][0]), int(ballimg[f][1])), 3,
                       (255, 255, 255), -1)

        t_now = (f - args.start) / fps
        for pr in prepared:
            r = pr['r']
            t0 = (r['frame0'] - args.start) / fps
            if t_now < t0 - 0.3 or t_now > t0 + pr['t_end'] + 0.4:
                continue
            sus = bool(r['suspect'])
            color, thick = ((0, 255, 255), 2) if not sus else ((175, 175, 175), 1)
            reveal = t_now - t0 + args.lookahead
            m = pr['taus'] <= max(reveal, 0.01)
            pts = pr['uv'][m]
            ok = np.isfinite(pts).all(1)
            if ok.sum() > 1:
                draw_dashed(frame, pts[ok], color, thick)
            if not sus:
                if r.get('land') and pr['land_t'] is not None and reveal >= pr['land_t']:
                    lw = (r['land'][0], CL - r['land'][1])
                    uv = project(P, [(lw[0], lw[1], 0.02)])
                    if np.isfinite(uv[0]).all():
                        x, y = int(uv[0, 0]), int(uv[0, 1])
                        cv2.drawMarker(frame, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                if pr['net_t'] is not None and reveal >= pr['net_t']:
                    i = int(np.searchsorted(pr['taus'], pr['net_t']))
                    if np.isfinite(pr['uv'][i]).all():
                        cv2.circle(frame, tuple(pr['uv'][i].astype(int)), 5, (255, 255, 0), 2)
                if 0 <= t_now - t0 <= 0.8:
                    uv = project(P, [r['p0']])
                    if np.isfinite(uv[0]).all():
                        cv2.circle(frame, tuple(uv[0].astype(int)), 9, (0, 255, 0), 2)
                        v = r.get('v_owner') or r['v_first']
                        tag_txt = f"v0~{v * 3.6:.0f}km/h"
                        en = TYPE_EN.get(r.get('type') or '', '')
                        if en:
                            tag_txt += f' {en}'
                        label(frame, (uv[0, 0] + 14, uv[0, 1] - 14), tag_txt, (0, 255, 0))
            elif 0 <= t_now - t0 <= 0.6:
                uv = project(P, [r['p0']])
                if np.isfinite(uv[0]).all():
                    label(frame, (uv[0, 0] + 12, uv[0, 1] - 12), '??', (175, 175, 175))
        writer.write(frame)

    cap.release()
    writer.release()
    if shutil.which('ffmpeg'):
        tmp = out + '.h264.mp4'
        r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', out,
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                            '-preset', 'fast', '-movflags', '+faststart', tmp])
        if r.returncode == 0:
            os.replace(tmp, out)
    print(f'完成: {out}')


if __name__ == '__main__':
    main()
