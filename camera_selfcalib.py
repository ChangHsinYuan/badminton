"""档1 相机自标定实验: 地面单应 H(image->court) -> 完整 K(f) + R + t(米制)

原理:
  G = inv(H) 是地面平面(场地米制坐标, Z=0)到图像的单应, 可分解为 G = K [r1 r2 t]
  假设方形像素、主点居中, 内参只剩 f 一个未知数;
  由 r1⊥r2 与 |r1|=|r2| 得到两个互相独立的 a=1/f² 方程:
      h1ᵀ M h2 = 0 与 h1ᵀ M h1 = h2ᵀ M h2,  M = K⁻ᵀK⁻¹ = a·N + E
  两个 f 估计的一致性 = 假设(主点居中/零畸变)违反程度的量化
  球场米制尺寸(6.1 x 13.4)锚定尺度, 因此 t 与相机中心 C 都是真实米制

朝向类(镜像歧义): canonicalize() 用镜像变换规范化场地朝向, 可能使场地坐标系
  成为左手系, 而分解假设 x×y=z(右手系) -> 相机被镜像到地面下方(C_z<0)。
  地面单应无法区分两个朝向类, 但物理可以: 正确解必须相机离地 C_z>0。
  本脚本对两个朝向类各解一次, 自动选取 C_z>0 的一支。

自动验证:
  1. 两个 f 估计的相对偏差
  2. 相机中心 C 的合理性(应在场边、离地 ~1-2m)
  3. 网顶线(高度 1.55m)投影回图像 vs 图像里白色网带的实测行(定量, px)
  4. 网柱 + 球员脚位竖直标杆(1.75m)投影叠加目检

用法:
  python camera_selfcalib.py --results-dir results/<name> [--frame N]
输出:
  results/<name>/calib3d_overlay.jpg
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

CW, CL = 6.1, 13.4
NET_Y, NET_H = 6.7, 1.55
STICK_H = 1.75


def project(P, X):
    """3D 点(米, 场地坐标系) -> 图像像素; 返回 (uv, depth)"""
    X = np.atleast_2d(np.asarray(X, float))
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    x = Xh @ P.T
    depth = x[:, 2]
    ok = depth > 1e-6
    uv = np.full((len(X), 2), np.nan)
    uv[ok] = x[ok, :2] / depth[ok, None]
    return uv, depth


def decompose(G, cx, cy):
    """G: court(米, Z=0) -> image. 返回 dict(f1, f2, f, K, R, t, C)"""
    G = G / G[2, 2]
    h1, h2, h3 = G[:, 0], G[:, 1], G[:, 2]
    N = np.array([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx**2 + cy**2]], float)

    def q0(h, g):
        return h @ N @ g

    a1 = -h1[2] * h2[2] / q0(h1, h2)
    a2 = -(h1[2] ** 2 - h2[2] ** 2) / (q0(h1, h1) - q0(h2, h2))
    f1 = 1.0 / np.sqrt(a1) if a1 > 0 else np.nan
    f2 = 1.0 / np.sqrt(a2) if a2 > 0 else np.nan
    if not np.isfinite(f1):
        raise RuntimeError(f'f 估计非法: a1={a1:.3e}, a2={a2:.3e} (假设被强烈违反?)')
    f = f1 if not np.isfinite(f2) else 0.5 * (f1 + f2)

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    r1 = Kinv @ h1
    s = np.linalg.norm(r1)          # = 1/|λ|, 同一尺度因子适用于三列
    r1 = r1 / s
    r2 = (Kinv @ h2) / s
    t = (Kinv @ h3) / s
    if t[2] < 0:                    # 保证球场原点在相机前方
        r1, r2, t = -r1, -r2, -t
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - (r2 @ r1) * r1        # Gram-Schmidt 正交化(噪声清理)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)
    C = -R.T @ t
    return dict(f1=f1, f2=f2, f=f, K=K, R=R, t=t, C=C)


def load_calib(results_dir):
    H = np.load(os.path.join(results_dir, 'H.npy'))
    G = np.linalg.inv(H)
    with open(os.path.join(results_dir, 'metadata.json')) as fp:
        meta = json.load(fp)
    W, Hh = meta['video']['width'], meta['video']['height']
    cx, cy = W / 2.0, Hh / 2.0
    # 两个朝向类: 原朝向 与 y 翻转(左手/右手系)。x 翻转与 y 翻转只差一个
    # 180° 旋转(proper), 属同一类, 无需单列。
    Ty = np.array([[1, 0, 0], [0, -1, CL], [0, 0, 1]], float)
    cands = []
    for name, Gc in (('原朝向', G), ('y翻转', G @ Ty)):
        try:
            c = decompose(Gc, cx, cy)
            c['orient'] = name
            cands.append(c)
        except RuntimeError as e:
            print(f'[{name}] 分解失败: {e}')
    if not cands:
        raise RuntimeError('两个朝向类均无法分解')
    up = [c for c in cands if c['C'][2] > 0.3]
    pick = up[0] if up else min(cands, key=lambda c: abs(c['C'][2]))
    for c in cands:
        tag = ' <-- 采用' if c is pick else ''
        print(f'朝向类[{c["orient"]}]: 相机中心 z={c["C"][2]:+.2f} m{tag}')
    return pick, meta


def net_tape_offset(frame, P, x_court):
    """定量验证: 网顶投影行 vs 该列附近白色网带实测行(中值), 返回 px 偏差"""
    uv, depth = project(P, [(x_court, NET_Y, NET_H)])
    if not np.isfinite(uv[0]).all():
        return None
    u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
    if not (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    col = hsv[:, u]
    white = np.where((col[:, 1] < 70) & (col[:, 2] > 170))[0]
    near = white[(np.abs(white - v) < 60)]
    if len(near) == 0:
        return None
    return float(np.median(near)) - v


def main():
    ap = argparse.ArgumentParser(description='档1 相机自标定(地面单应 -> f/R/t)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--frame', type=int, default=None, help='叠加图帧号(默认取中点)')
    args = ap.parse_args()

    calib, meta = load_calib(args.results_dir)
    f1, f2, f, K, R, t, C = (calib[k] for k in ('f1', 'f2', 'f', 'K', 'R', 't', 'C'))
    P = K @ np.concatenate([R, t[:, None]], 1)

    print(f'=== 档1 自标定: {args.results_dir} ===')
    print(f'f 估计1 (r1⊥r2):     {f1:8.1f} px')
    print(f'f 估计2 (|r1|=|r2|): {f2:8.1f} px')
    if np.isfinite(f2):
        print(f'两估计相对偏差:      {abs(f1 - f2) / f * 100:6.2f} %  (主点居中+零畸变假设的违反度)')
    print(f'使用 f = {f:.1f} px ({meta["video"]["width"]}x{meta["video"]["height"]})')
    # 工作系 = y 翻转系(右手系); H.npy/detections 用 y=13.4-y 的镜像系(左手系)
    C_hn = (C[0], CL - C[1], C[2])
    print(f'相机中心 (H.npy 场地坐标系, z 向上): x={C_hn[0]:+.2f}  y={C_hn[1]:+.2f}  '
          f'z(高)={C_hn[2]:.2f}  [工作系 y={C[1]:+.2f}]')
    ok_h = 0.3 < C[2] < 6.0
    ok_side = (-8 < C_hn[0] < CW + 8 and C_hn[0] < -0.3 or C_hn[0] > CW + 0.3) and -6 < C_hn[1] < CL + 8
    print(f'合理性: 离地高 {"OK" if ok_h else "异常!"} ({C_hn[2]:.2f}m), '
          f'场边 {"OK" if ok_side else "异常!"} (x={"场外" if ok_side else "场内"}, y={C_hn[1]:.2f})')

    # 帧与叠加图
    vid_path = meta['outputs']['video']
    if not os.path.isabs(vid_path) and not os.path.exists(vid_path):
        vid_path = os.path.join(args.results_dir, os.path.basename(vid_path))
    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = args.frame if args.frame is not None else max(total // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'无法读取视频帧 {idx}: {vid_path}')

    # 球员标杆(取该帧两球员脚位)
    dets = {}
    det_path = os.path.join(args.results_dir, 'detections.jsonl')
    with open(det_path) as fp:
        for line in fp:
            d = json.loads(line)
            if d['frame'] == idx:
                dets = d['players']
                break
    sticks = []
    for side in ('upper', 'lower'):
        p = dets.get(side, {}).get('court')
        if p is not None and 0 <= p[0] <= CW and 0 <= p[1] <= CL:
            sticks.append((side, p))

    # 绘制: 场地对角线参考 + 网顶线 + 网柱 + 标杆
    vis = frame.copy()

    def draw3d(pts, color, thick=2):
        uv, depth = project(P, pts)
        uv = uv[np.isfinite(uv).all(1)]
        if len(uv) > 1:
            cv2.polylines(vis, [uv.astype(int)], False, color, thick, cv2.LINE_AA)

    draw3d([(0, 0, 0), (CW, 0, 0), (CW, CL, 0), (0, CL, 0), (0, 0, 0)], (200, 200, 200), 1)
    draw3d([(0, NET_Y, NET_H), (CW, NET_Y, NET_H)], (0, 0, 255), 2)          # 网顶: 红
    draw3d([(0, NET_Y, 0), (0, NET_Y, NET_H)], (0, 140, 255), 2)             # 网柱: 橙
    draw3d([(CW, NET_Y, 0), (CW, NET_Y, NET_H)], (0, 140, 255), 2)
    for side, p in sticks:
        p_w = (p[0], CL - p[1])          # detections 是 H.npy 系(左手), 换到工作系
        draw3d([(p_w[0], p_w[1], 0), (p_w[0], p_w[1], STICK_H)], (255, 255, 0), 2)
        uv, _ = project(P, [(p_w[0], p_w[1], 0)])
        if np.isfinite(uv[0]).all():
            cv2.circle(vis, (int(uv[0, 0]), int(uv[0, 1])), 5, (255, 255, 0), -1)

    # 定量: 网带实测行 vs 投影行(逐列打印, 看中心 vs 边缘的畸变特征)
    offs = []
    for xc in (0.5, 1.8, 3.05, 4.3, 5.6):
        off = net_tape_offset(frame, P, xc)
        if off is not None:
            offs.append(off)
            print(f'  网顶投影 vs 网带实测 @ court_x={xc}: {off:+.1f}px')
    if offs:
        offs = np.array(offs)
        print(f'网顶投影偏差汇总: mean={offs.mean():+.1f}px  |max|={np.abs(offs).max():.1f}px  (n={len(offs)})')
    else:
        print('网顶投影 vs 网带实测: 该帧未找到可用网带像素')
    for side, p in sticks:
        print(f'标杆: {side} 脚位 court=({p[0]:.2f},{p[1]:.2f}) 1.75m 竖杆已叠加')

    out = os.path.join(args.results_dir, 'calib3d_overlay.jpg')
    cv2.imwrite(out, vis)
    print(f'叠加图已保存: {out}')

    calib_json = dict(
        orient=calib['orient'], f1=float(f1), f2=float(f2), f=float(f),
        frame=[meta['video']['width'], meta['video']['height']],
        K=K.tolist(), R=R.tolist(), t=t.tolist(),
        C_working=calib['C'].tolist(),
        C_hn=[float(v) for v in C_hn],
        note='工作系: 右手系, y_w=13.4-y_hn; H.npy/detections 为镜像系(左手). '
             '投影 H.npy 系 3D 点(含高度)前先做 y_w=13.4-y_hn, 再用 K[R|t].',
        net_offset_px=[float(o) for o in offs], overlay_frame=int(idx),
    )
    with open(os.path.join(args.results_dir, 'calib3d.json'), 'w') as fp:
        json.dump(calib_json, fp, indent=1)
    print(f'标定结果已保存: {os.path.join(args.results_dir, "calib3d.json")}')


if __name__ == '__main__':
    main()
