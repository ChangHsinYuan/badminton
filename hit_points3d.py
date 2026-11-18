"""第1级 3D 测量: 击球点高度解算 (球射线 x 击球者脚位竖直线)

原理:
  相机已由 camera_selfcalib.py 完整标定 (K, R, t, 米制, 工作系=y翻转右手系)。
  击球瞬间球在拍上, 水平位置 ≈ 击球者脚位 (前上方, 偏差 ~1m 内)。
  球像素 -> 射线 X(t) = C + t·d (工作系); 击球者脚位 (px,py) -> 竖直线。
  两线最近逼近处的 z 即击球点高度估计; 两线水平距离 gap 为质量指标
  (gap 应 < 臂长+拍长 ~1.2m; 过大 = 首检帧离接触太远 或 owner 判错)。

已知偏差:
  img0 是出手后首个检出帧 (快球落后 ~0.05-0.1s), 球已离开接触点
  -> 下落中的杀球 z 略偏低, gap 略偏大。精确值需第3级(物理轨迹拟合)。

用法:
  python hit_points3d.py --results-dir results/<name>
输出:
  results/<name>/hit_points3d.json + 终端汇总
"""
import argparse
import json
import os

import numpy as np

CW, CL = 6.1, 13.4
GAP_LIMIT = 1.6      # m, 超过则该击球点不可信
Z_RANGE = (0.0, 4.0)  # m, 物理合理区间


def load_calib(results_dir):
    with open(os.path.join(results_dir, 'calib3d.json')) as fp:
        c = json.load(fp)
    K = np.array(c['K'])
    R = np.array(c['R'])
    t = np.array(c['t'])
    C = np.array(c['C_working'])
    return K, R, t, C


def ball_ray(K, R, C, uv):
    """球图像点 -> 工作系射线 (C, 单位方向 d, 朝向球)"""
    d = np.linalg.inv(R) @ np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d = d / np.linalg.norm(d)
    return C, d


def closest_approach(C, d, p):
    """射线 (C,d) 与竖直线 (p, z轴) 最近逼近; 返回 (z_hit, gap, t)"""
    w0 = C - np.array([p[0], p[1], 0.0])
    e = np.array([0.0, 0.0, 1.0])
    b = d @ e
    denom = 1.0 - b * b
    if abs(denom) < 1e-6:                      # 射线近竖直: 退化, 用 x 分量解
        if abs(d[0]) < 1e-6:
            return None
        t = (p[0] - C[0]) / d[0]
    else:
        t = -(d @ w0 - b * (e @ w0)) / denom
    if t <= 0:
        return None
    P = C + t * d
    gap = float(np.hypot(P[0] - p[0], P[1] - p[1]))
    return float(P[2]), gap, float(t)


def main():
    ap = argparse.ArgumentParser(description='击球点高度解算(球射线 x 球员竖直线)')
    ap.add_argument('--results-dir', required=True)
    args = ap.parse_args()

    K, R, t, C = load_calib(args.results_dir)
    with open(os.path.join(args.results_dir, 'shot_analysis', 'shot_stats.json')) as fp:
        arcs = json.load(fp)['arcs']

    court = {}   # frame -> {upper: [x,y], lower: [x,y]}
    with open(os.path.join(args.results_dir, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            court[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                 if p and p.get('court')}

    hits = []
    for a in arcs:
        fr, owner = a['frame0'], a.get('owner')
        pos = court.get(fr, {}).get(owner)
        if pos is None:
            continue
        p_w = (float(pos[0]), CL - float(pos[1]))   # H.npy 系 -> 工作系
        _, d = ball_ray(K, R, C, a['img0'])
        res = closest_approach(C, d, p_w)
        if res is None:
            continue
        z, gap, _ = res
        hits.append(dict(frame=fr, time_sec=round(fr / 30.0, 2), owner=owner,
                         type=a.get('type'), conf=a.get('conf'),
                         z_hit=round(z, 2), gap=round(gap, 2),
                         court=pos, arc=(a['frame0'], a['frame1'])))

    ok = [h for h in hits if h['gap'] <= GAP_LIMIT and Z_RANGE[0] <= h['z_hit'] <= Z_RANGE[1]]
    bad_gap = [h for h in hits if h['gap'] > GAP_LIMIT]
    bad_z = [h for h in hits if h['gap'] <= GAP_LIMIT and not (Z_RANGE[0] <= h['z_hit'] <= Z_RANGE[1])]

    print(f'=== 击球点高度 (第1级 3D): {args.results_dir} ===')
    print(f'击球弧 {len(arcs)} -> 可解 {len(hits)} | 可信 {len(ok)} '
          f'(gap>{GAP_LIMIT}m: {len(bad_gap)}, z出界: {len(bad_z)})')

    by_type = {}
    for h in ok:
        by_type.setdefault(h['type'], []).append(h['z_hit'])
    print(f'\n{"击球类型":<14}{"n":>4}{"中位 z(m)":>10}{"四分位距":>14}')
    for tp, zs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        zs = np.array(zs)
        q1, q3 = np.percentile(zs, [25, 75])
        print(f'{str(tp):<14}{len(zs):>4}{np.median(zs):>10.2f}{f"[{q1:.2f},{q3:.2f}]":>14}')

    for side in ('upper', 'lower'):
        zs = [h['z_hit'] for h in ok if h['owner'] == side]
        if zs:
            print(f'{side}: n={len(zs)} 中位 z={np.median(zs):.2f}m '
                  f'max={max(zs):.2f}m')

    gaps = np.array([h['gap'] for h in ok]) if ok else np.array([0.0])
    print(f'\ngap (球射线与球员轴水平距): 中位 {np.median(gaps):.2f}m, '
          f'p90 {np.percentile(gaps, 90):.2f}m')

    out = dict(gap_limit=GAP_LIMIT, z_range=Z_RANGE, hits=hits,
               usable=[h['frame'] for h in ok])
    with open(os.path.join(args.results_dir, 'hit_points3d.json'), 'w') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f'已保存: {os.path.join(args.results_dir, "hit_points3d.json")}')


if __name__ == '__main__':
    main()
