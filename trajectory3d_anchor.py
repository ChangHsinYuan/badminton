"""第3级+ 击球锚定轨迹拟合 —— 用户巧思: 击球瞬间球必在击球人轴正上方

新文件, 不改动 trajectory3d.py (导入其 simulate/ray_dir/fit_arc/metrics 复用)。

原理:
  单目标定的本质: 每帧球观测给一条射线 (方向全对, 尺度差)。
  人的脚底经场地单应得精确 (x,y,0) -> 击球人/接球人位置处一根竖直线。
  首末观测射线 与 该竖直线求交 -> 3D 锚点, 高度 z 是几何解出的 (非假设)。
  一致性: 射线与竖直线的水平距离 miss <= MISS_MAX 才可信 (拍 reach ~1m + 检测延迟)。
  锚点作为 soft_l1 3D 先验进入两阶段拟合, 直接打破单目"慢近/快远"尺度简并。

A/B: 基线 = trajectory3d 生产拟合 (软 xy 距离锚)
     锚定 = 3D 击球锚 (有则用, 无则退回该端软锚) —— 信息严格超集, 对比公平

用法:
  python trajectory3d_anchor.py --results-dir results/<name> [--calib calib3d_2plane.json]
输出:
  results/<name>/trajectory3d_anchor.json + 终端对比表
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

import trajectory3d as T3

G = T3.G
FPS = T3.FPS
MISS_MAX = 1.2          # m, 射线-人轴水平距离门限
SIG_ANCHOR = 0.35       # m, 锚点先验 sigma
Z_HIT_RANGE = (0.0, 4.5)  # m, 锚点高度合理域 (超出=关联错误, 不锚)


def load_scene(results_dir, calib_name='calib3d_2plane.json'):
    path = os.path.join(results_dir, calib_name)
    if not os.path.exists(path):
        path = os.path.join(results_dir, 'calib3d.json')
    with open(path) as fp:
        c = json.load(fp)
    K = np.array(c['K']); R = np.array(c['R']); C = np.array(c['C_working'])
    return K, R, C, float(c['f']), os.path.basename(path)


def load_players(results_dir):
    court = {}
    with open(os.path.join(results_dir, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            court[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                 if p and p.get('court')}
    return court


def axis_anchor(C, d, xy):
    """射线 C+s·d 与竖直线 {(xy[0],xy[1],z)} 求交。
    返回 (miss, anchor3d) 或 (miss, None); miss=射线到轴的水平最近距离。"""
    dh = np.array([d[0], d[1]])
    nh = np.linalg.norm(dh)
    if nh < 0.05 or d[2] > -0.05:      # 近竖直射线/仰角射线: 高度不可辨
        return np.inf, None
    w = np.array([xy[0] - C[0], xy[1] - C[1]])
    s = float(w @ dh) / (nh * nh)
    miss = float(np.linalg.norm(w - s * dh))
    A = C + s * np.asarray(d, float)
    if not (Z_HIT_RANGE[0] + 0.01 < A[2] < Z_HIT_RANGE[1]):
        return miss, None
    return miss, A


REACH_HIT = 1.3       # m, 首观测水平轴距软门 (拍 reach + 人轴映射误差)
CAP_HIT = 2.0         # m, 铰链截幅 (晚检出弧不被过度拉扯)
Z_HIT_LO, Z_HIT_HI = 0.3, 3.2   # m, 击球高度物理范围 (弱先验)
W_HIT = 3.0
W_HIT_Z = 1.0


def fit_arc_3d(rays, C, A3d=None, B3d=None, a_start_xy=None, a_end_xy=None,
               hit_axis=None):
    """两阶段拟合 (同 trajectory3d 结构)。锚机制 (信息严格超集):
      A3d/B3d   射线∩人轴竖直线 3D 锚 (仅 dz<-0.05 下降段射线可用, 尖锐)
      hit_axis  击球人轴 (x,y): 首观测水平轴距 <= REACH_HIT 软铰链(截幅)
                + 首观测高度 z 物理范围弱先验 —— 只用水平射线强的 (x,y) 分量, 稳定
      a_*_xy    生产软 xy 锚 (兜底)"""
    taus = np.array([r[0] for r in rays])
    dirs = np.stack([r[1] for r in rays])
    if taus[-1] <= 0:
        return None
    init = T3.grid_init(rays, C)
    if init is None or not np.isfinite(init).all():
        return None
    zhat = np.array([0.0, 0.0, 1.0])

    def _anch_res(X_first, X_last):
        res = []
        if A3d is not None:
            res.extend(((X_first - A3d) / SIG_ANCHOR).tolist())
        if B3d is not None:
            res.extend(((X_last - B3d) / SIG_ANCHOR).tolist())
        if A3d is None and a_start_xy is not None:
            res.append(1.0 * max(0.0, float(np.hypot(X_first[0] - a_start_xy[0],
                                                     X_first[1] - a_start_xy[1])) - 3.0))
        if B3d is None and a_end_xy is not None:
            res.append(3.0 * max(0.0, float(np.hypot(X_last[0] - a_end_xy[0],
                                                     X_last[1] - a_end_xy[1])) - 1.5))
        if hit_axis is not None:
            gap = float(np.hypot(X_first[0] - hit_axis[0], X_first[1] - hit_axis[1]))
            res.append(W_HIT * min(max(0.0, gap - REACH_HIT), CAP_HIT))
            res.append(W_HIT_Z * (max(0.0, Z_HIT_LO - X_first[2])
                                  + max(0.0, X_first[2] - Z_HIT_HI)))
        return res

    def resid0(th):
        s0, vx, vy, vz = th
        v0 = np.array([vx, vy, vz])
        X = (C + s0 * dirs[0])[None, :] + v0[None, :] * taus[:, None] \
            + (0.5 * G * taus * taus)[:, None] * zhat[None, :]
        res = [T3.ray_dist(X[i], C, dirs[i]) for i in range(len(taus))]
        return res + _anch_res(X[0], X[-1])

    try:
        sol0 = least_squares(resid0, init, bounds=([0.5, -150, -150, -150],
                                                   [60, 150, 150, 150]), max_nfev=40)
        x1 = np.append(sol0.x, 0.15)
    except Exception:
        import traceback
        traceback.print_exc()
        x1 = np.append(init, 0.15)

    def resid(th):
        s0, vx, vy, vz, beta = th
        p0 = C + s0 * dirs[0]
        v0 = np.array([vx, vy, vz])
        pos, _, _ = T3.simulate(p0, v0, beta, taus)
        res = [T3.ray_dist(pos[i], C, dirs[i]) for i in range(len(taus))]
        return res + _anch_res(pos[0], pos[-1])

    lo = np.array([0.5, -120, -120, -120, 0.0])
    hi = np.array([60.0, 120, 120, 120, 0.6])
    try:
        sol = least_squares(resid, np.clip(x1, lo + 1e-6, hi - 1e-6),
                            bounds=(lo, hi), loss='soft_l1', f_scale=0.25, max_nfev=60)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    if not np.isfinite(sol.x).all():
        return None
    s0, vx, vy, vz, beta = sol.x
    p0 = C + s0 * dirs[0]
    v0 = np.array([vx, vy, vz])
    pos, _, _ = T3.simulate(p0, v0, beta, taus)
    res = np.array([T3.ray_dist(pos[i], C, dirs[i]) for i in range(len(taus))])
    return dict(p0=p0, v0=v0, beta=float(beta), rms=float(np.sqrt((res ** 2).mean())))


def pos_w(court, side, fr, win=5):
    """工作系脚位: (x, 13.4-y); fr 缺失时就近找 ±win 帧"""
    for f2 in [fr] + [fr2 for k in range(1, win + 1) for fr2 in (fr - k, fr + k)]:
        p = court.get(f2, {}).get(side)
        if p is not None:
            return (float(p[0]), T3.CL_W - float(p[1])), f2
    return None, fr


def main():
    ap = argparse.ArgumentParser(description='击球锚定 3D 轨迹拟合 (A/B 对比)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--calib', default='calib3d_2plane.json')
    ap.add_argument('--min-pts', type=int, default=6)
    args = ap.parse_args()

    K, R, C, f, calib_name = load_scene(args.results_dir, args.calib)
    _, _, _, _, ball, arcs = T3.load_scene(args.results_dir)
    court = load_players(args.results_dir)
    print(f'=== 击球锚定轨迹拟合: {args.results_dir} ===')
    print(f'标定: {calib_name}')

    rows, z_hits, misses = [], [], []
    n_both = n_one = 0
    for a in arcs:
        fr0, fr1 = a['frame0'], a['frame1']
        obs = [(fr, ball[fr]) for fr in range(fr0, fr1 + 1) if fr in ball]
        if len(obs) < args.min_pts:
            continue
        rays = [((fr - fr0) / FPS, T3.ray_dir(K, R, C, uv)) for fr, uv in obs]
        dirs = np.stack([r[1] for r in rays])

        # 基线: 生产软锚
        o0, _ = pos_w(court, a.get('owner'), fr0) if a.get('owner') else (None, fr0)
        r1, _ = pos_w(court, a.get('receiver'), fr1) if a.get('receiver') else (None, fr1)
        base = T3.fit_arc(rays, C, o0, r1)

        # 3D 锚: 首端=击球人, 末端=接球人
        A3d, B3d, mA, mB = None, None, np.inf, np.inf
        if o0 is not None:
            mA, A3d = axis_anchor(C, dirs[0], o0)
        if r1 is not None:
            mB, B3d = axis_anchor(C, dirs[-1], r1)
        fit3 = fit_arc_3d(rays, C, A3d, B3d, o0, r1, hit_axis=o0)
        if base is None or fit3 is None:
            continue
        gap0 = None if o0 is None else round(float(np.hypot(fit3['p0'][0] - o0[0],
                                                            fit3['p0'][1] - o0[1])), 2)
        z0 = round(float(fit3['p0'][2]), 2)
        mB_ = T3.metrics(base['p0'], base['v0'], base['beta'], rays[-1][0])
        m3 = T3.metrics(fit3['p0'], fit3['v0'], fit3['beta'], rays[-1][0])
        if A3d is not None:
            z_hits.append(('S', float(A3d[2]), float(mA), a.get('type')))
            n_both += B3d is not None
            n_one += B3d is None
        if B3d is not None:
            z_hits.append(('E', float(B3d[2]), float(mB), a.get('type')))
        mech = ('3D' if A3d is not None else '') + ('+hit' if o0 is not None else '') \
            + ('+3Dend' if B3d is not None else '')
        row = dict(frame0=fr0, frame1=fr1, type=a.get('type'),
                   owner=a.get('owner'), receiver=a.get('receiver'), mech=mech,
                   miss_start=None if not np.isfinite(mA) else round(mA, 2),
                   miss_end=None if not np.isfinite(mB) else round(mB, 2),
                   z_hit_start=None if A3d is None else round(float(A3d[2]), 2),
                   z_hit_end=None if B3d is None else round(float(B3d[2]), 2),
                   hit_gap=gap0,
                   hit_z=z0,
                   base=dict(v_first=round(mB_[0], 1), z_first=round(mB_[1], 2),
                             apex=round(mB_[2], 2), net_clear=mB_[3], land=mB_[4],
                             beta=round(base['beta'], 3), rms=round(base['rms'], 3)),
                   anch=dict(v_first=round(m3[0], 1), z_first=round(m3[1], 2),
                             apex=round(m3[2], 2), net_clear=m3[3], land=m3[4],
                             beta=round(fit3['beta'], 3), rms=round(fit3['rms'], 3)),
                   p0=[round(float(v), 3) for v in fit3['p0']],
                   v0=[round(float(v), 3) for v in fit3['v0']],
                   )
        rows.append(row)
        print(f"[{fr0:>5}-{fr1:>5}] {str(a.get('type'))[:10]:<10} {mech:<10} "
              f"hit轴距={row['hit_gap']}m z0={row['hit_z']} "
              f"zE={row['z_hit_end']} | "
              f"v {row['base']['v_first']:>5}->{row['anch']['v_first']:<5} "
              f"apex {row['base']['apex']}->{row['anch']['apex']} "
              f"net {row['base']['net_clear']}->{row['anch']['net_clear']} "
              f"land {row['base']['land']}->{row['anch']['land']} "
              f"b {row['base']['beta']}->{row['anch']['beta']} "
              f"rms {row['base']['rms']}->{row['anch']['rms']}")

    n = len(rows)
    print(f'\n--- 汇总 ({n} 弧) ---')
    print(f'锚定: 双端 {n_both} / 单端 {n_one} / 无 3D 锚 {n - n_both - n_one}')
    zs = [z for _, z, _, _ in z_hits]
    if zs:
        print(f'z_hit 分布: n={len(zs)} mean={np.mean(zs):.2f} '
              f'median={np.median(zs):.2f} p10={np.percentile(zs, 10):.2f} '
              f'p90={np.percentile(zs, 90):.2f} m')
    for tag in ('S', 'E'):
        zt = [z for t, z, _, _ in z_hits if t == tag]
        if zt:
            print(f'  {"首端(击球)" if tag == "S" else "末端(接球)"}: n={len(zt)} '
                  f'mean={np.mean(zt):.2f} median={np.median(zt):.2f} m')
    for k in ('v_first', 'apex', 'beta'):
        db = [r['base'][k] for r in rows]
        da = [r['anch'][k] for r in rows]
        print(f'{k}: 基线 mean={np.mean(db):.3f}  锚定 mean={np.mean(da):.3f}')
    zg = [r['hit_z'] for r in rows if r['hit_gap'] is not None]
    if zg:
        eng = sum(1 for r in rows if r['hit_gap'] is not None
                  and r['hit_gap'] <= REACH_HIT + 0.05)
        print(f'击球锚: 生效(拉到门) {eng}/{len(zg)}, 首观测高度 z0 mean={np.mean(zg):.2f} '
              f'median={np.median(zg):.2f} m')
    dv = [abs(r['base']['v_first'] - r['anch']['v_first']) for r in rows]
    print(f'|Δv_first| 基线vs锚定: mean={np.mean(dv):.1f} max={np.max(dv):.1f} m/s')
    vb = [np.hypot(*(np.array(r['base']['land']) - np.array(r['anch']['land']))) for r in rows
          if r['base']['land'] and r['anch']['land']]
    if vb:
        print(f'落点位移 基线vs锚定: mean={np.mean(vb):.2f} max={np.max(vb):.2f} m')
    out = os.path.join(args.results_dir, 'trajectory3d_anchor.json')
    with open(out, 'w') as fp:
        json.dump(dict(calib=calib_name, n_arcs=n, rows=rows,
                       z_hits=[dict(end=t, z=z, miss=m, type=tp) for t, z, m, tp in z_hits]),
                  fp, ensure_ascii=False, indent=1)
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()
