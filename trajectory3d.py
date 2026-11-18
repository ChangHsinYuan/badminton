"""第3级 3D 测量: 单目物理轨迹拟合 (出手球速 / 击球点 / 弧顶 / 过网余量 / 落点)

原理:
  相机已标定 (camera_selfcalib.py, calib3d.json, 工作系=y翻转右手系, z 向上)。
  每帧球观测给出一条 3D 射线: X_i 必在 X(t_i) = C + s_i·d_i 上 (2 约束/帧)。
  轨迹模型: 重力 + 二次空气阻力 dv/dt = -g·ẑ - β|v|v, 未知数 θ = (s0, v0, β) 共 5 个
  (首观测点参数化到其射线上: p0 = C + s0·d0, 少 2 个自由度)。
  两阶段: ①无阻力模型下方程对 (s0, v0) 线性 -> 闭式 lstsq 初值
          ②带阻力 RK4 (dt<=5ms) + scipy.least_squares (soft_l1 抗外点) 精修

物理自检:
  羽球终端速度 ~6.7 m/s -> β = g/v_t² ≈ 0.22 1/m。拟合 β 应聚在 0.2 附近;
  偏离大 = 弧太短 β 不可辨 或 弧段被错误合并。

已知边界:
  v_first 是"首检出帧"球速, 比真实出手球速低 (快球 0.03-0.1s 内衰减 5-15%);
  观测 <6 帧的弧不拟合 (杀球典型 3-5 帧 -> 缺席, 属球检出率瓶颈)。

用法:
  python trajectory3d.py --results-dir results/<name> [--min-pts 6]
输出:
  results/<name>/trajectory3d.json + 终端汇总
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

G = 9.8
FPS = 30.0
CL_W = 13.4                   # 工作系场地长 (y 翻转系)
NET_Y_W, NET_H = 6.7, 1.55    # 工作系网 y 与网高 (场地关于网对称, 6.7 不变)
DT_MAX = 0.005
RMS_LIMIT = 0.4                # m, 超过视为弧段合并/外点污染
Z_MIN = -0.5                   # m, 模拟下界 (允许略微入地用于求落点)


def load_scene(results_dir):
    with open(os.path.join(results_dir, 'calib3d.json')) as fp:
        c = json.load(fp)
    K = np.array(c['K']); R = np.array(c['R']); C = np.array(c['C_working'])
    f = c['f']
    ball = {}
    with open(os.path.join(results_dir, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            sc = d.get('shuttlecock') or {}
            if sc.get('image') is not None:
                ball[d['frame']] = sc['image']
    with open(os.path.join(results_dir, 'shot_analysis', 'shot_stats.json')) as fp:
        arcs = json.load(fp)['arcs']
    return K, R, C, f, ball, arcs


def ray_dir(K, R, C, uv):
    d = np.linalg.inv(R) @ np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    return d / np.linalg.norm(d)


def ray_dist(X, C, d):
    return float(np.linalg.norm(np.cross(X - C, d)))


def _acc(_p, _v, beta):
    return np.array([0.0, 0.0, -G]) - beta * np.linalg.norm(_v) * _v


def _rk4(p, v, beta, h):
    a1 = _acc(p, v, beta)
    p2 = p + 0.5 * h * v;  v2 = v + 0.5 * h * a1; a2 = _acc(p2, v2, beta)
    p3 = p + 0.5 * h * v2; v3 = v + 0.5 * h * a2; a3 = _acc(p3, v3, beta)
    p4 = p + h * v3;       v4 = v + h * a3;      a4 = _acc(p4, v4, beta)
    return p + h / 6 * (v + 2 * v2 + 2 * v3 + v4), v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)


def simulate(p0, v0, beta, taus):
    """精确在各 tau 时刻采样轨迹 (子步长 <= DT_MAX)"""
    taus = np.asarray(taus, float)
    order = np.argsort(taus)
    p, v = np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    t_now = 0.0
    out = np.zeros((len(taus), 3))
    for j in order:
        rem = float(taus[j]) - t_now
        if rem < 0:
            out[j] = p
            continue
        n = max(1, int(np.ceil(rem / DT_MAX)))
        h = rem / n
        for _ in range(n):
            p, v = _rk4(p, v, beta, h)
        t_now = float(taus[j])
        out[j] = p
    return out, p, v


def grid_init(rays, C):
    """粗网格几何枚举: X0 在首射线上扫深度 s0, 末射线扫深度 sN,
    无阻力弹道反推 v0, 取到各射线距离和最小的 (s0, v0) 作为初值。
    (闭式线性解在远端弧上病态: 射线近共线 -> s0 失辨 -> v0 爆炸)"""
    taus = np.array([r[0] for r in rays])
    dirs = np.stack([r[1] for r in rays])
    T = taus[-1]
    if T <= 0:
        return None
    zhat = np.array([0.0, 0.0, 1.0])
    s_grid = np.arange(2.0, 32.0, 1.0)
    e_grid = np.arange(2.0, 42.0, 1.5)
    X0 = C[None, :] + s_grid[:, None] * dirs[0][None, :]
    XN = C[None, :] + e_grid[:, None] * dirs[-1][None, :]
    V = (XN[:, None, :] - X0[None, :, :] - 0.5 * G * T * T * zhat[None, None, :]) / T
    P = X0[None, :, None, :] + V[:, :, None, :] * taus[None, None, :, None] \
        + (0.5 * G * taus * taus)[None, None, :, None] * zhat[None, None, None, :]
    dist = np.linalg.norm(np.cross(P - C[None, None, None, :], dirs[None, None, :, :]),
                          axis=3).mean(axis=2)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return np.array([s_grid[i], V[i, j, 0], V[i, j, 1], V[i, j, 2]])


def fit_arc(rays, C, anchor_start=None, anchor_end=None):
    """rays: [(tau, d̂)]; anchor_*: 工作系脚位 (x,y) 或 None
    锚点破单目'慢近/快远'简并: 末端球已减速(~7m/s), 末点应距接球者 <1.5m(强);
    首点距击球者 <3m(弱, 快球首检可能已飞出数米)"""
    taus = np.array([r[0] for r in rays])
    dirs = np.stack([r[1] for r in rays])
    if taus[-1] <= 0:
        return None
    init = grid_init(rays, C)
    if init is None or not np.isfinite(init).all():
        return None
    zhat = np.array([0.0, 0.0, 1.0])

    def resid0(th):     # 阶段1: 无阻力 (4 参数, 无积分, 快且稳)
        s0, vx, vy, vz = th
        X = (C + s0 * dirs[0])[None, :] + np.array([vx, vy, vz])[None, :] * taus[:, None] \
            + (0.5 * G * taus * taus)[:, None] * zhat[None, :]
        res = [ray_dist(X[i], C, dirs[i]) for i in range(len(taus))]
        if anchor_end is not None:
            gap = float(np.hypot(X[-1, 0] - anchor_end[0], X[-1, 1] - anchor_end[1]))
            res.append(3.0 * max(0.0, gap - 1.5))
        if anchor_start is not None:
            gap = float(np.hypot(X[0, 0] - anchor_start[0], X[0, 1] - anchor_start[1]))
            res.append(1.0 * max(0.0, gap - 3.0))
        return res

    try:
        sol0 = least_squares(resid0, init,
                             bounds=([0.5, -150, -150, -150], [60, 150, 150, 150]),
                             max_nfev=40)
        x1 = np.append(sol0.x, 0.15)
    except Exception:
        x1 = np.append(init, 0.15)

    def resid(th):      # 阶段2: 带阻力 RK4
        s0, vx, vy, vz, beta = th
        p0 = C + s0 * dirs[0]
        pos, _, _ = simulate(p0, np.array([vx, vy, vz]), beta, taus)
        res = [ray_dist(pos[i], C, dirs[i]) for i in range(len(taus))]
        if anchor_end is not None:
            gap = float(np.hypot(pos[-1, 0] - anchor_end[0], pos[-1, 1] - anchor_end[1]))
            res.append(3.0 * max(0.0, gap - 1.5))
        if anchor_start is not None:
            gap = float(np.hypot(p0[0] - anchor_start[0], p0[1] - anchor_start[1]))
            res.append(1.0 * max(0.0, gap - 3.0))
        return res

    lo = np.array([0.5, -120, -120, -120, 0.0])
    hi = np.array([60.0, 120, 120, 120, 0.6])
    try:
        sol = least_squares(resid, np.clip(x1, lo + 1e-6, hi - 1e-6),
                            bounds=(lo, hi), loss='soft_l1', f_scale=0.25, max_nfev=60)
    except Exception:
        return None
    if not np.isfinite(sol.x).all():
        return None
    s0, vx, vy, vz, beta = sol.x
    p0 = C + s0 * dirs[0]
    v0 = np.array([vx, vy, vz])
    pos, _, _ = simulate(p0, v0, beta, taus)
    res = np.array([ray_dist(pos[i], C, dirs[i]) for i in range(len(taus))])
    return dict(p0=p0, v0=v0, beta=float(beta), rms=float(np.sqrt((res ** 2).mean())))


def speed_at_owner_axis(p0, v0, beta, owner_w, t_max=0.8):
    """从首检出状态沿拟合 ODE 反推, 取路径上距击球者轴最近的点。
    返回 (出手球速, 该点高度 z, 反推用时) 或 None (0.8s 内最近距离 >0.8m)。
    反向时间导数: dp/d(-t) = -v, dv/d(-t) = +g·ẑ + β|v|v"""
    def f(_p, _v):
        return -_v, np.array([0.0, 0.0, G]) + beta * np.linalg.norm(_v) * _v

    p, v = np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    t, h = 0.0, 0.01
    best = None
    while t < t_max:
        gap = float(np.hypot(p[0] - owner_w[0], p[1] - owner_w[1]))
        if best is None or gap < best[0]:
            best = (gap, float(np.linalg.norm(v)), float(p[2]), t)
        if gap <= 0.3:
            break
        k1p, k1v = f(p, v)
        k2p, k2v = f(p + 0.5 * h * k1p, v + 0.5 * h * k1v)
        k3p, k3v = f(p + 0.5 * h * k2p, v + 0.5 * h * k2v)
        k4p, k4v = f(p + h * k3p, v + h * k3v)
        p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
    if best is None or best[0] > 0.8:
        return None
    return best[1], best[2], best[3]


def metrics(p0, v0, beta, tau_max):
    """稠密采样: 出手状态/弧顶/过网余量/落点 (落点/过网在 H.npy 系输出)"""
    dense = np.arange(0.0, tau_max + 0.01, 0.01)
    pos, p_end, _ = simulate(p0, v0, beta, dense)
    apex = float(pos[:, 2].max())
    z_first = float(p0[2])
    v_first = float(np.linalg.norm(v0))
    net_clear = None
    yy = pos[:, 1]
    cross = np.where((yy[:-1] - NET_Y_W) * (yy[1:] - NET_Y_W) <= 0)[0]
    if len(cross):
        i = cross[0]
        w = abs(yy[i] - NET_Y_W) / max(abs(yy[i] - NET_Y_W) + abs(yy[i + 1] - NET_Y_W), 1e-9)
        z_net = pos[i, 2] + w * (pos[i + 1, 2] - pos[i, 2])
        net_clear = float(z_net - NET_H)
    land = None
    below = np.where(pos[:, 2] <= 0)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        w = pos[i - 1, 2] / max(pos[i - 1, 2] - pos[i, 2], 1e-9)
        pt = pos[i - 1] + w * (pos[i] - pos[i - 1])
        land = [round(float(pt[0]), 2), round(13.4 - float(pt[1]), 2)]
    return v_first, z_first, apex, net_clear, land


def main():
    ap = argparse.ArgumentParser(description='单目物理轨迹拟合(出手球速/弧顶/落点)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--min-pts', type=int, default=6)
    args = ap.parse_args()

    K, R, C, f, ball, arcs = load_scene(args.results_dir)
    court = {}
    with open(os.path.join(args.results_dir, 'detections.jsonl')) as fp:
        for line in fp:
            d = json.loads(line)
            court[d['frame']] = {s: p['court'] for s, p in d['players'].items()
                                 if p and p.get('court')}
    print(f'=== 第3级 物理轨迹拟合: {args.results_dir} ===')
    results, skipped = [], {'obs不足': 0, '拟合失败': 0}
    for a in arcs:
        fr0, fr1 = a['frame0'], a['frame1']
        obs = [(fr, ball[fr]) for fr in range(fr0, fr1 + 1) if fr in ball]
        if len(obs) < args.min_pts:
            skipped['obs不足'] += 1
            continue
        rays = [((fr - fr0) / FPS, ray_dir(K, R, C, uv)) for fr, uv in obs]

        def pos_w(side, fr):
            p = court.get(fr, {}).get(side)
            return None if p is None else (float(p[0]), CL_W - float(p[1]))

        owner = a.get('owner')
        receiver = a.get('receiver')
        anchor_start = pos_w(owner, fr0) if owner else None
        anchor_end = pos_w(receiver, fr1) if receiver else None
        fit = fit_arc(rays, C, anchor_start, anchor_end)
        if fit is None:
            skipped['拟合失败'] += 1
            continue
        v_first, z_first, apex, net_clear, land = metrics(
            fit['p0'], fit['v0'], fit['beta'], rays[-1][0])
        v_owner = z_owner = t_back = None
        if anchor_start is not None:
            r = speed_at_owner_axis(fit['p0'], fit['v0'], fit['beta'], anchor_start)
            if r is not None:
                v_owner, z_owner, t_back = r
        depth = float(np.linalg.norm(fit['p0'] - C))
        results.append(dict(
            frame0=fr0, frame1=fr1, n_obs=len(obs), owner=owner,
            receiver=receiver, type=a.get('type'), conf=a.get('conf'),
            v_first=round(v_first, 1), v_kmh=round(v_first * 3.6),
            v_owner=None if v_owner is None else round(v_owner, 1),
            v_owner_kmh=None if v_owner is None else round(v_owner * 3.6),
            z_owner=None if z_owner is None else round(z_owner, 2),
            t_back=None if t_back is None else round(t_back, 2),
            z_first=round(z_first, 2), apex=round(apex, 2),
            net_clear=None if net_clear is None else round(net_clear, 2),
            land=land, beta=round(fit['beta'], 3),
            p0=[round(v, 3) for v in fit['p0']],
            v0=[round(v, 3) for v in fit['v0']],
            rms=round(fit['rms'], 3), rms_px=round(fit['rms'] * f / max(depth, 1), 1),
            suspect=fit['rms'] > RMS_LIMIT,
        ))

    ok = [r for r in results if not r['suspect']]
    n_attempt = len(arcs) - skipped['obs不足']
    print(f'弧 {len(arcs)} | 观测≥{args.min_pts} 进入拟合: {n_attempt} '
          f'| 成功 {len(results)} | 失败 {n_attempt - len(results)} '
          f'| 可信(RMS≤{RMS_LIMIT}m) {len(ok)} | 观测不足跳过 {skipped["obs不足"]}')
    if not ok:
        print('无可信拟合, 退出')
        return

    betas = np.array([r['beta'] for r in ok])
    print(f'\n阻力系数 β: 中位 {np.median(betas):.2f}  四分位 '
          f'[{np.percentile(betas, 25):.2f}, {np.percentile(betas, 75):.2f}]  '
          f'(物理参考 ≈0.22, 羽球终端速度 6.7m/s)')

    by_type = {}
    for r in ok:
        by_type.setdefault(r['type'], []).append(r)
    print(f'\n{"类型":<14}{"n":>4}{"首帧m/s":>9}{"出手m/s":>9}{"出手km/h":>10}{"首帧z":>7}{"反推z":>7}{"过网余量":>9}')
    for tp, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        v = np.array([r['v_first'] for r in rs])
        vo = [r['v_owner'] for r in rs if r['v_owner'] is not None]
        vos = f'{np.median(vo):.1f}' if vo else '-'
        vok = f'{np.median(vo) * 3.6:.0f}' if vo else '-'
        zo = [r['z_owner'] for r in rs if r['z_owner'] is not None]
        zos = f'{np.median(zo):.2f}' if zo else '-'
        z = np.array([r['z_first'] for r in rs])
        nc = [r['net_clear'] for r in rs if r['net_clear'] is not None]
        ncs = f'{np.median(nc):+.2f}m' if nc else '-'
        print(f'{str(tp):<14}{len(rs):>4}{np.median(v):>9.1f}{vos:>9}{vok:>10}'
              f'{np.median(z):>7.2f}{zos:>7}{ncs:>9}')

    top = sorted(ok, key=lambda r: -(r['v_owner'] or 0))[:5]
    print('\n估算出手球速最快 5 球:')
    for r in top:
        vo = f"{r['v_owner']:.1f}m/s ({r['v_owner_kmh']}km/h)" if r['v_owner'] else '-'
        zo = f"{r['z_owner']:.2f}m" if r['z_owner'] is not None else '-'
        tb = f"反推{r['t_back']:.2f}s" if r['t_back'] is not None else ''
        print(f"  #{r['frame0']:>5} 出手~{vo:<22} 击球点z={zo:<7} {tb:<10} "
              f"首帧{r['v_first']}m/s {r['type']} owner={r['owner']} rms={r['rms']}m")

    out = dict(min_pts=args.min_pts, rms_limit=RMS_LIMIT, results=results,
               skipped=skipped)
    with open(os.path.join(args.results_dir, 'trajectory3d.json'), 'w') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print(f'\n已保存: {os.path.join(args.results_dir, "trajectory3d.json")}')


if __name__ == '__main__':
    main()
