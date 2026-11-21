#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""档2 双平面自标定: 地面单应 H1 + 球网平面观测 (网带线 + 网柱线 + 竖直灭点)

相对 camera_selfcalib.py (单平面 + 主点居中假设) 的升级:
  - 网面单应的正确形式(平面->图像): H2 = K[r1, r3, t + Y0*r2], 网面局部坐标 (U,V)=(X,Z)
    (通用式 H = K(R + t n^T/d) 是图像到图像的诱导单应, 不适用于平面->图像分解)
  - 观测: 网带线 (z=1.55m) + 两根网柱线 (U=0 / U=6.1) + (可选)网面竖线 -> 竖直灭点 v_z
  - 内参 (f, cx, cy) 非线性最小二乘联合解出, 主点不再强制居中
  - 尺度仍由米制场地锚定 (6.1 x 13.4); 方形像素默认假设 (--free-xy 仅诊断)
  - 净新增信息 ~ 网带端点 + 竖直方向 4 个像素数, 属"假设松绑", 非全自由标定
  - 左右镜像歧义依旧无解 (网关于 x=3.05 对称), 只保证 C_z>0 朝向类

用法:
  python camera_selfcalib2.py --results-dir results/<name> [--n-frames 11] [--free-xy] [--k1]
输出:
  results/<name>/net_calib_debug.jpg
  results/<name>/calib3d_2plane.json   (不覆盖 calib3d.json)
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_selfcalib import CL, CW, NET_H, NET_Y, decompose, net_tape_offset, project

TY_FLIP = np.array([[1, 0, 0], [0, -1, CL], [0, 0, 1]], float)
TAPE_WIN = 55        # 网带行搜索半窗 (px)
POST_HALF = 14       # 网柱列搜索半宽 (px)
POST_UP_EXT = 15     # 网柱在网带高以上的延伸搜索 (px)


# ---------------------------------------------------------------- 基础工具

def apply_G(G, pts):
    """地面 2D 点(米) -> 图像像素 (经 G=court->image 单应)"""
    pts = np.atleast_2d(np.asarray(pts, float))
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ G.T
    return h[:, :2] / h[:, 2:3]


def group_runs(rows, gap=3):
    if len(rows) == 0:
        return []
    runs, s, p = [], int(rows[0]), int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r - p > gap:
            runs.append((s, p))
            s = r
        p = r
    runs.append((s, p))
    return runs


def robust_line(pts, tol=2.5, iters=800):
    """RANSAC + TLS 精化直线拟合 l=(a,b,c), a^2+b^2=1.
    返回 (l, inlier_mask, rms, n_in) 或 None"""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 6:
        return None
    rng = np.random.default_rng(0)
    best_cnt, best_in = 0, None
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        d = pts[j] - pts[i]
        nd = float(np.hypot(d[0], d[1]))
        if nd < 10:
            continue
        nv = np.array([-d[1], d[0]]) / nd
        cc = -float(nv @ pts[i])
        inl = np.abs(pts @ nv + cc) < tol
        if int(inl.sum()) > best_cnt:
            best_cnt, best_in = int(inl.sum()), inl
    if best_in is None or best_cnt < 6:
        return None
    keep = best_in
    nv = cc = None
    for _ in range(3):
        P = pts[keep]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        d = vt[0]
        nv = np.array([-d[1], d[0]])
        cc = -float(nv @ c)
        res = np.abs(pts @ nv + cc)
        nk = res < max(tol, 1.5 * np.median(res[keep]))
        if nk.sum() < 6 or np.array_equal(nk, keep):
            break
        keep = nk
    l = np.array([nv[0], nv[1], cc])
    l /= np.hypot(l[0], l[1])
    res = np.abs(pts @ l[:2] + l[2])
    rms = float(np.sqrt((res[keep] ** 2).mean())) if keep.sum() else 999.0
    return l, keep, rms, int(keep.sum())


def line_isect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def load_G(results_dir, W, Hh):
    """读 H.npy -> G, 选 C_z>0 朝向类, 返回 (orient, G, calib0)"""
    Hmat = np.load(os.path.join(results_dir, 'H.npy'))
    G0 = np.linalg.inv(Hmat)
    cands = []
    for name, Gc in (('原朝向', G0), ('y翻转', G0 @ TY_FLIP)):
        try:
            c = decompose(Gc, W / 2.0, Hh / 2.0)
            cands.append((name, Gc, c))
        except RuntimeError:
            continue
    if not cands:
        raise RuntimeError('两个朝向类均无法分解')
    up = [c for c in cands if c[2]['C'][2] > 0.3]
    pool = up if up else cands
    pool.sort(key=lambda x: abs(x[2]['C'][2]))
    return pool[0]


# ---------------------------------------------------------------- 帧累积与检测

def load_frames(video_path, n=11):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频: ' + video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.05, total * 0.95, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, fr = cap.read()
        if ret:
            frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError('未能读取任何帧: ' + video_path)
    return frames


def accumulate(frames):
    """静态场景时间累积: 白色计数 + 水平梯度幅值 (运动物体被平均掉)"""
    white_cnt = None
    sob_acc = None
    for fr in frames:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)).astype(np.float32)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sob = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        white_cnt = white if white_cnt is None else white_cnt + white
        sob_acc = sob if sob_acc is None else sob_acc + sob
    return white_cnt, sob_acc, len(frames)


def net_quad_mask(P0, G, W, Hh, shrink=6):
    tb = apply_G(G, [(0, NET_Y), (CW, NET_Y)])                      # 柱底(地面点, 精确)
    tt = project(P0, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)])[0]    # 网带高(近似)
    poly = np.array([tb[0], tt[0], tt[1], tb[1]], np.float32)
    c = poly.mean(0)
    h = np.linalg.norm(poly[1] - poly[0])
    poly_s = c + (poly - c) * (1.0 - shrink / max(h, 1.0))
    m = np.zeros((Hh, W), np.uint8)
    cv2.fillPoly(m, [poly_s.astype(np.int32)], 1)
    return m, poly_s.astype(np.int32)


def detect_tape(white_cnt, nf, P0, W, Hh, half=60):
    """网带检测: 预测线走廊内按行偏移统计静态白色支持, 取峰后逐列亚像素精化.
    返回 (pts (N,2) [u,v], diag)"""
    mask = white_cnt >= max(2.0, 0.55 * nf)
    uv, _ = project(P0, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)])
    (u1, v1), (u2, v2) = uv[0], uv[1]
    if not np.isfinite([u1, v1, u2, v2]).all() or not (0 <= u1 < W and 0 <= u2 < W):
        return np.zeros((0, 2)), '投影出界'
    us = np.arange(max(0, int(u1)), min(W, int(u2) + 1))
    if len(us) < 50:
        return np.zeros((0, 2)), '投影跨度过小'
    sup = np.zeros(2 * half + 1)
    vpred = np.zeros(W)
    vpred[:] = np.nan
    for u in us:
        t = (u - u1) / (u2 - u1)
        vp = v1 + t * (v2 - v1)
        vpred[u] = vp
        for k, dv in enumerate(range(-half, half + 1)):
            v = int(round(vp)) + dv
            if 0 <= v < Hh and mask[v, u]:
                sup[k] += 1
    cand, s = [], sup.copy()
    while True:
        k = int(np.argmax(s))
        if s[k] <= 0:
            break
        cand.append((k - half, s[k]))
        s[max(0, k - 6):k + 7] = 0
        if len(cand) >= 5:
            break
    if not cand:
        return np.zeros((0, 2)), '走廊内无静态白色像素'
    mx = cand[0][1]
    near = [c for c in cand if c[1] >= 0.6 * mx]
    dstar = min(near, key=lambda c: abs(c[0]))[0]
    pts = []
    for u in us:
        v0 = int(round(vpred[u])) + dstar
        rows = [v0 + d for d in range(-4, 5)
                if 0 <= v0 + d < Hh and mask[v0 + d, u]]
        if len(rows) >= 2:
            pts.append((float(u), float(np.mean(rows))))
    diag = (f'走廊峰 d={dstar:+d}px 支持={mx:.0f}列 '
            f'候选峰={[(int(a), int(b)) for a, b in cand]}')
    return np.array(pts, float).reshape(-1, 2), diag


def detect_junctions(white_cnt, nf, tape_line, P0, W):
    """网带端点扫描 -> 带柱交点 (u,v).
    实测剖面: 实心白带 -> 4-25px 暗带 (网柱) -> 柱外零星白.
    从预测端点沿强静态白 (>=0.7nf, 容忍 3px 缺口) 走到外端,
    紧随暗带 (宽 4-25px) 的中心 = 柱轴即交点; 无暗带则取白带外端.
    返回 (juncs {side: (u,v)}, diags [str])"""
    strong = white_cnt >= max(3.0, 0.70 * nf)
    juncs, diags = {}, []
    for U, su in ((0.0, -1), (CW, +1)):
        p0 = project(P0, [(U, NET_Y, NET_H)])[0][0]
        if not np.isfinite(p0).all() or tape_line is None:
            diags.append(f'U={U:.1f}: 预测出界')
            continue
        u0 = int(round(float(p0[0])))

        def vline(u):
            return -(tape_line[0] * u + tape_line[2]) / tape_line[1]

        rng = range(u0, max(1, u0 - 200), -1) if su < 0 else range(u0, min(W - 2, u0 + 200))

        def is_strong(u):
            v = int(round(vline(u)))
            return strong[v, u] or strong[max(0, v - 1), u] or strong[min(719, v + 1), u]

        # 1) 从 u0 沿强白走到外端 (容忍 3px 缺口)
        u_end, gap, seen_white = u0, 0, False
        for u in rng:
            if is_strong(u):
                u_end, gap, seen_white = u, 0, True
            elif seen_white:
                gap += 1
                if gap > 3:
                    break
        if not seen_white:
            diags.append(f'U={U:.1f}: 预测端点处无强静态白 (遮挡?), 预测 u={u0}')
            continue
        # 2) 外端之后的暗带 -> 下一处强白
        u_next = None
        u = u_end + su
        while 1 <= u <= W - 2:
            if is_strong(u):
                u_next = u
                break
            u += su
        if u_next is None:
            diags.append(f'U={U:.1f}: 白带外端 u={u_end} 之后无白 (遮挡/出画), 交点不可观')
            continue
        dk = abs(u_next - u_end) - 1
        if not (4 <= dk <= 25):
            diags.append(f'U={U:.1f}: 白带外端 u={u_end} 后暗带宽{dk}异常 (遮挡/网面延续), 交点不可观')
            continue
        u_j = 0.5 * (u_end + u_next)
        # 柱下延伸验证: 交点列在带线下方 20/40/80px 处应为静态暗 (真柱延续到地)
        v_j0 = int(round(float(vline(u_j))))
        ok_below = all(0 <= v_j0 + dv < white_cnt.shape[0]
                       and white_cnt[v_j0 + dv, int(round(u_j))] < 0.3 * nf
                       for dv in (20, 40, 80))
        if not ok_below:
            diags.append(f'U={U:.1f}: 交点 u={u_j:.1f} 带线下方非静态暗带, 疑似背景, 不可观')
            continue
        if abs(u_j - u0) > 65:
            diags.append(f'U={U:.1f}: 交点 u={u_j:.1f} 偏离预测 {u_j - u0:+.0f}px, 判为背景, 不可观')
            continue
        v_j = float(vline(u_j))
        juncs[U] = (float(u_j), v_j)
        diags.append(f'U={U:.1f}: 白带外端 u={u_end}, 暗带[{min(u_end, u_next) + 1},'
                     f'{max(u_end, u_next) - 1}]宽{dk}, 交点=({u_j:.1f},{v_j:.1f})')
    return juncs, diags


def detect_post(sob_acc, P0, G, side_x, W, Hh, half=40, white_cnt=None, nf=1):
    """网柱检测: 主判据=静态深色带 (white_cnt 低, 宽 3-18px, 跨行一致);
    网线是白线/挡板过宽/运动员瞬态, 均被排除. white_cnt=None 时回退
    Sobelx 走廊峰+近竖直斜率先验. 返回 (pts (N,2) [u,v], diag)"""
    pb = apply_G(G, [(side_x, NET_Y)])[0]
    pt = project(P0, [(side_x, NET_Y, NET_H)])[0][0]
    if not np.isfinite(np.concatenate([pb, pt])).all():
        return np.zeros((0, 2)), '投影出界'
    d = pt - pb
    nd = np.linalg.norm(d)
    if nd < 5:
        return np.zeros((0, 2)), '投影线过短'
    p1 = pt + d / nd * POST_UP_EXT      # 网柱可能高出网带
    p2 = pb
    d2 = p2 - p1
    n = np.array([-d2[1], d2[0]], float)
    nn = np.linalg.norm(n)
    a, b, c = n[0] / nn, n[1] / nn, -float(n @ p1) / nn
    rows = list(range(max(0, int(p1[1])), min(Hh - 1, int(p2[1]) + 4)))
    if len(rows) < 30:
        return np.zeros((0, 2)), '投影行数过少'
    v_lo = max(0, int(pt[1]) + 2)
    v_hi = min(Hh - 1, int(pb[1]) + 4)
    slope_pred = float((pb[0] - pt[0]) / max(pb[1] - pt[1], 1.0))

    def fit_slope_prior(pts, tol=2.5):
        """u = al*v + be, 斜率先验 ±3°, RANSAC(网格) + TLS; 返回 (al,be,n_in,rms)"""
        bj = None
        for al in np.arange(slope_pred - 0.052, slope_pred + 0.052 + 1e-9, 0.004):
            be = float(np.median(pts[:, 0] - al * pts[:, 1]))
            nin = int(np.sum(np.abs(pts[:, 0] - al * pts[:, 1] - be) < tol))
            if bj is None or nin > bj[0]:
                bj = (nin, float(al), be)
        nin, al, be = bj
        if nin < 8:
            return None
        P = pts[np.abs(pts[:, 0] - al * pts[:, 1] - be) < tol]
        A = np.stack([P[:, 1], np.ones(len(P))], 1)
        sol, *_ = np.linalg.lstsq(A, P[:, 0], rcond=None)
        al2, be2 = float(sol[0]), float(sol[1])
        if abs(al2 - slope_pred) > 0.06:
            return None
        res = P[:, 0] - (al2 * P[:, 1] + be2)
        return al2, be2, len(P), float(np.sqrt((res ** 2).mean()))

    # --- 主路径: 静态深色带 ---
    if white_cnt is not None:
        dark = white_cnt < max(2.0, 0.30 * nf)
        HW = 60
        hist = np.zeros(2 * HW + 1)
        row_runs = {}
        for v in range(v_lo, v_hi + 1):
            if abs(a) < 1e-6:
                continue
            u0 = int(round(-(b * v + c) / a))
            lo, hi = max(1, u0 - HW), min(W - 2, u0 + HW)
            dm = dark[v, lo:hi + 1]
            k = 0
            while k < len(dm):
                if dm[k]:
                    k2 = k
                    while k2 + 1 < len(dm) and dm[k2 + 1]:
                        k2 += 1
                    wdt = k2 - k + 1
                    if 3 <= wdt <= 18:
                        cen = lo + 0.5 * (k + k2)
                        du = cen - u0
                        if abs(du) <= HW - 2:
                            hist[int(round(du)) + HW] += 1
                            row_runs.setdefault(v, []).append(cen)
                    k = k2 + 1
                else:
                    k += 1
        if hist.max() >= 25:
            sm = np.convolve(hist, np.ones(3) / 3.0, 'same')
            cands = [k for k in range(2, 2 * HW - 1)
                     if sm[k] >= sm[k - 1] and sm[k] >= sm[k + 1] and sm[k] > 0.3 * sm.max()]
            cands.sort(key=lambda k_: -sm[k_])
            best = None
            for kd in cands[:5]:
                du0 = kd - HW
                pts = []
                for v, cens in row_runs.items():
                    uu0 = -(b * v + c) / a
                    oks = [cen for cen in cens if abs(cen - uu0 - du0) <= 8]
                    if not oks:
                        continue
                    cen = min(oks, key=lambda t_: abs(t_ - uu0 - du0))
                    # 亚像素: 暗带两侧 |Sobelx| 边缘峰中点
                    i0 = int(round(cen - 8))
                    i1 = int(round(cen + 8))
                    if i0 < 1 or i1 > W - 2:
                        pts.append((cen, float(v)))
                        continue
                    prof = sob_acc[v, i0:i1 + 1]
                    m = float(np.median(prof))
                    ks = [j for j in range(1, len(prof) - 1)
                          if prof[j] > max(2.5 * m, 60) and prof[j] >= prof[j - 1] and prof[j] > prof[j + 1]]
                    if len(ks) >= 2:
                        e1 = ks[0] + i0
                        e2 = ks[-1] + i0
                        pts.append((0.5 * (e1 + e2), float(v)))
                    else:
                        pts.append((cen, float(v)))
                if len(pts) < 25:
                    continue
                pts = np.array(pts, float)
                fitr = fit_slope_prior(pts)
                if fitr is None:
                    continue
                al2, be2, nin, rms = fitr
                nn2 = np.hypot(1.0, al2)
                line = (-1.0 / nn2, al2 / nn2, be2 / nn2)
                if best is None or nin > best[0]:
                    best = (nin, rms, line, pts, du0)
            if best is not None:
                nin, rms, l1, pts, du0 = best
                tilt_dev = np.degrees(np.arctan(l1[1] / abs(l1[0])) - np.arctan(slope_pred))
                diag = (f'暗带峰 du={du0:+d}px (支持行 {int(hist.max())}), 内点 {nin}, '
                        f'rms={rms:.2f}px, 倾角偏差 {tilt_dev:+.1f}°')
                return pts, diag
        # 暗带不可得 -> 回退
        diag0 = '暗带不足(峰支持=%d<25), 回退Sobelx' % int(hist.max()) if hist.max() > 0 else '无静态暗带'
    else:
        diag0 = '无white_cnt, 回退Sobelx'

    # --- 回退: Sobelx 走廊峰 (仅网区行) + 近竖直斜率先验 ---
    sup = np.zeros(2 * half + 1)
    for v in rows:
        if v < v_lo or abs(a) < 1e-6:
            continue
        u0 = int(round(-(b * v + c) / a))
        for k, du in enumerate(range(-half, half + 1)):
            u = u0 + du
            if 1 <= u < W - 2:
                sup[k] += sob_acc[v, u]
    if sup.max() <= 0:
        return np.zeros((0, 2)), diag0 + '; 走廊内无梯度'
    cands = []
    for k in range(1, 2 * half):
        if sup[k] > sup[k - 1] and sup[k] >= sup[k + 1] and sup[k] > 0.25 * sup.max():
            cands.append(k - half)
    if not cands:
        cands = [int(np.argmax(sup)) - half]
    cands.sort(key=lambda d_: -sup[d_ + half])

    def refine(line_abc, win, rows_=None, thr_rel=3.0, thr_abs=60.0):
        aa, bb, cc = line_abc
        pts = []
        for v in (rows_ if rows_ is not None else rows):
            if abs(aa) < 1e-6:
                continue
            u_pred = -(bb * v + cc) / aa
            u0 = int(round(u_pred))
            lo, hi = u0 - win, u0 + win + 1
            if lo < 1 or hi > W - 2:
                continue
            prof = sob_acc[v, lo:hi]
            k = int(np.argmax(prof))
            med = float(np.median(prof))
            if prof[k] < max(thr_rel * med, thr_abs):
                continue
            if 0 < k < len(prof) - 1:
                den = prof[k - 1] - 2 * prof[k] + prof[k + 1]
                off = 0.5 * (prof[k - 1] - prof[k + 1]) / den if abs(den) > 1e-9 else 0.0
                off = float(np.clip(off, -1.0, 1.0))
            else:
                off = 0.0
            pts.append((lo + k + off, float(v)))
        return np.array(pts, float).reshape(-1, 2)

    best = None
    for du in cands:
        l0 = (a, b, c - du * a)
        pts = refine(l0, win=12)
        if len(pts) < 20:
            continue
        fitr = fit_slope_prior(pts)
        if fitr is None:
            continue
        al2, be2, nin, rms = fitr
        if nin < 30:
            continue
        nn2 = np.hypot(1.0, al2)
        line = (-1.0 / nn2, al2 / nn2, be2 / nn2)
        if best is None or nin > best[0]:
            best = (nin, rms, line, pts, du)
    if best is None:
        return np.zeros((0, 2)), diag0 + '; 无近竖直结构'
    nin, rms, l1, pts, du = best
    tilt_dev = np.degrees(np.arctan(l1[1] / abs(l1[0])) - np.arctan(slope_pred))
    diag = (diag0 + f'; Sobelx峰 d={du:+d}px, 内点 {nin}, rms={rms:.2f}px, 倾角偏差 {tilt_dev:+.1f}°')
    return pts, diag


def detect_mesh_lines(sob_acc, quad_mask, W, Hh, max_lines=12):
    """网面竖线 (Hough, 可得则用): 返回 [(l, length), ...]"""
    region = sob_acc.copy()
    region[quad_mask == 0] = 0
    vals = region[quad_mask > 0]
    if vals.size < 500:
        return []
    thr = max(40.0, float(np.percentile(vals, 90)))
    binary = ((region > thr) * 255).astype(np.uint8)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=40,
                            minLineLength=25, maxLineGap=3)
    if lines is None:
        return []
    cands = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = abs(float(x2 - x1)), abs(float(y2 - y1))
        if dy < 20 or dx > 0.25 * dy:
            continue
        n = np.array([-(y2 - y1), x2 - x1], float)
        nn = np.linalg.norm(n)
        l = np.array([n[0] / nn, n[1] / nn, -(n[0] * x1 + n[1] * y1) / nn])
        cands.append((l, float(np.hypot(dx, dy))))
    cands.sort(key=lambda t: -t[1])
    out = []
    for l, length in cands:
        dup = any(abs(l[0] * l2[0] + l[1] * l2[1]) > 0.999 and abs(l[2] - l2[2]) < 8
                  for l2, _ in out)
        if not dup:
            out.append((l, length))
        if len(out) >= max_lines:
            break
    return out


def vanish_z(vlines):
    """竖直灭点: 加权直线的最小特征向量; 返回 (v_z 齐次, 谱散度) 或 (None, None)"""
    if len(vlines) < 2:
        return None, None
    M = np.zeros((3, 3))
    for l, w in vlines:
        M += w * np.outer(l, l)
    evals, evecs = np.linalg.eigh(M)
    v = evecs[:, 0]
    if v[2] < 0:
        v = -v
    return v, float(evals[0] / max(evals[1], 1e-12))


# ---------------------------------------------------------------- 优化

def rt_from(G, fx, fy, cx, cy):
    """K + 地面单应 -> (K, R, t) 及正交性诊断量 (G 尺度不变)"""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ G
    g1, g2, g3 = M[:, 0], M[:, 1], M[:, 2]
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    lam = 0.5 * (n1 + n2)
    f_avg = 0.5 * (fx + fy)
    c_ang = float(g1 @ g2 / (n1 * n2))
    ndiff = float((n1 - n2) / lam)
    r1 = g1 / n1
    r2 = g2 - (g2 @ r1) * r1
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    t = g3 / lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)
    return K, R, t, f_avg, c_ang, ndiff


def seg_line(P, pts3d):
    """投影 3D 线段 -> 齐次直线 (a,b,c), a^2+b^2=1"""
    uv, dep = project(P, pts3d)
    if not np.isfinite(uv).all() or (dep <= 1e-6).any():
        return None
    p1, p2 = uv
    d = p2 - p1
    n = np.array([-d[1], d[0]], float)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    return np.array([n[0] / nn, n[1] / nn, -float(n @ p1) / nn])


def solve_f(G, cx, cy, W):
    """给定主点, 由地面正交性约束一维解出 f (软最优: 最小化 c_ang^2+ndiff^2).
    返回 (f, g1_px, g2_px): 后两者为最优 f 处的地面张力 (px 单位, = f·c_ang / f·ndiff)"""
    def cost(f):
        _, _, _, f_avg, c_ang, ndiff = rt_from(G, f, f, cx, cy)
        return (f_avg * c_ang) ** 2 + (f_avg * ndiff) ** 2

    fs = np.linspace(0.3 * W, 8.0 * W, 120)
    cs = [cost(x) for x in fs]
    i = int(np.argmin(cs))
    lo = fs[max(0, i - 2)]
    hi = fs[min(len(fs) - 1, i + 2)]
    r = minimize_scalar(cost, bounds=(lo, hi), method='bounded',
                        options=dict(xatol=0.01))
    _, _, _, f_avg, c_ang, ndiff = rt_from(G, r.x, r.x, cx, cy)
    return float(r.x), float(f_avg * c_ang), float(f_avg * ndiff)


def build_res(G, obs):
    """像素单位残差. 嵌套参数化: 外层 (cx, cy), f 由地面正交性约束解出 (不进损失);
    地面约束与网观测的张力体现在 f1/f2 分叉诊断, 而非被鲁棒损失丢弃.
    free-xy 诊断模式例外: 4 参数全自由, 地面约束作为普通残差参与."""
    tape = obs['tape']
    posts = obs['posts']
    vlines = obs['vlines']
    free_xy = obs['free_xy']
    fix_cx = obs['fix_cx']
    fit_k1 = obs['fit_k1']

    def undist(p, k1, f, cx, cy):
        if abs(k1) < 1e-12:
            return p
        q = p - np.array([cx, cy])
        r2 = (q ** 2).sum(1) / max(f, 1.0) ** 2
        return np.array([cx, cy]) + q * (1.0 + k1 * r2)[:, None]

    def resid(th):
        # 显式参数化: f 自由, 地面张力 + f1/f2 分叉作为归一化残差;
        # delta = 网柱外移 (米, 移动支架常态立在边线外), 由带端交点约束
        if free_xy:
            fx, fy, cx, cy, delta = th
            f = 0.5 * (fx + fy)
            k1 = 0.0
        elif fit_k1:
            cx, cy, f, delta, k1 = th
            fx = fy = f
        elif fix_cx is not None:
            cy, f, delta = th
            cx, k1 = fix_cx, 0.0
            fx = fy = f
        else:
            cx, cy, f, delta = th
            k1 = 0.0
            fx = fy = f
        K, R, t, f_avg, c_ang, ndiff = rt_from(G, fx, fy, cx, cy)
        P = K @ np.concatenate([R, t[:, None]], 1)
        res = []
        if free_xy:
            res += [f_avg * c_ang, f_avg * ndiff]
        else:
            res += [f_avg * c_ang / 15.0, f_avg * ndiff / 15.0]
            # f1/f2 分叉锚: 仅依赖 (cx,cy), 不随 f 逃逸 (σ_d=2.5% 畸变容忍)
            try:
                cd = decompose(G, cx, cy)
                div = (cd['f1'] - cd['f2']) / (0.5 * (cd['f1'] + cd['f2']))
                res.append(div / 0.025)
            except RuntimeError:
                res.append(40.0)
        # 柱偏移软先验 δ~0.3±0.25m (交点可见时由交点主导)
        res.append((delta - 0.3) / 0.25)
        vh = K @ R[:, 2]
        if abs(vh[2]) > 1e-3:
            for l, wt in vlines:
                res.append(wt * float(l @ vh) / vh[2])
        tl = seg_line(P, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)])
        if tl is not None and len(tape):
            tp = undist(tape[:, :2], k1, f, cx, cy)
            res += list(np.abs(tp @ tl[:2] + tl[2]))
        for pd in posts:
            x_post = (0.0 - delta) if pd['side'] < 0.5 * CW else (CW + delta)
            pl = seg_line(P, [(x_post, NET_Y, 0.0),
                              (x_post, NET_Y, NET_H + 0.2)])
            if pl is None:
                continue
            if 'sub' in pd and len(pd['sub']):
                pp = undist(pd['sub'], k1, f, cx, cy)
                res += list(0.3 * np.abs(pp @ pl[:2] + pl[2]))
            if tl is not None and pd['junc'] is not None:
                j = np.cross(tl, pl)
                if abs(j[2]) > 1e-9:
                    ju, jv = j[0] / j[2], j[1] / j[2]
                    jm = undist(np.array(pd['junc'], float)[None, :], k1, f, cx, cy)[0]
                    res += [1.5 * (ju - jm[0]), 1.5 * (jv - jm[1])]
        return np.array(res)

    return resid


def run_fit(G, obs, th0, bounds):
    resid = build_res(G, obs)
    th0 = np.asarray(th0, float)
    r0 = resid(th0)
    print(f'残差项数: {len(r0)}  初值 cost={float((r0 ** 2).sum()):.0f} px2')
    return least_squares(resid, th0, bounds=bounds, method='trf',
                         loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=300)


# ---------------------------------------------------------------- 验证工具

def tape_resid(P, tape):
    """网带采样点 |投影行 - 实测行| (mean, max, n); x 由 u 经 P 反求"""
    if len(tape) == 0:
        return None
    uv0, _ = project(P, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)])
    if not np.isfinite(uv0).all() or abs(uv0[1, 0] - uv0[0, 0]) < 1e-6:
        return None
    (u1, _), (u2, _) = uv0
    xs = np.clip(CW * (tape[:, 0] - u1) / (u2 - u1), 0.0, CW)
    uv, _ = project(P, [(x, NET_Y, NET_H) for x in xs])
    ok = np.isfinite(uv).all(1)
    if not ok.any():
        return None
    d = uv[ok, 1] - tape[ok, 1]
    return float(np.abs(d).mean()), float(np.abs(d).max()), int(ok.sum())


def tape_off(P, tape, xc):
    """干净网带偏差 @x: dv = 预测行 - 实测行(插值), x 由 u 经 P 反求"""
    if len(tape) < 10:
        return None
    uv0, _ = project(P, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)])
    if not np.isfinite(uv0).all() or abs(uv0[1, 0] - uv0[0, 0]) < 1e-6:
        return None
    (u1, _), (u2, _) = uv0
    u = u1 + (u2 - u1) * xc / CW
    t = tape[np.argsort(tape[:, 0])]
    if not (t[0, 0] - 2 <= u <= t[-1, 0] + 2):
        return None
    v_det = float(np.interp(u, t[:, 0], t[:, 1]))
    uv, _ = project(P, [(xc, NET_Y, NET_H)])
    if not np.isfinite(uv).all():
        return None
    return float(uv[0, 1] - v_det)


def vline_rms(K, r3, vlines):
    """竖直线到预测灭点的 RMS 距离 (px)"""
    if not vlines:
        return None
    vh = K @ r3
    if abs(vh[2]) < 1e-3:
        return None
    d = np.array([float(l @ vh) / vh[2] for l, _ in vlines])
    return float(np.sqrt((d ** 2).mean()))


def _s(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _div(a, b):
    if a is None or b is None:
        return float('nan')
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return float('nan')
    return abs(a - b) / (0.5 * (a + b)) * 100.0


# ---------------------------------------------------------------- 叠加图

def draw_overlay(frame, P0, P1, obs, vz, cal):
    vis = frame.copy()
    Hh, W = vis.shape[:2]

    def pline(P, p3d, color, thick):
        uv, dep = project(P, p3d)
        ok = np.isfinite(uv).all(1) & (dep > 1e-6)
        if ok.sum() >= 2:
            cv2.polylines(vis, [uv[ok].astype(np.int32)], False, color, thick, cv2.LINE_AA)

    def draw_meas(l, color, thick=1):
        a, b, c = l
        if abs(b) > 1e-6:
            p1 = (0, int(round(-c / b)))
            p2 = (W - 1, int(round(-(a * (W - 1) + c) / b)))
        else:
            p1 = (int(round(-c / a)), 0)
            p2 = (int(round(-c / a)), Hh - 1)
        cv2.line(vis, p1, p2, color, thick, cv2.LINE_AA)

    pline(P0, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)], (170, 170, 170), 1)
    pline(P0, [(0, NET_Y, 0), (0, NET_Y, NET_H)], (170, 170, 170), 1)
    pline(P0, [(CW, NET_Y, 0), (CW, NET_Y, NET_H)], (170, 170, 170), 1)
    pline(P1, [(0, NET_Y, NET_H), (CW, NET_Y, NET_H)], (0, 255, 0), 2)
    pline(P1, [(0, NET_Y, 0), (0, NET_Y, NET_H)], (0, 255, 0), 2)
    pline(P1, [(CW, NET_Y, 0), (CW, NET_Y, NET_H)], (0, 255, 0), 2)

    if obs['tape_line'] is not None:
        draw_meas(obs['tape_line'], (0, 255, 255), 1)
    for u, v in obs['tape']:
        cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0, 255, 255), -1)
    for p in obs['posts']:
        if p.get('junc') is not None:
            cv2.circle(vis, (int(round(p['junc'][0])), int(round(p['junc'][1]))),
                       6, (0, 255, 0), 2)
    for p in obs.get('post_feats', []):
        draw_meas(p['line'], (255, 0, 255), 1)
        for u, v in p['pts']:
            cv2.circle(vis, (int(round(u)), int(round(v))), 1, (255, 0, 255), -1)
    for l, _w in obs.get('mesh', []):
        draw_meas(l, (255, 255, 0), 1)
    if vz is not None and abs(vz[2]) > 1e-6:
        px, py = vz[0] / vz[2], vz[1] / vz[2]
        if -W < px < 2 * W and -Hh < py < 2 * Hh:
            cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 0), 2)

    cv2.rectangle(vis, (0, 0), (W, 84), (0, 0, 0), -1)
    cv2.putText(vis, f"old: f={cal['f0']:.1f} pp=({cal['cx0']:.0f},{cal['cy0']:.0f})",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(vis, f"new: f={cal['f']:.1f} pp=({cal['cx']:.0f},{cal['cy']:.0f})",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, 'gray=old  green=new  yellow=tape  magenta=post  cyan=mesh/vz',
                (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description='档2 双平面自标定(地面单应 + 球网)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--n-frames', type=int, default=11)
    ap.add_argument('--free-xy', action='store_true', help='fx!=fy 诊断模式')
    ap.add_argument('--k1', action='store_true', help='附加一阶径向畸变 k1 (实验)')
    ap.add_argument('--use-vlines', action='store_true',
                    help='竖直灭点残差参与拟合 (默认仅诊断)')
    ap.add_argument('--dump-obs', default=None,
                    help='将 G/obs pickle 导出供离线诊断后退出')
    args = ap.parse_args()

    rd = args.results_dir
    with open(os.path.join(rd, 'metadata.json')) as fp:
        meta = json.load(fp)
    W, Hh = meta['video']['width'], meta['video']['height']
    video_path = meta['video']['path']
    if not os.path.exists(video_path):
        video_path = os.path.join(rd, os.path.basename(video_path))

    print(f'=== 档2 双平面自标定: {rd} ===')
    orient, G, calib0 = load_G(rd, W, Hh)
    K0, R0, t0 = calib0['K'], calib0['R'], calib0['t']
    P0 = K0 @ np.concatenate([R0, t0[:, None]], 1)
    f0 = float(calib0['f'])
    print(f'朝向类: {orient}  单平面初值: f={f0:.1f}px, 主点=({W / 2:.0f},{Hh / 2:.0f})')

    frames = load_frames(video_path, args.n_frames)
    white_cnt, sob_acc, nf = accumulate(frames)
    qmask, qpoly = net_quad_mask(P0, G, W, Hh)

    tape, tape_diag = detect_tape(white_cnt, nf, P0, W, Hh)
    tape_line, tape_rms, tape_n = None, None, 0
    if len(tape) >= 6:
        rl = robust_line(tape)
        if rl is not None:
            tape_line, _, tape_rms, tape_n = rl
    print(f'网带: {len(tape)} 采样点 [{tape_diag}]' +
          (f', 直线内点 {tape_n}, rms={tape_rms:.2f}px' if tape_line is not None
           else ' (拟合失败)'))
    if len(tape) == 0:
        raise RuntimeError('未检测到网带像素, 无法双平面标定 (检查视频/投影先验)')

    juncs, jdiags = detect_junctions(white_cnt, nf, tape_line, P0, W)
    for jd in jdiags:
        print('带端扫描: ' + jd)

    post_feats = []
    for side, name in ((0.0, '左柱 U=0'), (CW, '右柱 U=6.1')):
        pts, pdiag = detect_post(sob_acc, P0, G, side, W, Hh, white_cnt=white_cnt, nf=nf)
        l, rms, n_in = None, None, 0
        if len(pts) >= 15:
            rl = robust_line(pts)
            if rl is not None:
                l, _, rms, n_in = rl
        junc = juncs.get(side)
        if l is not None:
            jline = line_isect(tape_line, l) if tape_line is not None else None
            post_feats.append(dict(side=side, line=l, pts=pts, n=n_in, rms=rms,
                                   junc_line=jline))
            print(f'{name}: [{pdiag}], 柱线∩带=({jline[0]:.1f},{jline[1]:.1f})'
                  if jline is not None else f'{name}: [{pdiag}]')
        else:
            print(f'{name}: [{pdiag}] (柱线诊断不可用)')

    posts = [dict(side=U, junc=juncs[U]) for U in (0.0, CW) if U in juncs]
    print(f'拟合用带端交点: {len(posts)}/2' +
          ('' if len(posts) == 2 else ' !! 交点不足, 拟合退化'))

    mesh = detect_mesh_lines(sob_acc, qmask, W, Hh)
    print(f'网面竖线: {len(mesh)} 条' + ('' if mesh else ' (不可得, 降级)'))

    vlines = ([(p['line'], 1.0) for p in post_feats] +
              [(l, min(1.0, wlen / 300.0)) for l, wlen in mesh])
    vz, spread = vanish_z(vlines)

    obs = dict(tape=tape, posts=posts, post_feats=post_feats,
               vlines=vlines if args.use_vlines else [],
               mesh=[(l, min(1.0, wlen / 300.0)) for l, wlen in mesh],
               tape_line=tape_line,
               free_xy=False, fit_k1=False, fix_cx=None, W=W)
    if args.use_vlines:
        print('(竖直灭点残差已启用 --use-vlines; 默认关闭: 两网柱线近平行, 灭点残差病态)')
    if args.free_xy:
        if len(posts) < 1:
            print('!! --free-xy 需要至少一个带端交点, 已忽略')
        else:
            obs['free_xy'] = True
    if args.k1:
        obs['fit_k1'] = True
    if len(posts) < 1 and not obs['free_xy'] and not obs['fit_k1']:
        obs['fix_cx'] = W / 2.0
        print('!! 无带端交点, 水平约束退化 -> 固定 cx=居中, 仅拟合 cy (f 由地面约束解出)')

    if obs['free_xy']:
        th0 = [f0, f0, W / 2.0, Hh / 2.0, 0.3]
        bounds = ([0.3 * W, 0.3 * W, 0, 0, -0.2], [8 * W, 8 * W, W, Hh, 1.0])
    elif obs['fit_k1']:
        th0 = [W / 2.0, Hh / 2.0, f0, 0.3, 0.0]
        bounds = ([0, 0, 0.3 * W, -0.2, -0.3], [W, Hh, 8 * W, 1.0, 0.3])
    elif obs['fix_cx'] is not None:
        th0 = [Hh / 2.0, f0, 0.3]
        bounds = ([0, 0.3 * W, -0.2], [Hh, 8 * W, 1.0])
    else:
        th0 = [W / 2.0, Hh / 2.0, f0, 0.3]
        bounds = ([0, 0, 0.3 * W, -0.2], [W, Hh, 8 * W, 1.0])

    if args.dump_obs:
        with open(args.dump_obs, 'wb') as fp:
            pickle.dump({'G': G, 'obs': obs}, fp)
        print(f'观测已导出: {args.dump_obs}')
        return

    r = run_fit(G, obs, th0, bounds)

    if obs['fit_k1']:
        cx, cy, f, delta, k1 = r.x
        fx = fy = f
    elif obs['free_xy']:
        fx, fy, cx, cy, delta = r.x
        f, k1 = 0.5 * (fx + fy), 0.0
    elif obs['fix_cx'] is not None:
        cy, f, delta = r.x
        cx, k1 = obs['fix_cx'], 0.0
        fx = fy = f
    else:
        cx, cy, f, delta = r.x
        k1 = 0.0
        fx = fy = f

    K1, R1, t1, f_avg, c_ang, ndiff = rt_from(G, fx, fy, cx, cy)
    P1 = K1 @ np.concatenate([R1, t1[:, None]], 1)
    C1 = -R1.T @ t1

    print(f'\n--- 拟合结果 ({len(r.fun)} 项残差, cost={2 * r.cost:.1f} px2) ---')
    print(f'f={f:.1f}px  cx={cx:.1f}  cy={cy:.1f}  网柱外移 δ={delta:.2f}m' +
          (f'  k1={k1:+.4f}' if obs['fit_k1'] else '') +
          (f'  (fx={fx:.1f}, fy={fy:.1f})' if obs['free_xy'] else ''))
    print(f'地面正交性残差: c_ang*f={f_avg * c_ang:+.2f}px  ndiff*f={f_avg * ndiff:+.2f}px')

    def f12(ppx, ppy):
        try:
            c = decompose(G, ppx, ppy)
            return c['f1'], c['f2']
        except RuntimeError:
            return None, None

    f1o, f2o = f12(W / 2.0, Hh / 2.0)
    f1n, f2n = f12(cx, cy)
    print(f'f1/f2 分叉(可证伪预测, 应收敛): '
          f'旧主点 {_div(f1o, f2o):.2f}% -> 新主点 {_div(f1n, f2n):.2f}%')
    print(f'  旧: f1={_s(f1o)} f2={_s(f2o)}   新: f1={_s(f1n)} f2={_s(f2n)}')

    tro = tape_resid(P0, tape)
    trn = tape_resid(P1, tape)
    if tro and trn:
        print(f'网带残差 |dv|: 旧 mean={tro[0]:.2f}/max={tro[1]:.2f}px -> '
              f'新 mean={trn[0]:.2f}/max={trn[1]:.2f}px')
    vro = vline_rms(K0, R0[:, 2], vlines)
    vrn = vline_rms(K1, R1[:, 2], vlines)
    if vro is not None and vrn is not None:
        print(f'竖直线过灭点 RMS: 旧={vro:.1f}px -> 新={vrn:.1f}px (n={len(vlines)})')

    C0 = calib0['C']
    print(f'相机中心(工作系): 旧 C=({C0[0]:+.2f},{C0[1]:+.2f},{C0[2]:+.2f}) -> '
          f'新 C=({C1[0]:+.2f},{C1[1]:+.2f},{C1[2]:+.2f})')
    C_hn = (float(C1[0]), float(CL - C1[1]), float(C1[2]))

    frame_mid = frames[len(frames) // 2]
    offs_old, offs_new = [], []
    for xc in (0.5, 1.8, 3.05, 4.3, 5.6):
        oo = tape_off(P0, tape, xc)
        on = tape_off(P1, tape, xc)
        if oo is not None:
            offs_old.append(oo)
        if on is not None:
            offs_new.append(on)
        print(f'  网带偏差 @x={xc}: 旧 {_s(oo)}px -> 新 {_s(on)}px')

    cal = dict(f0=f0, cx0=W / 2.0, cy0=Hh / 2.0, f=float(f), cx=float(cx), cy=float(cy))
    vis = draw_overlay(frame_mid, P0, P1, obs, vz, cal)
    out_img = os.path.join(rd, 'net_calib_debug.jpg')
    cv2.imwrite(out_img, vis)
    print(f'调试叠加图: {out_img}')

    out = dict(
        orient=orient, f=float(f), cx=float(cx), cy=float(cy),
        fx=float(fx), fy=float(fy), k1=float(k1),
        post_offset_m=float(delta),
        K=K1.tolist(), R=R1.tolist(), t=t1.tolist(),
        C_working=[float(v) for v in C1], C_hn=C_hn, frame=[W, Hh],
        residuals=dict(n=int(len(r.fun)), cost_px2=float(2 * r.cost),
                       ground=[float(f_avg * c_ang), float(f_avg * ndiff)]),
        diag=dict(
            f1f2_old_pp=dict(f1=_s(f1o), f2=_s(f2o), div_pct=_s(_div(f1o, f2o))),
            f1f2_new_pp=dict(f1=_s(f1n), f2=_s(f2n), div_pct=_s(_div(f1n, f2n))),
            tape_resid_old=tro and [float(tro[0]), float(tro[1])],
            tape_resid_new=trn and [float(trn[0]), float(trn[1])],
            vline_rms_old=_s(vro), vline_rms_new=_s(vrn),
            vz=[_s(x) for x in vz] if vz is not None else None,
            vz_spread=_s(spread),
            net_offset_old=[_s(o) for o in offs_old],
            net_offset_new=[_s(o) for o in offs_new]),
        features=dict(tape_pts=int(len(tape)), tape_line_rms=_s(tape_rms),
                      junctions=[dict(side=p['side'],
                                      u=_s(p['junc'][0]), v=_s(p['junc'][1]))
                                 for p in posts],
                      post_lines=[dict(side=p['side'], n=p['n'], rms=_s(p['rms']),
                                       line=[_s(x) for x in p['line']],
                                       junc_line=[_s(x) for x in p['junc_line']]
                                       if p.get('junc_line') is not None else None)
                                  for p in post_feats],
                      mesh_lines=len(mesh), n_frames=nf),
        note='双平面自标定 v2: H1(地面) + 网带/网柱/竖直灭点. 工作系(右手). '
             '投影 H.npy 系 3D 点(含高度)前先做 y_w=13.4-y_hn. '
             '方形像素假设保留; 尺度由米制场地锚定; 左右镜像歧义未解.')
    with open(os.path.join(rd, 'calib3d_2plane.json'), 'w') as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print(f'标定已保存: {os.path.join(rd, "calib3d_2plane.json")} (未覆盖 calib3d.json)')


if __name__ == '__main__':
    main()
