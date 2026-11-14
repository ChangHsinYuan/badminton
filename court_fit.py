"""低视角球场自动标定 v2：线对应假设枚举 + 双向倒角评分 + 单应性直接精修

原理：羽毛球标线布局固定（6.1 x 13.4 m）。从白线 mask 提取霍夫线并按方向分两族
（对应场地两个正交方向），枚举"检测线 <-> 模型线值"的对应假设，用 2x2 线交点
DLT 出候选单应性，以双向倒角距离评分（投影线要贴住白线、白线也要被投影线解释），
最优假设再做 8 自由度 Nelder-Mead 精修。相机固定视频全程自动，无需手动点角点。
"""
import cv2
import itertools
import numpy as np
from scipy.optimize import minimize

CW, CL = 6.1, 13.4
X_VALS = [0.0, 0.46, 3.05, 5.64, 6.1]
Y_VALS = [0.0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4]

LINE_NAMES = ['端线y=0', '端线y=13.4', '边线x=0', '边线x=6.1', '单打边线x=0.46',
              '单打边线x=5.64', '前发球线y=4.72', '前发球线y=8.68', '后发球线y=0.76',
              '后发球线y=12.64', '中线', '网线']


def court_lines():
    L = []

    def add(p, q, w=1.0):
        L.append((p, q, w))

    add((0, 0), (CW, 0), 2.0)
    add((0, CL), (CW, CL), 2.0)
    add((0, 0), (0, CL), 2.0)
    add((CW, 0), (CW, CL), 2.0)
    add((0.46, 0), (0.46, CL))
    add((CW - 0.46, 0), (CW - 0.46, CL))
    add((0, CL / 2 - 1.98), (CW, CL / 2 - 1.98))
    add((0, CL / 2 + 1.98), (CW, CL / 2 + 1.98))
    add((0, 0.76), (CW, 0.76))
    add((0, CL - 0.76), (CW, CL - 0.76))
    add((CW / 2, 0.76), (CW / 2, CL - 0.76), 0.6)
    add((0, CL / 2), (CW, CL / 2), 0.4)
    return L


LINES = court_lines()
SAMPLES = np.array([np.array(p) + (np.array(q) - np.array(p)) * np.linspace(0, 1, 60)[:, None]
                    for p, q, _ in LINES])
WEIGHTS = np.array([w for _, _, w in LINES])


def apply_H(H, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    return h[:, :2] / h[:, 2:3]


def dedup_lines(lines_mask, min_len=90):
    """霍夫线段 -> 共线合并 -> [(l(3,), len)]，l 满足 ax+by+c=0 且 |(a,b)|=1"""
    seg = cv2.HoughLinesP(lines_mask, 1, np.pi / 360, threshold=60,
                          minLineLength=min_len, maxLineGap=12)
    if seg is None:
        return []
    segs = seg.reshape(-1, 4).astype(float)
    lines = []
    for x1, y1, x2, y2 in segs:
        L = float(np.hypot(x2 - x1, y2 - y1))
        l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        l /= np.linalg.norm(l[:2]) + 1e-12
        merged = False
        for ex in lines:
            if abs(float(ex[0][0] * l[0] + ex[0][1] * l[1])) > np.cos(np.radians(1.5)) \
               and abs(float(ex[0][2] - l[2])) < 6:
                if L > ex[1]:
                    ex[0], ex[1] = l, L
                merged = True
                break
        if not merged:
            lines.append([l, L])
    return lines


def isect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def line_points_in_image(l, W=1280, H=720, n=8):
    """线 ax+by+c=0 与图像边界的交点之间均匀采样 n 点; 不可见返回 None"""
    a, b, c = l
    pts = []
    if abs(b) > 1e-9:
        for x in (0, W - 1):
            y = -c / b - a * x / b
            if 0 <= y < H:
                pts.append((x, y))
    if abs(a) > 1e-9:
        for y in (0, H - 1):
            x = -c / a - b * y / a
            if 0 <= x < W:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    p, q = np.array(pts[0]), np.array(pts[-1])
    if np.hypot(*(q - p)) < 40:
        return None
    t = np.linspace(0, 1, n)[:, None]
    return p + (q - p) * t


def consistency(H, dlines, is_a=None, smooth=False):
    """场地坐标系一致性: 每条检测线经 H 映到 court 后, 自动判断它是常量x还是常量y
    (取两种轴中得分高者), 且值须接近该轴的模型线值. 不依赖图像空间的角度分族.
    smooth=True 返回连续版本(用于精修); 否则离散判定.
    返回 (score, matched) — matched: (axis, value, length), axis: 0(x)/1(y)."""
    tot, good = 0.0, 0.0
    matched = []
    for (l, L) in dlines:
        tot += L
        pts = line_points_in_image(l)
        if pts is None:
            continue
        g = apply_H(H, pts)
        if not np.isfinite(g).all():
            continue
        best_s, best_axis, best_v = 0.0, None, None
        for axis in (0, 1):
            vals = g[:, axis]
            other = g[:, 1 - axis]
            # 常量x线沿y(长13.4)延伸; 常量y线沿x(宽6.1)延伸
            span_hi, span_lo, ext = (CL, 0.0, CL) if axis == 0 else (CW, 0.0, CW)
            span = max(0.0, min(other.max(), span_hi) - max(other.min(), span_lo)) / ext
            if span < 0.12:
                continue
            refs = X_VALS if axis == 0 else Y_VALS
            c = float(np.std(vals))
            v = float(np.median(vals))
            near = min(abs(v - r) for r in refs)
            if smooth:
                s = span * np.exp(-(c / 0.12) ** 2) * np.exp(-(near / 0.18) ** 2)
            else:
                if c > 0.2 or near > 0.15:
                    continue
                s = span
            if s > best_s:
                best_s, best_axis, best_v = s, axis, v
        if best_axis is None:
            continue
        good += L * best_s
        if not smooth:
            matched.append((best_axis, best_v, float(L)))
    return good / max(tot, 1e-9), matched


class Scorer:
    """双向倒角: 投影模型线 vs 白线 mask"""

    def __init__(self, lines_mask, n_white=3000, seed=0):
        self.H_img, self.W_img = lines_mask.shape
        # distanceTransform 对输入非零像素算其到最近零像素的距离;
        # 取反相输入使任意像素的值 = 到最近白线像素的距离
        self.dt = cv2.distanceTransform((lines_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        self.flat = self.dt.ravel()
        ys, xs = np.where(lines_mask > 0)
        idx = np.random.RandomState(seed).choice(len(xs), size=min(n_white, len(xs)), replace=False)
        self.white_pts = np.stack([xs[idx], ys[idx]], 1).astype(float)
        self.sub = [l[:: 2] for l in SAMPLES]

    def forward(self, H):
        """投影线贴白线的程度 (court->image)"""
        Hc2i = np.linalg.inv(H)
        s, n = 0.0, 0.0
        for line, w in zip(self.sub, WEIGHTS):
            uv = apply_H(Hc2i, line)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < self.W_img) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H_img)
            u = np.clip(uv[:, 0], 0, self.W_img - 1).astype(int)
            v = np.clip(uv[:, 1], 0, self.H_img - 1).astype(int)
            s += w * np.where(ok, np.exp(-self.flat[v * self.W_img + u] / 2.5), -0.3).sum()
            n += w * len(line)
        return s / n

    def reverse(self, H):
        """白线被投影线解释的程度 (image->court 距离场)"""
        proj = np.zeros((self.H_img, self.W_img), np.uint8)
        Hc2i = np.linalg.inv(H)
        for line in self.sub:
            uv = apply_H(Hc2i, line)
            ok = (uv[:, 0] > -50) & (uv[:, 0] < self.W_img + 50) & \
                 (uv[:, 1] > -50) & (uv[:, 1] < self.H_img + 50)
            pts = uv[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(proj, [pts], False, 255, 3)
        if proj.max() == 0:
            return 0.0
        dt2 = cv2.distanceTransform((proj == 0).astype(np.uint8), cv2.DIST_L2, 3)
        d = dt2[self.white_pts[:, 1].astype(int), self.white_pts[:, 0].astype(int)]
        return float(np.exp(-d / 2.5).mean())

    def coverage(self, H):
        g = apply_H(H, self.white_pts)
        ok = np.isfinite(g).all(1)
        if ok.sum() == 0:
            return 0.0
        inside = (g[ok, 0] >= 0) & (g[ok, 0] <= CW) & (g[ok, 1] >= 0) & (g[ok, 1] <= CL)
        return float(inside.mean())

    def strict_forward(self, H):
        """逐线硬支持度: 每条模型线大部分采样点须紧贴白线（DT<2 好, DT>6 坏）"""
        Hc2i = np.linalg.inv(H)
        tot, wsum = 0.0, 0.0
        for line, w in zip(SAMPLES, WEIGHTS):
            uv = apply_H(Hc2i, line)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < self.W_img) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H_img)
            if ok.sum() < 8:
                continue
            u = np.clip(uv[ok, 0], 0, self.W_img - 1).astype(int)
            v = np.clip(uv[ok, 1], 0, self.H_img - 1).astype(int)
            d = self.flat[v * self.W_img + u]
            s = (d < 2.0).mean() - 0.5 * (d > 6.0).mean()
            tot += w * s * ok.mean()
            wsum += w
        return tot / wsum if wsum else -1.0

    def strict_reverse(self, H):
        """白线被投影线解释: 光栅化后要求白点贴近某条投影线"""
        proj = np.zeros((self.H_img, self.W_img), np.uint8)
        Hc2i = np.linalg.inv(H)
        for line in self.sub:
            uv = apply_H(Hc2i, line)
            ok = (uv[:, 0] > -50) & (uv[:, 0] < self.W_img + 50) & \
                 (uv[:, 1] > -50) & (uv[:, 1] < self.H_img + 50)
            pts = uv[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(proj, [pts], False, 255, 3)
        if proj.max() == 0:
            return 0.0
        dt2 = cv2.distanceTransform((proj == 0).astype(np.uint8), cv2.DIST_L2, 3)
        d = dt2[self.white_pts[:, 1].astype(int), self.white_pts[:, 0].astype(int)]
        return float((d < 3.0).mean() - 0.5 * (d > 9.0).mean())

    def total(self, H):
        return self.forward(H) + 0.7 * self.reverse(H) + 0.3 * self.coverage(H)

    def total2(self, H):
        return self.strict_forward(H) + 0.7 * self.strict_reverse(H)


def vec_to_H(v):
    H = np.append(v, 1.0).reshape(3, 3)
    return H / np.linalg.norm(H)


def H_to_vec(H):
    return (H / H[2, 2]).reshape(-1)[:8]


def refine(H0, scorer, maxiter=3000, strict=True, obj=None):
    if obj is None:
        obj = (lambda H: scorer.total2(H) if strict else scorer.total(H))

    def neg(v):
        return -obj(vec_to_H(v))

    res = minimize(neg, H_to_vec(H0), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-7})
    return vec_to_H(res.x), -res.fun


def quad_ok(H):
    """退化检查: 四角须互不重合、构成凸四边形 (面积占画面 5%~85%)"""
    try:
        c = corners_from_H(H)
    except Exception:
        return False
    if not np.isfinite(c).all() or np.abs(c).max() > 4000:
        return False
    area = cv2.contourArea(c.astype(np.float32))
    if area < 0.05 * 1280 * 720 or area > 0.85 * 1280 * 720:
        return False
    if not cv2.isContourConvex(c.astype(np.float32)):
        return False
    # 四角两两距离须显著(真球场四角在透视下不会重合)
    for i in range(4):
        for j in range(i + 1, 4):
            if np.hypot(*(c[i] - c[j])) < 60:
                return False
    return True


def fit(lines_mask, seed=0, verbose=True):
    """返回 (H: image->court 3x3, score, 诊断 dict)"""
    scorer = Scorer(lines_mask, seed=seed)
    dlines = dedup_lines(lines_mask)
    if verbose:
        print(f'去重后检测线: {len(dlines)} 条')
    is_a = []
    for l, L in dlines:
        ang = np.degrees(np.arctan2(l[1], l[0])) % 180.0
        is_a.append((ang % 90.0) < 45)
    famA = [dl for dl, ia in zip(dlines, is_a) if ia]
    famB = [dl for dl, ia in zip(dlines, is_a) if not ia]

    def enum_hyp(la, lb, fast_pts, flat_dt):
        """la: 视为常量x边线族(配X_VALS), lb: 视为常量y端线族(配Y_VALS)
        交点 (la_i x lb_j) <-> 场地点 (X_VALS[ai], Y_VALS[bj])"""
        hyps = []
        W_, H_ = 1280, 720
        for i1 in range(len(la)):
            for i2 in range(i1 + 1, len(la)):
                for j1 in range(len(lb)):
                    for j2 in range(j1 + 1, len(lb)):
                        p11 = isect(la[i1][0], lb[j1][0])
                        p12 = isect(la[i1][0], lb[j2][0])
                        p21 = isect(la[i2][0], lb[j1][0])
                        p22 = isect(la[i2][0], lb[j2][0])
                        if any(p is None for p in (p11, p12, p21, p22)):
                            continue
                        img = np.array([p11, p12, p22, p21])
                        if np.abs(img).max() > 2200:
                            continue
                        if cv2.contourArea(img.astype(np.float32)) < 4000:
                            continue
                        for a1 in range(len(X_VALS)):
                            for a2 in range(a1 + 1, len(X_VALS)):
                                for b1 in range(len(Y_VALS)):
                                    for b2 in range(b1 + 1, len(Y_VALS)):
                                        court = np.array([
                                            [X_VALS[a1], Y_VALS[b1]],
                                            [X_VALS[a1], Y_VALS[b2]],
                                            [X_VALS[a2], Y_VALS[b2]],
                                            [X_VALS[a2], Y_VALS[b1]]])
                                        Hm, _ = cv2.findHomography(
                                            img.reshape(-1, 1, 2).astype(np.float64),
                                            court.reshape(-1, 1, 2).astype(np.float64), 0)
                                        if Hm is None:
                                            continue
                                        # 快速预打分: 投影模型线贴白线
                                        Hc2i = np.linalg.inv(Hm)
                                        uv = apply_H(Hc2i, fast_pts)
                                        ok = ((uv[:, 0] >= 0) & (uv[:, 0] < W_) &
                                              (uv[:, 1] >= 0) & (uv[:, 1] < H_))
                                        if ok.mean() < 0.5:
                                            continue
                                        u = np.clip(uv[:, 0], 0, W_ - 1).astype(int)
                                        v = np.clip(uv[:, 1], 0, H_ - 1).astype(int)
                                        sup = np.exp(-flat_dt[v * W_ + u] / 2.5)
                                        hyps.append((float(sup[ok].mean()) if ok.any() else 0.0, Hm))
        hyps.sort(key=lambda t: -t[0])
        return [H for _, H in hyps]

    hyps = []
    if famA and famB:
        fast_pts = SAMPLES[:, ::3].reshape(-1, 2)  # 20 点/线
        la_a = sorted(famA, key=lambda t: -t[1])[:6]
        lb_b = sorted(famB, key=lambda t: -t[1])[:6]
        # 假设1: famA=边线族(常量x), famB=端线族(常量y)
        hyps += enum_hyp(la_a, lb_b, fast_pts, scorer.flat)
        # 假设2: 交换
        hyps += enum_hyp(lb_b, la_a, fast_pts, scorer.flat)
    if verbose:
        print(f'有效假设数: {len(hyps)}')

    # 全量评场地坐标系一致性 (软倒角仅作廉价剪枝与并列微调)
    scored = []
    for i, H in enumerate(hyps):
        f = scorer.forward(H)
        if f < 0.25 or not quad_ok(H):
            continue
        c, _ = consistency(H, dlines, smooth=True)
        scored.append((c + 0.3 * f, c, i))
    scored.sort(key=lambda t: -t[0])
    if verbose and scored:
        print(f'假设一致性最优 {scored[0][1]:.3f} (组合 {scored[0][0]:.3f})')
    if not scored:
        raise RuntimeError(
            '未找到有效的单应性假设: 请检查视频是否满足拍摄要求 '
            '(固定机位、球场边线大部分可见、光线均匀), 或掩码质量过低')
    cands = [hyps[i] for _, _, i in scored[:12]]

    def objective(H):
        c, _ = consistency(H, dlines, smooth=True)
        return c + 0.3 * scorer.forward(H)

    best_H, best_s, best_c = None, -9, -9
    for H0 in cands:
        H, s = refine(H0, scorer, maxiter=800, obj=objective)
        if not quad_ok(H):
            continue
        c_hard, _ = consistency(H, dlines, smooth=False)
        # 以硬一致性为主排序, 防止像素项把解拖向退化构型
        if c_hard > best_c + 1e-6 or (abs(c_hard - best_c) < 1e-6 and s > best_s):
            best_H, best_s, best_c = H, s, c_hard

    # 对应关系扩展 (子集ICP): 紧匹配作基底, 宽松候选逐子集试拼,
    # 以诚实分 (硬一致性 + strict_forward) 选优. 远端线在近半场解下会有偏差,
    # 直接全量 DLT 会被坏候选拖垮, 故逐子集验证.
    def honest(H):
        c, _ = consistency(H, dlines, smooth=False)
        return c + 0.5 * scorer.strict_forward(H)

    for _ in range(2):
        info = {}
        for k, (l, L) in enumerate(dlines):
            pts = line_points_in_image(l, n=10)
            if pts is None:
                continue
            g = apply_H(best_H, pts)
            if not np.isfinite(g).all():
                continue
            info[k] = (pts, g)
        tight, loose = [], []  # (off, k, axis, ref)
        for k, (pts, g) in info.items():
            for axis in (0, 1):
                refs = X_VALS if axis == 0 else Y_VALS
                vals = g[:, axis]
                v, std = float(np.median(vals)), float(np.std(vals))
                for ref in refs:
                    off = abs(v - ref)
                    if off > 0.8:
                        continue
                    if off <= 0.2 and std <= 0.35:
                        tight.append((off, k, axis, ref))
                    elif std <= 0.9:
                        loose.append((off, k, axis, ref))
        # 每个模型槽位取最近的紧匹配作基底
        base, used = [], set()
        for off, k, axis, ref in sorted(tight):
            if (axis, ref) in used or k in used:
                continue
            base.append((k, axis, ref))
            used.add((axis, ref))
            used.add(k)
        # 松候选: 每槽位最多1条, 按偏差排序, 最多4条
        loose_pick, lused = [], set()
        for off, k, axis, ref in sorted(loose):
            if (axis, ref) in used or k in lused or (axis, ref) in lused:
                continue
            loose_pick.append((k, axis, ref))
            lused.add((axis, ref))
            lused.add(k)
        loose_pick = loose_pick[:4]
        if not loose_pick:
            break

        def build(subset):
            img_pts, court_pts = [], []
            for k, axis, ref in list(base) + list(subset):
                pts, g = info[k]
                other = g[:, 1 - axis]
                lo, hi = (0, CL) if axis == 0 else (0, CW)
                keep = (other > lo - 0.4) & (other < hi + 0.4)
                for pt, o in zip(pts[keep], other[keep]):
                    img_pts.append(pt)
                    court_pts.append([ref, o] if axis == 0 else [o, ref])
            if len(img_pts) < 12:
                return None
            Hn, _ = cv2.findHomography(np.array(img_pts, np.float64).reshape(-1, 1, 2),
                                       np.array(court_pts, np.float64).reshape(-1, 1, 2), 0)
            if Hn is None or not quad_ok(Hn):
                return None
            return Hn

        best_variant = (honest(best_H), None)
        for r in range(1, len(loose_pick) + 1):
            for combo in itertools.combinations(range(len(loose_pick)), r):
                Hn = build([loose_pick[i] for i in combo])
                if Hn is None:
                    continue
                H, _ = refine(Hn, scorer, maxiter=3000, obj=objective)
                if not quad_ok(H):
                    continue
                s = honest(H)
                if s > best_variant[0]:
                    best_variant = (s, H)
        if best_variant[1] is None:
            break
        best_H = best_variant[1]

    # 逐线硬支持度诊断
    Hc2i = np.linalg.inv(best_H)
    per_line = []
    for name, line, w in zip(LINE_NAMES, SAMPLES, WEIGHTS):
        uv = apply_H(Hc2i, line)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
        if ok.sum() < 8:
            per_line.append((name, 0.0, int(ok.sum())))
            continue
        u = np.clip(uv[ok, 0], 0, 1279).astype(int)
        v = np.clip(uv[ok, 1], 0, 719).astype(int)
        d = scorer.flat[v * 1280 + u]
        sup = (d < 2.0).mean() - 0.5 * (d > 6.0).mean()
        per_line.append((name, float(sup * ok.mean()), int(ok.sum())))
    cons_hard, matched = consistency(best_H, dlines, smooth=False)
    diag = {'per_line': per_line, 'strict_forward': scorer.strict_forward(best_H),
            'strict_reverse': scorer.strict_reverse(best_H),
            'consistency': cons_hard, 'matched': matched}
    if verbose:
        print(f'\n=== 最优 score={best_s:.4f} ===')
        print(f"strict_forward={diag['strict_forward']:.3f} "
              f"strict_reverse={diag['strict_reverse']:.3f} 一致性={cons_hard:.3f}")
        for name, sup, nin in per_line:
            flag = 'OK ' if sup > 0.5 else ('~  ' if sup > 0.25 else 'BAD')
            print(f'  [{flag}] {name}: 支持度 {sup:.2f} (画面内 {nin}/60)')
        print('吸附线:')
        for axis, v, L in matched:
            refs = X_VALS if axis == 0 else Y_VALS
            near = min(refs, key=lambda r: abs(r - v))
            print(f'  常量{"x" if axis == 0 else "y"} v={v:.2f} -> {near} (len {L:.0f})')
    return best_H, best_s, diag


def corners_from_H(H):
    """场地四角 (0,0),(6.1,0),(6.1,13.4),(0,13.4) 的图像位置（ court->image )"""
    Hi = np.linalg.inv(H)
    cs = np.array([[0, 0], [CW, 0], [CW, CL], [0, CL]], float)
    return apply_H(Hi, cs)
