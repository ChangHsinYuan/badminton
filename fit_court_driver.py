"""低视角球场自动标定入口（Fork 新增）

用法:
  python fit_court_driver.py --video videos/xxx.mp4
  python fit_court_driver.py --video videos/xxx.mp4 --frame-index 750 --output-dir results/xxx

流程:
  1. 从视频均匀采样 45 帧, 白线掩码取时间中值(消除球员/球遮挡)
  2. 绿色地垫颜色先验圈定球场区域
  3. Hough 检测线段 -> 场地坐标系一致性评分枚举单应性 -> Chamfer 精修
  4. 朝向规范化后输出 results/<视频名>/court_annotations.txt, 供 main.py 直接复用
"""
import argparse
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'badminton_analysis'))

import cv2
import numpy as np
from court_fit import CL, CW, LINES, SAMPLES, WEIGHTS, apply_H, corners_from_H, fit

LINE_COLORS = {2.0: (0, 0, 255), 1.0: (0, 255, 255), 0.6: (255, 0, 255), 0.4: (255, 0, 255)}
N_MEDIAN_FRAMES = 45


def _filter_line_components(mask, min_len=90):
    """只保留能被长直线(Hough, 与 dedup 同参)支持的像素, 剔除网带/球员残影等团块"""
    seg = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                          minLineLength=min_len, maxLineGap=12)
    keep = np.zeros_like(mask)
    if seg is not None:
        for x1, y1, x2, y2 in seg[:, 0]:
            cv2.line(keep, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
    return cv2.bitwise_and(mask, keep)


def build_lines_mask(video_path, n_samples=N_MEDIAN_FRAMES, verbose=True):
    """采样多帧做时间中值, 生成球场白线掩码(消除运动物体遮挡)"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(n_samples, max(total, 1))).astype(int)
    whites, greens = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        whites.append((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170))
        greens.append((hsv[:, :, 0] >= 60) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 40))
    cap.release()
    if not whites:
        raise RuntimeError(f'无法从视频读取帧: {video_path}')
    white_med = (np.median(np.stack(whites).astype(np.float32), axis=0) > 0.5).astype(np.uint8) * 255
    green_med = (np.median(np.stack(greens).astype(np.float32), axis=0) > 0.5).astype(np.uint8)
    green_med = cv2.morphologyEx(green_med, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    roi = cv2.dilate(green_med * 255, np.ones((31, 31), np.uint8))
    mask = cv2.bitwise_and(white_med, roi)
    mask = _filter_line_components(mask)  # 先过滤(线各自独立)再闭运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if verbose:
        print(f'采样 {len(whites)} 帧, 白线掩码像素: {int((mask > 0).sum())}')
    return mask


def save_overlay(H, frame, path):
    """把拟合的球场线 + 网格投影回图像, 人工检查对齐用"""
    Hc2i = np.linalg.inv(H)
    vis = frame.copy()
    for (p, q, w), line in zip(LINES, SAMPLES):
        uv = apply_H(Hc2i, line)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < vis.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < vis.shape[0])
        pts = uv[ok].astype(int)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, LINE_COLORS[w], 2)
    def seg(p, q, color, thick=1):
        pts = np.stack([p, q])
        if not np.isfinite(pts).all():
            return
        pts = np.clip(pts, -3000, 5000).astype(int)
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), color, thick)

    for x in range(int(CW) + 1):
        uv = apply_H(Hc2i, [(x, 0), (x, CL)])
        seg(uv[0], uv[1], (180, 180, 180))
    for y in range(int(CL) + 1):
        uv = apply_H(Hc2i, [(0, y), (CW, y)])
        seg(uv[0], uv[1], (180, 180, 180))
    cs = corners_from_H(H)
    for (x, y), label in zip(cs, ['C1(0,0)', 'C2(6.1,0)', 'C3(6.1,13.4)', 'C4(0,13.4)']):
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(vis, f'{label}({int(x)},{int(y)})', (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(path, vis)
    print(f'已保存叠加图(请人工确认对齐): {path}')


def order_corners(pts):
    """四点 -> 图像 TL,TR,BR,BL 循环排序(质心角排序, 从最左上开始, 保证第二步向右)"""
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cyc = [pts[i] for i in np.argsort(ang)]
    k = min(range(4), key=lambda i: cyc[i][0] + cyc[i][1])
    cyc = cyc[k:] + cyc[:k]
    nxt = cyc[1]
    if nxt[1] - cyc[0][1] > abs(nxt[0] - cyc[0][0]) or nxt[0] < cyc[0][0]:
        cyc = [cyc[0]] + cyc[1:][::-1]
    return [tuple(p) for p in cyc[:4]]


def canonicalize(H):
    """线集关于 x/y 镜像对称, 一致性无法区分朝向;
    按 demo 约定 (0,0)=远左角 规范化, 使 court 坐标 y 向近端增大、x 向右增大"""
    cs = corners_from_H(H)
    if cs[0][1] > cs[2][1]:
        H = np.array([[1, 0, 0], [0, -1, CL], [0, 0, 1]], float) @ H
        cs = corners_from_H(H)
    if cs[0][0] > cs[1][0]:
        H = np.array([[-1, 0, CW], [0, 1, 0], [0, 0, 1]], float) @ H
        cs = corners_from_H(H)
    return H, cs


def main():
    ap = argparse.ArgumentParser(description='低视角球场自动标定')
    ap.add_argument('--video', required=True, help='输入视频路径')
    ap.add_argument('--mask', default=None, help='已生成的白线掩码 npy(跳过采样)')
    ap.add_argument('--frame-index', type=int, default=None, help='用于叠加图的帧号(默认取视频中点)')
    ap.add_argument('--output-dir', default=None, help='输出目录(默认 results/<视频名>)')
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.output_dir or os.path.join('results', name)
    os.makedirs(out_dir, exist_ok=True)

    if args.mask and os.path.exists(args.mask):
        lines_mask = np.load(args.mask)
        print(f'使用已有掩码: {args.mask}')
    else:
        lines_mask = build_lines_mask(args.video)
        np.save(os.path.join(out_dir, 'lines_mask.npy'), lines_mask)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = args.frame_index if args.frame_index is not None else max(total // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'无法读取视频第 {idx} 帧')

    order_corners([(440, 269), (838, 269), (974, 642), (311, 642)])  # 排序函数自检

    best = None
    for seed in [0, 1]:
        H, score, diag = fit(lines_mask, seed=seed)
        if best is None or score > best[1]:
            best = (H, score, diag)
    H, score, _ = best
    print(f'最佳拟合得分: {score:.3f}')

    H, cs = canonicalize(H)
    print('规范化后四角(模型顺序):', cs.round(0).tolist())

    save_overlay(H, frame, os.path.join(out_dir, 'fit_overlay.jpg'))
    np.save(os.path.join(out_dir, 'H.npy'), H)

    model_corners = corners_from_H(H)
    net_uv = apply_H(np.linalg.inv(H), [(CW / 2, CL / 2)])
    mid_height = int(net_uv[0][1])

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'badminton_analysis'))
    from court.mapper import compute_expanded_roi
    roi_corners = compute_expanded_roi([tuple(c) for c in model_corners], frame.shape)

    # system.py 读取格式: 第1行 corners, 第2行 roi_corners(会被重算), 第3行 mid_height
    with open(os.path.join(out_dir, 'court_annotations.txt'), 'w') as f:
        f.write(f'corners={[(int(x), int(y)) for x, y in model_corners]}\n')
        f.write(f'roi_corners={[(int(x), int(y)) for x, y in roi_corners]}\n')
        f.write(f'mid_height={mid_height}\n')
    print('corners (模型顺序 (0,0),(6.1,0),(6.1,13.4),(0,13.4)):',
          [(int(x), int(y)) for x, y in model_corners])
    print(f'mid_height: {mid_height}')
    print(f'标定完成, 已写入 {os.path.join(out_dir, "court_annotations.txt")}')


if __name__ == '__main__':
    main()
