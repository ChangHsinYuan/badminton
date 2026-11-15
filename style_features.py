"""从 detections.jsonl 提取球风/跑位/步法特征

输入: results/<视频名>/detections.jsonl + metadata.json(含角点, 用于手部坐标换算)
输出: 特征 dict + 中文摘要(供 LLM 提示词使用)
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'badminton_analysis'))

HALF_LEN = 13.4 / 2  # 半场长 6.7m
COURT_W = 6.1
SPRINT_MS = 4.0      # 冲刺速度阈值 m/s
TURN_DEG = 60.0      # 变向角度阈值
SIDES = ('upper', 'lower')
SIDE_NAME = {'upper': '远端球员', 'lower': '近端球员'}


def load_records(detections_path):
    recs = []
    with open(detections_path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def dist_from_net(y, side):
    """到网线的距离(米): upper 半场 y∈[0,6.7] 底线在 y=0; lower 半场 y∈[6.7,13.4] 底线在 y=13.4"""
    return (HALF_LEN - y) if side == 'upper' else (y - HALF_LEN)


def zone_of(x, y, side):
    """半场 3x3 分区: 返回 (深度区, 宽度区), 深度 0=网前 1=中场 2=后场(底线端)"""
    dfn = dist_from_net(y, side)
    d = min(int(dfn / (HALF_LEN / 3)), 2)
    w = min(int(x / (COURT_W / 3)), 2)
    return d, w


def player_features(recs, side, mapper=None):
    court, speeds, hands_img, base_img = [], [], {'left': [], 'right': []}, []
    reach_m = []
    for r in recs:
        p = (r.get('players') or {}).get(side)
        if not p:
            continue
        c = p.get('court')
        if c and all(np.isfinite(c)):
            court.append(c)
            speeds.append(p.get('speed') or 0.0)
        if p.get('image'):
            base_img.append(p['image'])
        h = p.get('hands') or {}
        for k in ('left', 'right'):
            hk = h.get(k)
            # rtmpose 低置信关键点会被置 (0,0), 需剔除
            if hk and hk[0] > 1 and hk[1] > 1:
                hands_img[k].append(hk)
                if mapper is not None and p.get('image'):
                    try:
                        hc = mapper.image_to_court(hk)
                        bc = mapper.image_to_court(p['image'])
                        d = float(np.hypot(hc[0] - bc[0], hc[1] - bc[1]))
                        if 0.3 < d < 2.5:  # 物理合理范围(躯干到拍击点)
                            reach_m.append(d)
                    except Exception:
                        pass

    if len(court) < 30:
        return None
    court = np.array(court)
    speeds = np.array(speeds)
    dt = 1.0 / 30.0
    x, y = court[:, 0], court[:, 1]
    dfn = np.array([dist_from_net(v, side) for v in y])  # 到网距离: 0=网前, 6.7=底线

    # 分区占比
    zones = np.zeros((3, 3))
    for xi, yi in zip(x, y):
        d, w = zone_of(xi, yi, side)
        zones[d][w] += 1
    zones /= zones.sum()

    # 步法: 平滑后以 2 帧步长计算速度向量, 抑制检测抖动导致的假变向
    kern = np.ones(5) / 5
    cs = np.stack([np.convolve(x, kern, 'valid'), np.convolve(y, kern, 'valid')], 1)
    stride = 2
    v = (cs[stride:] - cs[:-stride]) / (stride * dt)
    sp = np.linalg.norm(v, axis=1)
    turn_rate = 0.0
    if len(sp) > 2:
        dot = (v[:-1] * v[1:]).sum(1) / np.maximum(sp[:-1] * sp[1:], 1e-6)
        ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        moving = (sp[:-1] > 1.2) & (sp[1:] > 1.2)
        turn_rate = float((ang[moving] > 45).mean() * (30.0 / stride)) if moving.any() else 0.0  # 次/秒

    # 冲刺事件: 连续超阈值帧合并为一次
    sprint_frames = speeds > SPRINT_MS
    sprints = int(np.sum(np.diff(sprint_frames.astype(int)) == 1) + (1 if sprint_frames[0] else 0))

    hand_pref = '未知'
    nl, nr = len(hands_img['left']), len(hands_img['right'])
    if nl + nr > 0:
        hand_pref = '右手' if nr > nl else ('左手' if nl > nr * 1.5 else '双手均势')

    f = {
        'side': side, 'name': SIDE_NAME[side],
        'frames': len(court), 'duration_min': len(court) / 30 / 60,
        'base': [float(np.median(x)), float(np.median(y))],
        'net_dist_mean': float(dfn.mean()), 'depth_std': float(dfn.std()),
        'x_range': float(np.percentile(x, 95) - np.percentile(x, 5)),
        'x_std': float(x.std()),
        'speed_mean': float(speeds.mean()), 'speed_max': float(speeds.max()),
        'speed_p90': float(np.percentile(speeds, 90)),
        'sprints': sprints, 'sprint_share': float(sprint_frames.mean()),
        'distance_m': float((speeds * dt).sum()),
        'turn_per_sec': turn_rate,
        'zones': zones.round(3).tolist(),
        'front_share': float((dfn < HALF_LEN / 3).mean()),
        'back_share': float((dfn > HALF_LEN * 2 / 3).mean()),
        'hand_pref': hand_pref,
        'reach_mean_m': float(np.median(reach_m)) if reach_m else None,
        'reach_max_m': float(np.percentile(reach_m, 95)) if reach_m else None,
    }
    # 风格指数(0~100 规则打分)
    sSprint = min(f['sprint_share'] / 0.15, 1.0)
    sFront = f['front_share']
    sBack = f['back_share']
    sSpeed = min(f['speed_p90'] / 5.0, 1.0)
    f['style_index'] = {
        '进攻性': int(100 * (0.5 * sSprint + 0.3 * sFront + 0.2 * sSpeed)),
        '防守韧性': int(100 * (0.5 * sBack + 0.3 * min(f['x_range'] / 5.0, 1.0) + 0.2 * min(f['distance_m'] / 200, 1.0))),
        '控球节奏': int(100 * (0.4 * (1 - sSprint) + 0.3 * min(turn_rate / 1.5, 1.0) + 0.3 * (1 - abs(f['net_dist_mean'] - HALF_LEN / 2) / (HALF_LEN / 2)))),
        '网前压迫': int(100 * min(sFront / 0.35, 1.0)),
    }
    return f


def shuttle_rhythm(recs):
    """击球节奏: 用球轨迹方向反转估计击球次数(图像空间)"""
    ys, xs, t = [], [], []
    for r in recs:
        s = r.get('shuttlecock')
        if s and s.get('image'):
            xs.append(s['image'][0]); ys.append(s['image'][1]); t.append(r['frame'])
    if len(ys) < 50:
        return {'hits': 0, 'pace_per_sec': 0.0, 'active_ratio': 0.0}
    ys, xs, t = map(np.array, (ys, xs, t))
    # 平滑后找 dy 反号且幅度足够的时刻
    k = 5
    kern = np.ones(k) / k
    ys_s = np.convolve(ys, kern, 'valid')
    dy = np.diff(ys_s)
    flips = []
    for i in range(1, len(dy)):
        if dy[i - 1] * dy[i] < 0 and abs(dy[i - 1]) > 1.5 and abs(dy[i]) > 1.5:
            if not flips or t[i] - flips[-1] > 10:  # 间隔 >1/3 秒才算新击球
                flips.append(t[i])
    total_frames = recs[-1]['frame'] - recs[0]['frame'] + 1
    active = len(t) / max(total_frames, 1)
    span = (t[-1] - t[0]) / 30.0
    return {'hits': len(flips), 'pace_per_sec': round(len(flips) / max(span, 1), 2),
            'active_ratio': round(active, 2)}


def extract(detections_path):
    recs = load_records(detections_path)
    # 从同目录 metadata.json 拿角点做手部坐标换算
    mapper = None
    meta_path = os.path.join(os.path.dirname(detections_path), 'metadata.json')
    if os.path.exists(meta_path):
        try:
            from court.mapper import CourtMapper
            corners = json.load(open(meta_path))['court']['corners']
            mapper = CourtMapper([tuple(c) for c in corners])
        except Exception:
            mapper = None
    out = {'players': {}, 'rhythm': shuttle_rhythm(recs)}
    for side in SIDES:
        f = player_features(recs, side, mapper)
        if f:
            out['players'][side] = f
    out['n_frames'] = len(recs)
    return out


def summarize(features):
    """特征 -> 中文摘要表(嵌进 LLM 提示词)"""
    L = [f"视频共 {features['n_frames']} 帧, 估计击球 {features['rhythm']['hits']} 次, "
         f"回合节奏 {features['rhythm']['pace_per_sec']} 次/秒, 球在场时间占比 {features['rhythm']['active_ratio']:.0%}"]
    ZN = ['网前', '中场', '后场']
    WN = ['左区', '中路', '右区']
    for side in SIDES:
        f = features['players'].get(side)
        if not f:
            continue
        L.append(f"\n【{f['name']}】 出镜 {f['frames']} 帧 ({f['duration_min']:.1f} 分钟), 持拍手: {f['hand_pref']}")
        zt = ', '.join(f"{ZN[d]}{WN[w]} {f['zones'][d][w]:.0%}" for d in range(3) for w in range(3) if f['zones'][d][w] >= 0.03)
        L.append(f"  站位热区: {zt}")
        L.append(f"  基准位置: ({f['base'][0]:.1f}, {f['base'][1]:.1f}) m, 平均距网 {f['net_dist_mean']:.2f} m, "
                 f"前后移动幅度 σ={f['depth_std']:.2f} m, 左右覆盖 {f['x_range']:.1f} m")
        L.append(f"  速度: 平均 {f['speed_mean']:.2f} / P90 {f['speed_p90']:.2f} / 峰值 {f['speed_max']:.2f} m/s, "
                 f"冲刺 {f['sprints']} 次 (占比 {f['sprint_share']:.1%}), 总移动 {f['distance_m']:.0f} m")
        L.append(f"  变向率 {f['turn_per_sec']:.2f} 次/秒"
                 + (f", 拍击平均伸展 {f['reach_mean_m']:.2f} m (P95 {f['reach_max_m']:.2f} m)" if f['reach_mean_m'] else ''))
        si = f['style_index']
        L.append(f"  风格指数: 进攻性 {si['进攻性']} | 防守韧性 {si['防守韧性']} | 控球节奏 {si['控球节奏']} | 网前压迫 {si['网前压迫']}")
    return '\n'.join(L)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/low_angle_linDanCup/detections.jsonl'
    feats = extract(path)
    print(summarize(feats))
