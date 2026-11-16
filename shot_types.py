"""击球类型识别(杀球/吊球/高远/平抽)与回合转移统计

方法: 球检测点按帧间隙切成 run -> 按"下落转上升/速度突变"判定击球边界并合并 run
      -> 对原始点计算弧线几何特征 -> 规则分类。纯几何规则, 无需训练;
      阈值在低视角 720p/30fps 实测数据上校准 (见 docs/IMPROVEMENTS.md)
"""
import json
import os

import numpy as np

FPS_DEFAULT = 30.0
RUN_GAP = 2         # 相邻检测间隔超过该帧数 -> 新 run
BRIDGE_GAP = 20     # run 间隔不超过该帧数且运动连续 -> 属同一条弧线
MIN_PTS = 4         # 一条弧线至少需要的数据点
SHOT_TYPES = ['杀球/扑球', '吊球', '高远球/挑球', '平抽', '未识别']


def load_records(detections_path):
    out = []
    with open(detections_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _get_mapper(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        meta = json.load(open(meta_path))
        from badminton_analysis.court.mapper import CourtMapper
        return CourtMapper([tuple(c) for c in meta['court']['corners']])
    except Exception:
        return None


def _court(pt, mapper):
    if mapper is None:
        return None
    try:
        c = mapper.image_to_court(tuple(pt))
        return [float(c[0]), float(c[1])]
    except Exception:
        return None


def split_runs(pts):
    """[(frame,u,v)] -> 连续 run 列表(间隙 > RUN_GAP 即断开)"""
    runs, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] <= RUN_GAP:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return runs


def _vel(seq):
    """相邻点速度(px/帧), 长度 n-1"""
    if len(seq) < 2:
        return np.array([])
    arr = np.array(seq, float)
    dt = np.diff(arr[:, 0])
    dv = np.diff(arr[:, 1:3], axis=0)
    return dv / np.maximum(dt[:, None], 1)


def _falling(v, thr=1.0):
    return len(v) and v[-1][1] > thr  # 图像 y 向下, dy>0 为下落


def _rising(v, thr=1.0):
    return len(v) and v[-1][1] < -thr


def merge_runs(runs):
    """run 序列 -> 弧线列表; 边界即击球帧(取相邻 run 中点帧)"""
    arcs = []
    cur = [runs[0]]
    for a, b in zip(runs, runs[1:]):
        gap = b[0][0] - a[-1][0]
        va, vb = _vel(a), _vel(b)
        if len(va) < 1 or len(vb) < 1:
            cur.append(b)
            continue
        # 击球主判据: 击球后球强力上升(出射侧), 入射侧只要求不在强上升;
        # 下压类击球(杀/扑)出射不上升, 靠速度突跳兜底
        flip = (vb[0][1] < -3.0 and va[-1][1] > -1.5)
        speed_jump = np.linalg.norm(vb[0]) > 2.5 * max(np.linalg.norm(va[-1]), 15)
        if gap > BRIDGE_GAP or flip or speed_jump:
            arcs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    arcs.append(cur)
    return arcs


def split_within_run(run):
    """run 内部: 下落转上升的局部最低点 -> 切成多段弧线"""
    if len(run) < 5:
        return [run]
    v = _vel(run)
    cuts = [0]
    for i in range(1, len(v)):
        # 击球后强力上升的翻转点(过滤检测抖动)
        if v[i][1] < -2.5 and v[i - 1][1] > -1.5 and i - cuts[-1] >= 3:
            cuts.append(i)
    cuts.append(len(run))
    return [run[a:b] for a, b in zip(cuts, cuts[1:]) if b - a >= 2]


def arc_features(pts, mapper, fps):
    """原始点弧线 -> 特征"""
    if len(pts) < MIN_PTS:
        return None
    arr = np.array(pts, float)
    fs, us, vs = arr[:, 0], arr[:, 1], arr[:, 2]
    dur = (fs[-1] - fs[0]) / fps
    if dur < 0.12:
        return None
    chord = float(np.hypot(us[-1] - us[0], vs[-1] - vs[0]))
    t = (fs - fs[0]) / max(fs[-1] - fs[0], 1)
    chord_y = vs[0] + t * (vs[-1] - vs[0])
    bulges = chord_y - vs
    apex_i = int(np.argmax(bulges))
    bulge_rel = float(bulges[apex_i]) / max(chord, 1.0)
    apex_frac = float(t[apex_i])

    def slope(lo, hi):
        n = len(fs)
        i0, i1 = int(n * lo), max(int(n * hi), int(n * lo) + 2)
        if i1 - i0 < 2:
            return 0.0
        return float(np.polyfit(fs[i0:i1] / fps, vs[i0:i1], 1)[0])
    # 图像 y 向下: 斜率为正 = 球在下行(下坠)。统一用"下坠速度"表示, 正值
    descent_end = slope(0.6, 1.0)
    rise_start = -slope(0.0, 0.4)
    dts = np.diff(fs) / fps
    dys = np.diff(vs)
    vels = dys / np.maximum(dts, 1e-6)
    # 次大值抗单帧抖动
    srt = np.sort(vels)
    peak_descent = float(srt[-2]) if len(srt) >= 2 else (float(srt[-1]) if len(srt) else 0.0)
    peak_rise = float(-srt[1]) if len(srt) >= 2 else (float(-srt[0]) if len(srt) else 0.0)
    # 二维速度(px/s): 水平为主, 平抽的关键特征
    dxy = np.diff(arr[:, 1:3], axis=0)
    sp2 = np.linalg.norm(dxy, axis=1) / np.maximum(dts, 1e-6)
    srt2 = np.sort(sp2)
    mean_speed = float(chord / dur)
    peak_speed = float(srt2[-2]) if len(srt2) >= 2 else (float(srt2[-1]) if len(srt2) else 0.0)
    f = {
        'n_pts': int(len(pts)), 'dur': round(dur, 2),
        'frame0': int(fs[0]), 'frame1': int(fs[-1]),
        'img0': [float(us[0]), float(vs[0])], 'img1': [float(us[-1]), float(vs[-1])],
        'chord_px': round(chord, 1), 'bulge_rel': round(bulge_rel, 3),
        'apex_frac': round(apex_frac, 2),
        'descent_end': int(descent_end), 'rise_start': int(rise_start),
        'peak_descent': int(peak_descent), 'peak_rise': int(peak_rise),
        'mean_speed': int(mean_speed), 'peak_speed': int(peak_speed),
    }
    # 弧线最低点(最接近地面, 地面投影误差最小), 仅信息参考
    low_i = int(np.argmax(vs))
    c_low = _court([us[low_i], vs[low_i]], mapper)
    if c_low and -3.0 <= c_low[0] <= 9.0 and -8.0 <= c_low[1] <= 21.0:
        f['low_court'] = [round(c_low[0], 2), round(c_low[1], 2)]
    return f


def classify(f):
    """规则分类: 返回 (类型, 置信度 0-1)。下坠速度正值=向下"""
    dur, br, af = f['dur'], f['bulge_rel'], f['apex_frac']
    pd, pr = f['peak_descent'], f['peak_rise']
    de = f['descent_end']
    ms, ps = f['mean_speed'], f['peak_speed']
    if ms < 60:  # 静止/悬停(捡球、发球等待) 非飞行段
        return '未识别', 0.1
    # 快速下坠且顶点在开头: 杀/扑 (从严, 保精度)
    if af < 0.25 and br < 0.13 and pd > 600 and dur < 0.7:
        return '杀球/扑球', 0.75
    # 高弧线长时: 高远/挑
    if br > 0.30 and dur > 0.85 and pr > 200:
        return '高远球/挑球', 0.7
    # 快且平: 平抽 (长多拍平抽可 >1s)
    if br < 0.15 and ms > 450 and ps > 700 and dur < 1.5:
        return '平抽', 0.6
    # 短弧落前场: 吊球/网前 (含中等凸比)
    if dur < 0.8 and f.get('land_zone') == 'front' and br < 0.32:
        return '吊球', 0.55
    if dur < 1.05 and f.get('land_zone') == 'front':
        return '吊球', 0.5
    if br > 0.25:
        return '高远球/挑球', 0.45
    return '未识别', 0.3


def hit_owner(frame, ball_img, recs_by_frame):
    """击球者 = 击球帧附近手/身体离球最近的球员"""
    r = None
    for d in (0, -1, 1, -2, 2, -3, 3):
        r = recs_by_frame.get(frame + d)
        if r:
            break
    if not r:
        return None, 999.0
    best, bd = None, 1e9
    for side in ('upper', 'lower'):
        p = r['players'].get(side)
        if not p or not p.get('image'):
            continue
        pi = p['image']
        for hand in (p.get('hands') or {}).values():
            if hand:
                d = float(np.hypot(hand[0] - ball_img[0], hand[1] - ball_img[1]))
                if d < bd:
                    best, bd = side, d
        d = float(np.hypot(pi[0] - ball_img[0], pi[1] - ball_img[1]))
        if d < bd * 0.8:
            best, bd = side, d
    return best, round(bd, 1)


def _land_zone(y_court, side):
    """按到网距离分前后中: side 为该半场所属球员"""
    dfn = (HALF - y_court) if side == 'upper' else (y_court - HALF)
    return 'front' if dfn < 2.23 else ('mid' if dfn < 4.47 else 'back')


HALF = 6.7


def analyze(recs, mapper=None, fps=FPS_DEFAULT):
    pts = []
    for r in recs:
        s = r.get('shuttlecock')
        if s and s.get('image'):
            pts.append((int(r['frame']), float(s['image'][0]), float(s['image'][1])))
    if len(pts) < 30:
        return {'arcs': [], 'stats': {}}
    pts.sort(key=lambda p: p[0])
    runs = [r for r in split_runs(pts) if len(r) >= 2]
    runs = [seg for r in runs for seg in split_within_run(r)]
    runs = [r for r in runs if len(r) >= 2]
    if not runs:
        return {'arcs': [], 'stats': {}}
    arcs_runs = merge_runs(runs)
    recs_by_frame = {int(r['frame']): r for r in recs}
    arcs, prev_type, prev_owner = [], None, None
    for group in arcs_runs:
        flat = [p for run in group for p in run]
        f = arc_features(flat, mapper, fps)
        if not f:
            continue
        st, conf = classify(f)
        owner, d = hit_owner(f['frame0'], f['img0'], recs_by_frame)
        if d > 420:  # 击球帧附近没有球员靠近 => 归属不可信(近端透视像素距离更大)
            owner = None
        f.update(type=st, conf=conf, owner=owner, owner_dist=d,
                 incoming_type=prev_type, incoming_owner=prev_owner)
        arcs.append(f)
        prev_type, prev_owner = st, owner
    # 落点区: 空中球的地面投影不可信, 用下一拍击球者的站位(地面真值)标注
    for i, a in enumerate(arcs):
        nxt = arcs[i + 1] if i + 1 < len(arcs) else None
        if nxt and nxt.get('owner'):
            r = recs_by_frame.get(nxt['frame0'])
            pos = (r or {}).get('players', {}).get(nxt['owner'], {}).get('court')
            if pos:
                a['land_zone'] = _land_zone(pos[1], nxt['owner'])
                a['land_court'] = [round(float(pos[0]), 2), round(float(pos[1]), 2)]
                a['receiver'] = nxt['owner']
        if 'land_zone' not in a and a.get('low_court'):
            y = a['low_court'][1]
            side = 'upper' if y < HALF else 'lower'
            a['land_zone'] = _land_zone(y, side)
    return {'arcs': arcs, 'stats': summarize_stats(arcs)}


def summarize_stats(arcs):
    """per-player 击球类型分布 + 来球->回球转移"""
    stats = {}
    for side in ('upper', 'lower'):
        mine = [a for a in arcs if a['owner'] == side and a['type'] != '未识别']
        cnt = {}
        for a in mine:
            cnt[a['type']] = cnt.get(a['type'], 0) + 1
        trans = {}
        for a in mine:
            inc = a.get('incoming_type')
            if inc and inc != '未识别':
                key = f"{inc}->{a['type']}"
                trans[key] = trans.get(key, 0) + 1
        total = sum(cnt.values())
        stats[side] = {
            'total': total, 'counts': cnt,
            'shares': {k: round(v / total, 2) for k, v in cnt.items()} if total else {},
            'transitions': trans,
        }
    return stats


def summarize_text(shot):
    """击球统计 -> 中文摘要(嵌进 LLM 提示词)"""
    if not shot.get('stats'):
        return '击球类型识别: 无有效数据'
    arcs = shot['arcs']
    valid = sum(s['total'] for s in shot['stats'].values())
    owned = sum(1 for a in arcs if a.get('owner'))
    alt = sum(1 for a, b in zip(arcs, arcs[1:])
              if a.get('owner') and b.get('owner') and a['owner'] != b['owner'])
    alt_rate = alt / max(1, sum(1 for a, b in zip(arcs, arcs[1:])
                                if a.get('owner') and b.get('owner')))
    L = [f"击球类型识别(规则分类, 共 {len(arcs)} 条弧线, {valid} 次有效出手, "
         f"轮转交替率 {alt_rate:.0%}):"]
    for side in ('upper', 'lower'):
        s = shot['stats'].get(side, {})
        if not s.get('total'):
            continue
        name = '远端球员' if side == 'upper' else '近端球员'
        shares = ', '.join(f"{k} {v:.0%}" for k, v in sorted(
            s['shares'].items(), key=lambda kv: -kv[1]))
        L.append(f"【{name}】 出手 {s['total']} 次: {shares}")
        top = sorted(s['transitions'].items(), key=lambda kv: -kv[1])[:4]
        if top:
            L.append('  常见应对(来球->回球): ' + '; '.join(f"{k} {v}次" for k, v in top))
    return '\n'.join(L)


def plot_overview(shot, out_path):
    """全部弧线按类别着色叠画, 供人工/视觉抽查"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
               '/usr/share/fonts/truetype/arphic/uming.ttc'):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=fp).get_name()]
            break
    plt.rcParams['axes.unicode_minus'] = False
    colors = {'杀球/扑球': 'red', '吊球': 'orange', '高远球/挑球': 'tab:blue',
              '平抽': 'green', '未识别': 'lightgray'}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for a in shot['arcs']:
        c = colors.get(a['type'], 'gray')
        ax.plot([a['img0'][0], a['img1'][0]], [a['img0'][1], a['img1'][1]],
                color=c, lw=1.2, alpha=0.75)
        ax.plot(a['img0'][0], a['img0'][1], 'o', color=c, ms=3, alpha=0.9)
    handles = [plt.Line2D([], [], color=c, label=t) for t, c in colors.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.invert_yaxis()
    ax.set_title('击球弧线分类总览 (圆点=出手点)')
    ax.set_xlabel('图像 x (px)')
    ax.set_ylabel('图像 y (px)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def extract_shot_analysis(detections_path, out_dir=None):
    """style_coach 调用入口"""
    recs = load_records(detections_path)
    meta_path = os.path.join(os.path.dirname(detections_path), 'metadata.json')
    mapper = _get_mapper(meta_path)
    fps = FPS_DEFAULT
    if os.path.exists(meta_path):
        try:
            fps = float(json.load(open(meta_path)).get('video', {}).get('fps', FPS_DEFAULT))
        except Exception:
            pass
    shot = analyze(recs, mapper, fps)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'shot_stats.json'), 'w') as fp:
            json.dump({'arcs': shot['arcs'], 'stats': shot['stats']},
                      fp, ensure_ascii=False, indent=1)
        try:
            plot_overview(shot, os.path.join(out_dir, 'arcs_overview.png'))
        except Exception:
            pass
    return shot


if __name__ == '__main__':
    import sys
    dets = sys.argv[1] if len(sys.argv) > 1 else 'results/low_angle_linDanCup/detections.jsonl'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(dets), 'shot_analysis')
    shot = extract_shot_analysis(dets, out_dir)
    print(summarize_text(shot))
    print('\n弧线明细(前 30 条):')
    for a in shot['arcs'][:30]:
        land = a.get('land_zone', '?')
        via = '接球者' if a.get('receiver') else ('低点' if a.get('low_court') else '')
        print(f"  f{a['frame0']:>5}-{a['frame1']:<5} {a['dur']:.2f}s 凸比{a['bulge_rel']:.2f} "
              f"顶位{a['apex_frac']:.2f} 峰降{a['peak_descent']:>5} => {a['type']} "
              f"({a['owner'] or '?'}, d={a['owner_dist']}) 落区={land}{via}")
