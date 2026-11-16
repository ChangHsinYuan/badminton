"""职业风格特征卡与最近邻匹配

相似度 = 加权特征向量距离(可复现的计算), 替代 LLM 凭印象的"风格联想"。
特征卡来源两种:
  1) 实测: 对某段比赛视频跑 --build, 由本管线特征提取生成 (source=measured)
  2) 模板: 手工归纳的参考卡, 仅作占位对照 (source=curated, 不参与精确断言)
用法:
  python pro_profiles.py --build results/<名>/detections.jsonl --name 桃田贤斗 --side lower
  python pro_profiles.py --match results/<名>/detections.jsonl --side lower
"""
import argparse
import json
import os
import re

import numpy as np

from style_features import extract as extract_style, SIDES
from shot_types import extract_shot_analysis

PROFILES_DIR = 'profiles'

# 特征名: (归一化上限, 权重); 缺失特征自动跳过并重归一
SPEC = {
    '进攻性': (100, 1.0), '防守韧性': (100, 1.0),
    '控球节奏': (100, 1.0), '网前压迫': (100, 1.0),
    'front_share': (1.0, 0.8), 'mid_share': (1.0, 0.8), 'back_share': (1.0, 0.8),
    'speed_mean': (5.0, 0.6), 'speed_p90': (8.0, 0.6),
    'sprint_share': (0.3, 0.5), 'turn_rate': (2.0, 0.5),
    'net_dist_mean': (6.7, 0.8),
    'shot_杀球/扑球': (1.0, 1.0), 'shot_吊球': (1.0, 1.0),
    'shot_高远球/挑球': (1.0, 1.0), 'shot_平抽': (1.0, 1.0),
}


def feature_vector(style_f, shot_stats):
    """style_features 单人特征 + shot_types 统计 -> 扁平特征向量"""
    v = {}
    for k in ('进攻性', '防守韧性', '控球节奏', '网前压迫'):
        if k in style_f.get('style_index', {}):
            v[k] = float(style_f['style_index'][k])
    z = style_f.get('zones')  # 3x3: 行=深度(0网前/1中场/2后场), 列=宽度
    if z:
        tot = sum(sum(r) for r in z) or 1
        v['front_share'] = sum(z[0]) / tot
        v['mid_share'] = sum(z[1]) / tot
        v['back_share'] = sum(z[2]) / tot
    for src, dst in (('speed_mean', 'speed_mean'), ('speed_p90', 'speed_p90'),
                     ('sprint_share', 'sprint_share'), ('turn_per_sec', 'turn_rate'),
                     ('net_dist_mean', 'net_dist_mean')):
        if src in style_f:
            v[dst] = float(style_f[src])
    total = sum(shot_stats.get('counts', {}).values()) or 0
    if total:
        for t, c in shot_stats['counts'].items():
            v[f'shot_{t}'] = c / total
    return v


def _normalize(vec):
    out = {}
    for k, (hi, _w) in SPEC.items():
        if k in vec:
            out[k] = float(vec[k]) / hi
    return out


def match(vec, profiles, topk=3):
    """返回 [(name, similarity%, 共同特征数, 主要差异), ...] 按相似度降序"""
    q = _normalize(vec)
    out = []
    for p in profiles:
        pv = _normalize(p.get('features', {}))
        common = [k for k in q if k in pv]
        if len(common) < 4:
            continue
        w = np.array([SPEC[k][1] for k in common])
        d = np.sqrt(np.sum(w * (np.array([q[k] for k in common]) -
                                np.array([pv[k] for k in common])) ** 2))
        dmax = np.sqrt(np.sum(w))
        sim = max(0.0, 1.0 - d / dmax)
        diffs = sorted(common, key=lambda k: -abs(q[k] - pv[k]) * SPEC[k][1])[:3]
        out.append({
            'name': p.get('name', '?'), 'source': p.get('source', '?'),
            'similarity': round(sim * 100, 1), 'n_common': len(common),
            'diff': [{'feature': k, 'this': round(q[k], 2), 'that': round(pv[k], 2)}
                     for k in diffs],
        })
    out.sort(key=lambda x: -x['similarity'])
    return out[:topk]


def load_profiles(d):
    ps = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                try:
                    ps.append(json.load(open(os.path.join(d, fn))))
                except Exception:
                    pass
    return ps


def build(detections_path, name, side, source='measured', out_dir=PROFILES_DIR):
    style = extract_style(detections_path)
    shot = extract_shot_analysis(detections_path)
    f = style['players'].get(side)
    if not f:
        raise SystemExit(f'该视频没有 {side} 侧球员特征')
    vec = feature_vector(f, shot['stats'].get(side, {}))
    slug = re.sub(r'\s+', '', name)
    prof = {
        'name': name, 'side': side, 'source': source,
        'video': os.path.basename(os.path.dirname(detections_path)),
        'features': {k: round(x, 3) for k, x in vec.items()},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{slug}.json')
    with open(path, 'w') as fp:
        json.dump(prof, fp, ensure_ascii=False, indent=1)
    print(f'特征卡已保存: {path}')
    print(json.dumps(prof['features'], ensure_ascii=False, indent=1))
    return prof


def match_side(detections_path, side, topk=3, out_dir=PROFILES_DIR):
    style = extract_style(detections_path)
    shot = extract_shot_analysis(detections_path)
    f = style['players'].get(side)
    if not f:
        raise SystemExit(f'该视频没有 {side} 侧球员特征')
    vec = feature_vector(f, shot['stats'].get(side, {}))
    profiles = load_profiles(out_dir)
    if not profiles:
        print('profiles/ 下没有特征卡, 先用 --build 生成')
        return []
    return match(vec, profiles, topk)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='职业风格特征卡构建与匹配')
    ap.add_argument('--detections', required=True)
    ap.add_argument('--build', metavar='NAME', help='把该侧球员特征存为特征卡')
    ap.add_argument('--side', default='lower', choices=SIDES)
    ap.add_argument('--source', default='measured', choices=['measured', 'curated'])
    ap.add_argument('--match', action='store_true', help='与 profiles/ 下特征卡匹配')
    args = ap.parse_args()
    if args.build:
        build(args.detections, args.build, args.side, args.source)
    if args.match or not args.build:
        res = match_side(args.detections, args.side)
        for r in res:
            print(f"{r['name']}({r['source']}): 相似度 {r['similarity']}% "
                  f"(共同特征 {r['n_common']}), 最大差异: "
                  + ', '.join(f"{d['feature']} 本={d['this']}/卡={d['that']}" for d in r['diff']))
