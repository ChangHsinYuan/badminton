"""AI 羽球教练: 特征数据 + 关键帧 -> 大模型生成球风分析报告

用法:
  python style_coach.py --name low_angle_linDanCup
  python style_coach.py --name xxx --skip-frames   # 不做视觉动作点评
  python style_coach.py --name xxx --player-hint lower=桃田贤斗  # 已知球员身份
输出: results/<name>/style_report.md
依赖: config/llm_api.json (LLM 网关配置, 模板见 config/llm_api.example.json)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import style_features
import shot_types
import pro_profiles
from ask_vision import ask, config

TEXT_PROMPT = """你是一名专业羽毛球教练, 请基于以下由视频分析系统自动提取的比赛数据, 撰写一份球员分析报告。

数据说明:
- 场地全长 13.4m × 宽 6.1m, upper=远端球员(画面上方, 半场 y∈[0,6.7], y=0 为其底线), lower=近端球员(画面下方, 半场 y∈[6.7,13.4], y=13.4 为其底线), 网在 y=6.7
- "挥拍伸展指数"为地面投影估计, 近端球员因透视原因系统性偏大, 仅作两名球员之间的相对比较
- 若为双打比赛, upper/lower 可能是前后站位两名球员的合并轨迹, 请在报告开头注明此不确定性
- 站位热区按网前/中场/后场 × 左/中/右 九宫格统计
- 击球类型由轨迹几何规则识别, 覆盖率有限, 分布仅作参考, 不要对单次击球下断言

{summary}

{hint_block}

请用中文 Markdown 撰写报告, 包含以下四个部分, 每部分都要引用具体数据支撑观点:

## 一、球风画像
分别给两名球员定性(如 进攻型/防守反击型/控球型/网前压迫型), 引用速度、冲刺、站位热区、击球类型分布数据。

## 二、跑位与步法分析
分析基准位置是否合理(单打应居中偏后)、左右覆盖是否充分、变向与回位节奏, 指出跑位上的问题。

## 三、职业风格相似度(计算结果解读)
下面是系统用加权特征向量最近邻算出的相似度(基于特征卡库, 可复现), 请解读这些计算结果: 结合"最大差异"说明为什么相似度高/低, 只评价风格倾向而非水平。{pro_block}
若球员身份已知(见身份提示), 请把计算结果与该球员的公认技术特点对照验证, 指出哪些指标与公认印象一致、哪些可能受数据质量影响。

## 四、训练建议
每人 3 条具体可执行的训练建议, 针对数据暴露的短板(如网前占有率低、冲刺后回位慢、某类击球占比异常等)。

语气专业但易懂, 面向业余爱好者读者。不要编造数据里不存在的信息。"""

VISION_PROMPT = """这是羽毛球比赛视频的关键帧, 骨架为姿态估计结果。请从专业角度点评画面中球员的技术动作:
1. 重心与站位是否合理 2. 引拍/挥拍/步法形态 3. 可见的技术问题。
简短作答(100字内), 若画面模糊或多人重叠请说明。"""


def ask_with_retry(prompt, img_path=None, retries=3, model=None, max_tokens=None, **kw):
    """默认用视觉模型做文本分析(输出更干净), 推理型模型易把思考过程混入正文"""
    cfg = config()
    model = model or cfg.get('vision_model')
    max_tokens = max_tokens or max(cfg.get('max_tokens', 2000), 4000)
    last = None
    for i in range(retries):
        try:
            return ask(prompt, img_path=img_path, model=model, max_tokens=max_tokens, **kw)
        except Exception as e:
            last = e
            print(f'  LLM 调用失败({i + 1}/{retries}): {e}')
            time.sleep(3 * (i + 1))
    raise RuntimeError(f'LLM 调用失败: {last}')


def _hint_block(hints):
    """身份提示 -> 提示词段落"""
    if not hints:
        return ''
    L = ['身份提示(用户已知信息, 可信):']
    for side, who in hints.items():
        nm = '远端球员(upper)' if side == 'upper' else '近端球员(lower)'
        L.append(f'- {nm} 是: {who}')
    return '\n'.join(L)


def _pro_block(detections, hints, matches=None):
    """特征卡最近邻计算 -> 提示词段落(计算数据, LLM 只解读不发明)"""
    L = []
    for side in ('upper', 'lower'):
        if matches is not None and side in matches:
            res = matches[side]
        else:
            try:
                res = pro_profiles.match_side(detections, side)
            except SystemExit:
                res = []
        who = (hints or {}).get(side)
        nm = '远端球员(upper)' if side == 'upper' else '近端球员(lower)'
        if who:
            nm += f' = {who}'
        if not res:
            L.append(f'【{nm}】特征卡库为空, 无法计算; 请基于数据本身谨慎给出风格对照。')
            continue
        L.append(f'【{nm}】最近邻计算结果:')
        for r in res:
            diffs = '; '.join(f"{d['feature']} 本方={d['this']}/卡={d['that']}"
                              for d in r['diff'])
            L.append(f"- {r['name']}(来源:{r['source']}) 相似度 {r['similarity']}%, "
                     f"共同特征 {r['n_common']} 个, 主要差异: {diffs}")
    return '\n'.join(L)


def build_report(name, detections, skip_frames=False, max_vision=3, model=None,
                 hints=None):
    feats = style_features.extract(detections)
    # 特征卡匹配: >=90% 命中实测卡时, 持拍手继承卡片记录
    # (近球投票依赖相机几何, 跨机位不稳; 同一人持拍手不变, 卡片更可信)
    matches, hand_hint = {}, {}
    for side in ('upper', 'lower'):
        try:
            matches[side] = pro_profiles.match_side(detections, side)
        except SystemExit:
            matches[side] = []
        for r in matches[side]:
            if r['source'] != 'measured' or r['similarity'] < 90:
                continue
            hp = pro_profiles.card_hand_pref(r['name'])
            if hp in ('左手', '右手'):
                hand_hint[side] = (hp, r['name'])
            break
    summary = style_features.summarize(feats, hand_hint)
    out_dir = os.path.dirname(detections)
    shot = shot_types.extract_shot_analysis(
        detections, os.path.join(out_dir, 'shot_analysis'))
    shot_txt = shot_types.summarize_text(shot)
    print('特征与击球类型提取完成, 请求文本模型分析...')
    hint_block = _hint_block(hints)
    pro_block = _pro_block(detections, hints, matches)
    body = ask_with_retry(TEXT_PROMPT.format(summary=summary + '\n\n' + shot_txt,
                                             hint_block=hint_block,
                                             pro_block=pro_block), model=model)

    parts = [f'# 羽球 AI 教练分析报告 — {name}\n',
             f'> 数据来源: `{detections}` | 关键指标摘要:\n\n```text\n{summary}\n```\n',
             body]

    if not skip_frames:
        meta = os.path.join(os.path.dirname(detections), 'metadata.json')
        video = json.load(open(meta))['video']['path'] if os.path.exists(meta) else None
        if video and os.path.exists(video):
            import keyframes
            frames_dir = os.path.join(os.path.dirname(detections), 'coach_frames')
            frames = keyframes.draw_keyframes(video, detections, frames_dir)
            # 优先多样标签: swing/sprint/ball_contact 各取一
            picked, seen = [], set()
            for p, tag in frames:
                cat = tag.split('_')[-1]
                if cat not in seen:
                    picked.append((p, tag))
                    seen.add(cat)
            picked = (picked + [f for f in frames if f not in picked])[:max_vision]
            reviews = []
            for p, tag in picked:
                print(f'视觉模型点评: {os.path.basename(p)}')
                try:
                    txt = ask_with_retry(VISION_PROMPT, img_path=p)
                    reviews.append((p, tag, txt))
                except RuntimeError as e:
                    print(e)
            if reviews:
                parts.append('\n## 五、关键技术帧动作点评\n')
                for p, tag, txt in reviews:
                    parts.append(f'### {os.path.basename(p)} ({tag})\n\n{txt}\n')

    report = '\n'.join(parts)
    out = os.path.join(os.path.dirname(detections), 'style_report.md')
    with open(out, 'w') as f:
        f.write(report)
    print(f'\n报告已保存: {out}')
    return out, report


def _parse_hints(s):
    """'lower=桃田贤斗,upper=林丹' -> {'lower': '桃田贤斗', 'upper': '林丹'}"""
    hints = {}
    for part in (s or '').split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ('upper', 'far', '远端'):
                k = 'upper'
            elif k in ('lower', 'near', '近端'):
                k = 'lower'
            if k in ('upper', 'lower') and v:
                hints[k] = v
    return hints


def main():
    ap = argparse.ArgumentParser(description='AI 羽球教练分析')
    ap.add_argument('--name', default='low_angle_linDanCup', help='results 下的视频任务名')
    ap.add_argument('--detections', default=None, help='detections.jsonl 路径(默认 results/<name>/)')
    ap.add_argument('--skip-frames', action='store_true', help='跳过关键帧视觉点评')
    ap.add_argument('--max-frames', type=int, default=3, help='视觉点评帧数上限')
    ap.add_argument('--model', default=None, help='LLM 模型名(默认用 config 的 vision_model)')
    ap.add_argument('--player-hint', default=None, metavar='SIDE=姓名',
                    help="已知球员身份, 如 'lower=桃田贤斗,upper=林丹'")
    args = ap.parse_args()
    det = args.detections or os.path.join('results', args.name, 'detections.jsonl')
    if not os.path.exists(det):
        sys.exit(f'找不到 {det}, 请先运行 main.py 完成分析')
    hints = _parse_hints(args.player_hint)
    _, report = build_report(args.name, det, args.skip_frames, args.max_frames,
                             args.model, hints)
    print('\n' + '=' * 60 + '\n')
    print(report)


if __name__ == '__main__':
    main()
