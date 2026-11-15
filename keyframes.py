"""关键时刻抽帧 + 骨架绘制, 供视觉模型做技术动作点评

选取规则:
  - 每名球员速度最高的 2 帧(冲刺/救球瞬间)
  - 每名球员挥拍伸展最大的 1 帧(拍击瞬间)
  - 球在画面中靠近该球员的 1 帧(回合接触球概率高)
去重后统一用 rtmlib 画骨架, 存 results/<视频名>/coach_frames/
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'badminton_analysis'))

BODY_CENTER = (11, 12)  # COCO17 双髋索引


def _hip_center(kp17):
    return kp17[list(BODY_CENTER), :2].mean(0)


def pick_moments(detections_path, max_frames=8):
    recs = []
    with open(detections_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    moments = {}  # frame -> tag
    for side in ('upper', 'lower'):
        rows = [r for r in recs if (r.get('players') or {}).get(side)
                and (r['players'][side].get('court'))]
        if not rows:
            continue
        rows.sort(key=lambda r: -(r['players'][side].get('speed') or 0))
        for r in rows[:2]:  # 冲刺帧
            moments.setdefault(r['frame'], f'{side}_sprint')
        # 挥拍伸展帧: 手部远离身体中心(图像空间)
        def reach(r):
            p = r['players'][side]
            h = p.get('hands') or {}
            cands = [h[k] for k in ('left', 'right') if h.get(k) and h[k][0] > 1]
            if not cands or not p.get('image'):
                return 0.0
            hc = np.array(cands, float).mean(0)
            return float(np.hypot(*(hc - np.array(p['image']))))
        rows.sort(key=lambda r: -reach(r))
        for r in rows[:1]:
            moments.setdefault(r['frame'], f'{side}_swing')
    # 球接触帧: 球与任一球员图像距离最小
    def ball_near(r):
        s = r.get('shuttlecock')
        if not (s and s.get('image')):
            return 1e9
        best = 1e9
        for side in ('upper', 'lower'):
            p = (r.get('players') or {}).get(side)
            if p and p.get('image'):
                d = np.hypot(s['image'][0] - p['image'][0], s['image'][1] - p['image'][1])
                best = min(best, d)
        return best
    cand = sorted(recs, key=ball_near)[:40]
    for r in cand:
        if ball_near(r) < 1e8:
            moments.setdefault(r['frame'], 'ball_contact')
            break
    # 间隔去重(>=1.5s), 最多 max_frames
    picked = []
    for fr in sorted(moments):
        if all(abs(fr - p) > 45 for p in picked):
            picked.append(fr)
        if len(picked) >= max_frames:
            break
    return [(fr, moments[fr]) for fr in picked]


def draw_keyframes(video_path, detections_path, out_dir, max_frames=8):
    """返回 [(帧图路径, 标签), ...]"""
    from detection.rtmpose import RTMPoseProcessor
    from rtmlib import draw_skeleton

    moments = pick_moments(detections_path, max_frames)
    if not moments:
        return []
    # 球员图像位置: 找骨架时匹配最近的人
    ref = {}
    with open(detections_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r['frame'] in {fr for fr, _ in moments}:
                for side in ('upper', 'lower'):
                    p = (r.get('players') or {}).get(side)
                    if p and p.get('image'):
                        ref.setdefault(r['frame'], []).append(p['image'])

    os.makedirs(out_dir, exist_ok=True)
    proc = RTMPoseProcessor(mode='balanced', device='cuda' if _cuda_ok() else 'cpu')
    cap = cv2.VideoCapture(video_path)
    out = []
    for fr, tag in moments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            kps, scores = proc.process_frame(frame)
        except Exception as e:
            print(f'帧 {fr} 姿态失败: {e}')
            kps = None
        if kps is not None and len(kps):
            want = ref.get(fr, [])
            # 选骨架中心最靠近记录位置的
            def dist(k):
                return min(np.hypot(*(_hip_center(k) - np.array(w))) for w in want) if want else 0
            k = kps[int(np.argmin([dist(k) for k in kps]))]
            s = np.ones(len(k)) if scores is None or len(scores) == 0 else scores[int(np.argmin([dist(k2) for k2 in kps]))]
            frame = draw_skeleton(frame, k[None, ...], np.asarray(s)[None, ...], kpt_thr=0.3)
        path = os.path.join(out_dir, f'frame_{fr:05d}_{tag}.jpg')
        cv2.imwrite(path, frame)
        out.append((path, tag))
        print(f'关键帧: {os.path.basename(path)}')
    cap.release()
    return out


def _cuda_ok():
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        return False


if __name__ == '__main__':
    import style_features
    name = sys.argv[1] if len(sys.argv) > 1 else 'low_angle_linDanCup'
    meta = json.load(open(f'results/{name}/metadata.json'))
    frames = draw_keyframes(meta['video']['path'], f'results/{name}/detections.jsonl',
                            f'results/{name}/coach_frames')
    print(f'共 {len(frames)} 张关键帧')
