"""读图/文本 LLM 工具（配置见 config/llm_api.json）
用法:
  python ask_vision.py <图片路径> [问题] [模型]
  python -c "from ask_vision import ask; print(ask('img.jpg','问题'))"
"""
import base64
import json
import os
import sys
import urllib.request

_CONFIG = None


def config():
    global _CONFIG
    if _CONFIG is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'llm_api.json')
        with open(path) as f:
            _CONFIG = json.load(f)
    return _CONFIG


def ask(prompt, img_path=None, model=None, max_tokens=None):
    cfg = config()
    model = model or (cfg['vision_model'] if img_path else cfg['text_model'])
    max_tokens = max_tokens or cfg['max_tokens']
    content = [{'type': 'text', 'text': prompt}]
    if img_path:
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': max_tokens,
    }).encode()
    req = urllib.request.Request(
        cfg['base_url'].rstrip('/') + '/chat/completions', data=body,
        headers={'Authorization': f"Bearer {cfg['api_key']}",
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=cfg['timeout']) as r:
        out = json.loads(r.read())
        msg = out['choices'][0]['message']
        return msg.get('content') or msg.get('reasoning_content', '')


if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        img = sys.argv[1]
        q = sys.argv[2] if len(sys.argv) > 2 else '描述这张图'
        m = sys.argv[3] if len(sys.argv) > 3 else None
        print(ask(q, img_path=img, model=m))
    else:
        q = sys.argv[1] if sys.argv else '你好'
        print(ask(q))
