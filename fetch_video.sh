#!/usr/bin/env bash
# 下载验证用视频到 videos/ (B站/YouTube 等站点, 依赖 yt-dlp)
# 用法:
#   ./fetch_video.sh <URL>              # 整个视频
#   ./fetch_video.sh <URL> <起秒> <止秒>  # 只下片段 (如 ./fetch_video.sh URL 180 480)
#
# 选片建议 (标定与分析的前提):
#   1. 机位固定不动 —— 广播镜头有推拉摇移, 标定只做一次, 一动就错
#   2. 低视角(网后 1~2 人高, 观众席/场边手机架)最容易复现 fork 的自动标定
#   3. 分辨率 >=720p, 球场边线大部分可见
#   4. B站多P视频在 URL 带 ?p=N 即下载对应分P
set -e
YTDLP=${YTDLP:-$(command -v yt-dlp || echo /home/zxy/.conda/envs/comfyui/bin/yt-dlp)}
mkdir -p videos
url=$1
extra=()
if [ $# -ge 3 ]; then
  t0=$2; t1=$3
  fmt() { printf '%d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }
  extra=(--download-sections "*$(fmt "$t0")-$(fmt "$t1")" --force-keyframes-at-cuts)
fi
"$YTDLP" -f 'bv*[height<=720]+ba/b[height<=720]/b' \
  --merge-output-format mp4 \
  -o 'videos/%(title).60s.%(ext)s' \
  --no-playlist "${extra[@]}" "$url"
echo "完成, 输出在 videos/"
