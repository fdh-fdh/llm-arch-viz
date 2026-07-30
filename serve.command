#!/bin/bash
# macOS 双击启动:本地起服务并打开浏览器
cd "$(dirname "$0")"
PORT=8000
( sleep 1; open "http://localhost:$PORT" ) &
python3 -m http.server $PORT
