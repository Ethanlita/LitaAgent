"""
启动可视化服务器的快捷入口。

默认监听 0.0.0.0:8080，便于远程访问（前端由 Cloudflare/WAF 保护）。
可通过环境变量覆盖端口/主机：
  VIS_HOST=0.0.0.0 VIS_PORT=8080 python start_visualizer.py
"""

import os
from scml_analyzer.visualizer import start_server


if __name__ == "__main__":
    host = os.environ.get("VIS_HOST", "0.0.0.0")
    port = int(os.environ.get("VIS_PORT", "8080"))
    start_server(port=port, open_browser=False, host=host)
