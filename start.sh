#!/bin/bash
# kill_and_start.sh
# 功能：先杀掉旧的进程，再启动新的 start.py

# Conda 初始化
source /mnt/disk1/ftp/data/60397193/miniconda3/etc/profile.d/conda.sh
conda activate nerfstream

# 要杀掉的脚本
TARGETS="app.py|start.py|run_dynamic.py"

echo ">>> Killing old processes..."
ps aux | grep -E "$TARGETS" | grep -v grep | awk '{print $2}' | xargs -r kill -9

sleep 2  # 等待 2 秒确保进程退出

# 创建日志目录
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"
mkdir -p $LOG_DIR

echo ">>> Starting start.py..."
python /mnt/disk1/ftp/file/60397193/Unimed/start.py > $LOG_DIR/start.log 2>&1 &

# 输出新进程 PID
NEW_PID=$!
echo ">>> Started start.py with PID $NEW_PID"
echo ">>> Logs: $LOG_DIR/start.log"
