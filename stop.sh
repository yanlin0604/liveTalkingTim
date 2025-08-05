#!/bin/bash
# LiveTalking 服务停止脚本
# 功能：停止所有 LiveTalking 相关进程

# 配置参数
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"

# 要停止的脚本
TARGETS="app.py|start.py|run_dynamic.py|management_server.py"

echo ">>> LiveTalking 服务停止脚本"
echo ""

# 方法1：通过PID文件停止（如果存在）
if [ -f "$LOG_DIR/main.pid" ]; then
    MAIN_PID=$(cat $LOG_DIR/main.pid)
    if ps -p $MAIN_PID > /dev/null 2>&1; then
        echo ">>> 停止主服务 (PID: $MAIN_PID)..."
        kill -TERM $MAIN_PID
        sleep 2
        if ps -p $MAIN_PID > /dev/null 2>&1; then
            echo ">>> 强制停止主服务..."
            kill -KILL $MAIN_PID
        fi
        echo ">>> 主服务已停止"
    else
        echo ">>> 主服务进程不存在"
    fi
    rm -f $LOG_DIR/main.pid
fi

if [ -f "$LOG_DIR/management.pid" ]; then
    MANAGEMENT_PID=$(cat $LOG_DIR/management.pid)
    if ps -p $MANAGEMENT_PID > /dev/null 2>&1; then
        echo ">>> 停止管理服务器 (PID: $MANAGEMENT_PID)..."
        kill -TERM $MANAGEMENT_PID
        sleep 2
        if ps -p $MANAGEMENT_PID > /dev/null 2>&1; then
            echo ">>> 强制停止管理服务器..."
            kill -KILL $MANAGEMENT_PID
        fi
        echo ">>> 管理服务器已停止"
    else
        echo ">>> 管理服务器进程不存在"
    fi
    rm -f $LOG_DIR/management.pid
fi

# 方法2：通过进程名停止（确保所有相关进程都被停止）
echo ">>> 检查并停止所有相关进程..."
ps aux | grep -E "$TARGETS" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    CMD=$(echo $line | awk '{print $11, $12, $13, $14, $15}')
    echo ">>> 停止进程: $CMD (PID: $PID)"
    kill -TERM $PID 2>/dev/null
done

sleep 2

# 强制停止剩余进程
echo ">>> 强制停止剩余进程..."
ps aux | grep -E "$TARGETS" | grep -v grep | while read line; do
    PID=$(echo $line | awk '{print $2}')
    CMD=$(echo $line | awk '{print $11, $12, $13, $14, $15}')
    echo ">>> 强制停止进程: $CMD (PID: $PID)"
    kill -KILL $PID 2>/dev/null
done

# 检查是否还有进程在运行
REMAINING=$(ps aux | grep -E "$TARGETS" | grep -v grep | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo ">>> ✅ 所有服务已成功停止"
else
    echo ">>> ⚠️  仍有 $REMAINING 个进程在运行:"
    ps aux | grep -E "$TARGETS" | grep -v grep
fi

echo ""
echo ">>> 服务状态检查:"
echo ">>> 主服务端口 8010: $(netstat -tlnp 2>/dev/null | grep :8010 || echo '未监听')"
echo ">>> 管理服务器端口 8011: $(netstat -tlnp 2>/dev/null | grep :8011 || echo '未监听')" 