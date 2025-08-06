#!/bin/bash
# 主数字人服务停止脚本
# 功能：只停止主服务（app.py），不影响管理服务器

# 配置参数
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"
MAIN_PORT=8010

echo ">>> 主数字人服务停止脚本"
echo ">>> 日志目录: $LOG_DIR"
echo ">>> 主服务端口: $MAIN_PORT"
echo ""

# 删除PID文件
if [ -f "$LOG_DIR/main.pid" ]; then
    rm -f $LOG_DIR/main.pid
    echo ">>> 已删除PID文件"
fi

# 查找并停止app.py进程
echo ">>> 查找并停止app.py进程..."
APP_PIDS=$(ps aux | grep "python.*app.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$APP_PIDS" ]; then
    echo ">>> 发现app.py进程: $APP_PIDS"
    for pid in $APP_PIDS; do
        echo ">>> 停止进程: $pid"
        kill -TERM $pid
    done
    
    sleep 2
    
    # 强制停止仍在运行的进程
    REMAINING_PIDS=$(ps aux | grep "python.*app.py" | grep -v grep | awk '{print $2}')
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo ">>> 强制停止剩余进程: $REMAINING_PIDS"
        for pid in $REMAINING_PIDS; do
            kill -KILL $pid
        done
    fi
    
    echo ">>> ✅ 主服务已停止"
else
    echo ">>> 未发现app.py进程"
fi

# 检查端口
if netstat -tlnp 2>/dev/null | grep ":$MAIN_PORT " > /dev/null; then
    echo ">>> ⚠️  端口 $MAIN_PORT 仍被占用:"
    netstat -tlnp 2>/dev/null | grep ":$MAIN_PORT "
else
    echo ">>> ✅ 端口 $MAIN_PORT 已释放"
fi

echo ""
echo ">>> 主数字人服务停止完成!"
echo ">>> 管理服务器继续运行" 