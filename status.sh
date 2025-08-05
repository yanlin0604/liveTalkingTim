#!/bin/bash
# LiveTalking 服务状态检查脚本
# 功能：检查服务运行状态和端口监听情况

# 配置参数
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"

# 要检查的脚本
TARGETS="app.py|start.py|run_dynamic.py|management_server.py"

echo ">>> LiveTalking 服务状态检查"
echo "=================================="

# 检查进程状态
echo ">>> 进程状态:"
PROCESSES=$(ps aux | grep -E "$TARGETS" | grep -v grep)
if [ -z "$PROCESSES" ]; then
    echo ">>> ❌ 没有找到运行中的 LiveTalking 进程"
else
    echo ">>> ✅ 找到以下运行中的进程:"
    echo "$PROCESSES" | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CMD=$(echo $line | awk '{print $11, $12, $13, $14, $15}')
        echo ">>>   PID: $PID | 命令: $CMD"
    done
fi

echo ""

# 检查端口监听状态
echo ">>> 端口监听状态:"
echo ">>> 主服务端口 8010:"
if netstat -tlnp 2>/dev/null | grep -q :8010; then
    echo ">>>   ✅ 正在监听"
    netstat -tlnp 2>/dev/null | grep :8010
else
    echo ">>>   ❌ 未监听"
fi

echo ">>> 管理服务器端口 8011:"
if netstat -tlnp 2>/dev/null | grep -q :8011; then
    echo ">>>   ✅ 正在监听"
    netstat -tlnp 2>/dev/null | grep :8011
else
    echo ">>>   ❌ 未监听"
fi

echo ""

# 检查PID文件
echo ">>> PID文件状态:"
if [ -f "$LOG_DIR/main.pid" ]; then
    MAIN_PID=$(cat $LOG_DIR/main.pid)
    if ps -p $MAIN_PID > /dev/null 2>&1; then
        echo ">>>   ✅ 主服务PID文件存在且进程运行中 (PID: $MAIN_PID)"
    else
        echo ">>>   ⚠️  主服务PID文件存在但进程不存在 (PID: $MAIN_PID)"
    fi
else
    echo ">>>   ❌ 主服务PID文件不存在"
fi

if [ -f "$LOG_DIR/management.pid" ]; then
    MANAGEMENT_PID=$(cat $LOG_DIR/management.pid)
    if ps -p $MANAGEMENT_PID > /dev/null 2>&1; then
        echo ">>>   ✅ 管理服务器PID文件存在且进程运行中 (PID: $MANAGEMENT_PID)"
    else
        echo ">>>   ⚠️  管理服务器PID文件存在但进程不存在 (PID: $MANAGEMENT_PID)"
    fi
else
    echo ">>>   ❌ 管理服务器PID文件不存在"
fi

echo ""

# 检查日志文件
echo ">>> 日志文件状态:"
if [ -f "$LOG_DIR/main.log" ]; then
    MAIN_LOG_SIZE=$(du -h $LOG_DIR/main.log | cut -f1)
    MAIN_LOG_LINES=$(wc -l < $LOG_DIR/main.log)
    echo ">>>   ✅ 主服务日志: $LOG_DIR/main.log ($MAIN_LOG_SIZE, $MAIN_LOG_LINES 行)"
else
    echo ">>>   ❌ 主服务日志不存在: $LOG_DIR/main.log"
fi

if [ -f "$LOG_DIR/management.log" ]; then
    MANAGEMENT_LOG_SIZE=$(du -h $LOG_DIR/management.log | cut -f1)
    MANAGEMENT_LOG_LINES=$(wc -l < $LOG_DIR/management.log)
    echo ">>>   ✅ 管理服务器日志: $LOG_DIR/management.log ($MANAGEMENT_LOG_SIZE, $MANAGEMENT_LOG_LINES 行)"
else
    echo ">>>   ❌ 管理服务器日志不存在: $LOG_DIR/management.log"
fi

echo ""

# 检查服务可用性
echo ">>> 服务可用性检查:"
echo ">>> 主服务 (http://localhost:8010):"
if curl -s --connect-timeout 3 http://localhost:8010/webrtc/status > /dev/null 2>&1; then
    echo ">>>   ✅ 可访问"
else
    echo ">>>   ❌ 不可访问"
fi

echo ">>> 管理服务器 (http://localhost:8011):"
if curl -s --connect-timeout 3 http://localhost:8011/get_status > /dev/null 2>&1; then
    echo ">>>   ✅ 可访问"
else
    echo ">>>   ❌ 不可访问"
fi

echo ""

# 总结
echo ">>> 服务状态总结:"
MAIN_PROCESSES=$(ps aux | grep -E "app.py|start.py|run_dynamic.py" | grep -v grep | wc -l)
MANAGEMENT_PROCESSES=$(ps aux | grep -E "management_server.py" | grep -v grep | wc -l)

if [ $MAIN_PROCESSES -gt 0 ] && [ $MANAGEMENT_PROCESSES -gt 0 ]; then
    echo ">>> 🟢 分离式服务模式运行中"
elif [ $MAIN_PROCESSES -gt 0 ] && [ $MANAGEMENT_PROCESSES -eq 0 ]; then
    echo ">>> 🟡 单服务模式运行中"
elif [ $MAIN_PROCESSES -eq 0 ] && [ $MANAGEMENT_PROCESSES -gt 0 ]; then
    echo ">>> 🟡 仅管理服务器运行中"
else
    echo ">>> 🔴 没有服务运行"
fi

echo ""
echo ">>> 使用以下命令管理服务:"
echo ">>>   ./start.sh              # 启动分离式服务"
echo ">>>   ./start.sh --single     # 启动单服务模式"
echo ">>>   ./stop.sh               # 停止所有服务"
echo ">>>   ./status.sh             # 查看服务状态" 