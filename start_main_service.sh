#!/bin/bash
# 主数字人服务启动脚本
# 功能：只启动主服务（app.py），不影响管理服务器

# Conda 初始化
source /mnt/disk1/ftp/data/60397193/miniconda3/etc/profile.d/conda.sh
conda activate nerfstream

# 配置参数
SCRIPT_DIR="/mnt/disk1/ftp/file/60397193/LiveTalking"
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"
CONFIG_FILE="config.json"

# 默认端口
MAIN_PORT=8010

# 要杀掉的脚本（只针对主服务相关进程）
TARGETS="app.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --main-port)
            MAIN_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --main-port PORT     主服务端口（默认: 8010）"
            echo "  --config FILE        配置文件路径（默认: config.json）"
            echo "  --help, -h           显示帮助信息"
            echo ""
            echo "Examples:"
            echo "  $0                    # 使用默认配置启动主服务"
            echo "  $0 --main-port 8020   # 自定义端口启动主服务"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ">>> 主数字人服务启动脚本"
echo ">>> 配置文件: $CONFIG_FILE"
echo ">>> 主服务端口: $MAIN_PORT"
echo ""

# 杀掉旧进程
echo ">>> 停止旧的主服务进程..."
ps aux | grep -E "$TARGETS" | grep -v grep | awk '{print $2}' | xargs -r kill -9

sleep 2  # 等待进程退出

# 创建日志目录
mkdir -p $LOG_DIR

# 切换到脚本目录
cd $SCRIPT_DIR

# 检查端口是否被占用
echo ">>> 检查端口 $MAIN_PORT 是否被占用..."
if netstat -tlnp 2>/dev/null | grep ":$MAIN_PORT " > /dev/null; then
    echo ">>> 端口 $MAIN_PORT 仍被占用，查找占用进程..."
    PORT_PID=$(netstat -tlnp 2>/dev/null | grep ":$MAIN_PORT " | awk '{print $7}' | cut -d'/' -f1)
    if [ ! -z "$PORT_PID" ]; then
        echo ">>> 端口 $MAIN_PORT 被进程 $PORT_PID 占用，停止该进程..."
        kill -TERM $PORT_PID
        sleep 2
        if ps -p $PORT_PID > /dev/null 2>&1; then
            echo ">>> 强制停止进程 $PORT_PID..."
            kill -KILL $PORT_PID
        fi
        echo ">>> 端口 $MAIN_PORT 的进程已停止"
    fi
else
    echo ">>> 端口 $MAIN_PORT 未被占用"
fi

# 启动主服务
echo ">>> 启动主数字人服务 (端口: $MAIN_PORT)..."
python app.py --config_file $CONFIG_FILE --listenport $MAIN_PORT > $LOG_DIR/main.log 2>&1 &
MAIN_PID=$!
echo ">>> 主服务已启动，PID: $MAIN_PID"

# 保存PID到文件
echo $MAIN_PID > $LOG_DIR/main.pid
echo ">>> 已保存PID到文件: $LOG_DIR/main.pid"

# 等待服务启动
echo ">>> 等待服务启动..."
sleep 3

# 检查服务状态
if ps -p $MAIN_PID > /dev/null 2>&1; then
    echo ">>> ✅ 主数字人服务启动成功!"
    echo ">>> 主服务PID: $MAIN_PID"
    echo ">>> 主服务端口: $MAIN_PORT"
    echo ">>> 主服务日志: $LOG_DIR/main.log"
    echo ">>> 主服务地址: http://localhost:$MAIN_PORT"
    echo ">>> 数字人界面: http://localhost:$MAIN_PORT/dashboard.html"
    echo ">>> 配置管理: http://localhost:$MAIN_PORT/config_manager.html"
else
    echo ">>> ❌ 主数字人服务启动失败!"
    echo ">>> 请检查日志文件: $LOG_DIR/main.log"
    exit 1
fi

echo ""
echo ">>> 使用以下命令停止主服务:"
echo ">>> ./stop_main_service.sh" 