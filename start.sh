#!/bin/bash
# LiveTalking 服务启动脚本
# 功能：支持分离式服务模式（主服务 + 管理服务器）和单服务模式

# Conda 初始化
source /mnt/disk1/ftp/data/60397193/miniconda3/etc/profile.d/conda.sh
conda activate nerfstream

# 配置参数
SCRIPT_DIR="/mnt/disk1/ftp/file/60397193/Unimed"
LOG_DIR="/mnt/disk1/ftp/data/60397193/logs"
CONFIG_FILE="config.json"

# 默认端口
MAIN_PORT=8010
MANAGEMENT_PORT=8011

# 要杀掉的脚本（包括管理服务器）
TARGETS="app.py|start.py|run_dynamic.py|management_server.py"

# 解析命令行参数
MODE="separated"  # 默认分离式模式
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            MODE="single"
            shift
            ;;
        --main-port)
            MAIN_PORT="$2"
            shift 2
            ;;
        --management-port)
            MANAGEMENT_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --single             启动单服务模式（兼容旧版本）"
            echo "  --main-port PORT     主服务端口（默认: 8010）"
            echo "  --management-port PORT 管理服务器端口（默认: 8011）"
            echo "  --config FILE        配置文件路径（默认: config.json）"
            echo "  --help, -h           显示帮助信息"
            echo ""
            echo "Examples:"
            echo "  $0                    # 启动分离式服务模式"
            echo "  $0 --single           # 启动单服务模式"
            echo "  $0 --main-port 8020 --management-port 8021  # 自定义端口"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ">>> LiveTalking 服务启动脚本"
echo ">>> 模式: $MODE"
echo ">>> 配置文件: $CONFIG_FILE"
echo ">>> 主服务端口: $MAIN_PORT"
if [ "$MODE" = "separated" ]; then
    echo ">>> 管理服务器端口: $MANAGEMENT_PORT"
fi
echo ""

# 杀掉旧进程
echo ">>> 停止旧进程..."
ps aux | grep -E "$TARGETS" | grep -v grep | awk '{print $2}' | xargs -r kill -9

sleep 2  # 等待进程退出

# 创建日志目录
mkdir -p $LOG_DIR

# 切换到脚本目录
cd $SCRIPT_DIR

if [ "$MODE" = "separated" ]; then
    # 分离式服务模式
    echo ">>> 启动分离式服务模式..."
    
    # 检查管理服务器文件是否存在
    if [ ! -f "management_server.py" ]; then
        echo ">>> 错误: management_server.py 不存在，请使用 --single 模式"
        exit 1
    fi
    
    # 启动管理服务器
    echo ">>> 启动管理服务器 (端口: $MANAGEMENT_PORT)..."
    python management_server.py --port $MANAGEMENT_PORT --config_file $CONFIG_FILE > $LOG_DIR/management.log 2>&1 &
    MANAGEMENT_PID=$!
    echo ">>> 管理服务器已启动，PID: $MANAGEMENT_PID"
    
    # 等待管理服务器启动
    sleep 3
    
    # 启动主服务
    echo ">>> 启动主数字人服务 (端口: $MAIN_PORT)..."
    python app.py --config_file $CONFIG_FILE --listenport $MAIN_PORT > $LOG_DIR/main.log 2>&1 &
    MAIN_PID=$!
    echo ">>> 主服务已启动，PID: $MAIN_PID"
    
    echo ""
    echo ">>> 服务启动完成!"
    echo ">>> 主数字人服务: http://localhost:$MAIN_PORT"
    echo ">>> 管理服务器: http://localhost:$MANAGEMENT_PORT"
    echo ">>> API文档: http://localhost:$MANAGEMENT_PORT/swagger"
    echo ">>> 主服务日志: $LOG_DIR/main.log"
    echo ">>> 管理服务器日志: $LOG_DIR/management.log"
    
    # 保存PID到文件
    echo $MAIN_PID > $LOG_DIR/main.pid
    echo $MANAGEMENT_PID > $LOG_DIR/management.pid
    
else
    # 单服务模式
    echo ">>> 启动单服务模式..."
    
    # 启动主服务
    echo ">>> 启动主服务 (端口: $MAIN_PORT)..."
    python app.py --config_file $CONFIG_FILE --listenport $MAIN_PORT > $LOG_DIR/main.log 2>&1 &
    MAIN_PID=$!
    echo ">>> 主服务已启动，PID: $MAIN_PID"
    
    echo ""
    echo ">>> 服务启动完成!"
    echo ">>> 数字人界面: http://localhost:$MAIN_PORT/dashboard.html"
    echo ">>> 配置管理: http://localhost:$MAIN_PORT/config_manager.html"
    echo ">>> 服务日志: $LOG_DIR/main.log"
    
    # 保存PID到文件
    echo $MAIN_PID > $LOG_DIR/main.pid
fi

echo ""
echo ">>> 使用以下命令停止服务:"
echo ">>> ./stop.sh"
