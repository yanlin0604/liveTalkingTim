#!/usr/bin/env python3
"""
Unimed 启动器
支持运行时热更新配置，无需重启服务
支持分离式架构：主数字人服务 + 管理服务器

使用方法:
    python run_dynamic.py                    # 使用默认config.json启动分离式服务
    python run_dynamic.py --config my.json  # 使用指定配置文件
    python run_dynamic.py --single          # 启动单服务模式（兼容旧版本）
"""

import json
import argparse
import sys
import os
import subprocess
import time
import signal
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Unimed 启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
特性:
    ✅ 运行时修改配置无需重启
    ✅ Web界面管理配置参数  
    ✅ 自动监控配置文件变化
    ✅ 分离式架构（主服务 + 管理服务器）
    ✅ 兼容单服务模式

访问地址:
    分离式模式:
        主数字人服务: http://localhost:8010
        管理服务器: http://localhost:8011
        API文档: http://localhost:8011/swagger
    
    单服务模式:
        数字人界面: http://localhost:8010/dashboard.html
        配置管理: http://localhost:8010/config_manager.html
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='配置文件路径 (默认: config.json)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='创建默认配置文件并退出'
    )
    
    parser.add_argument(
        '--single',
        action='store_true',
        help='启动单服务模式（兼容旧版本）'
    )
    
    parser.add_argument(
        '--main-port',
        type=int,
        default=8010,
        help='主数字人服务端口 (默认: 8010)'
    )
    
    parser.add_argument(
        '--management-port',
        type=int,
        default=8011,
        help='管理服务器端口 (默认: 8011)'
    )
    
    args = parser.parse_args()
    
    # 创建默认配置文件
    if args.create_config:
        create_default_config(args.config)
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print(f"💡 使用 --create-config 创建默认配置文件")
        sys.exit(1)
    
    # 检查必要文件
    if not os.path.exists('app.py'):
        print("❌ app.py 不存在，请确保在正确的目录中运行")
        sys.exit(1)
    
    if not args.single and not os.path.exists('management_server.py'):
        print("❌ management_server.py 不存在，请确保在正确的目录中运行")
        print("💡 使用 --single 启动单服务模式")
        sys.exit(1)
    
    # 根据模式启动服务
    if args.single:
        start_single_service(args)
    else:
        start_separated_services(args)

def start_single_service(args):
    """启动单服务模式（兼容旧版本）"""
    print("🚀 Unimed 单服务模式启动")
    print("=" * 50)
    print(f"📁 配置文件: {args.config}")
    print(f"🎭 数字人界面: http://localhost:{args.main_port}/dashboard.html")
    print(f"⚙️  配置管理: http://localhost:{args.main_port}/config_manager.html")
    print("=" * 50)
    print("按 Ctrl+C 停止服务")
    print()
    
    # 构建启动命令
    cmd = [
        sys.executable, 'app.py',
        '--config_file', args.config,
        '--listenport', str(args.main_port)
    ]
    
    # 启动服务
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def start_separated_services(args):
    """启动分离式服务（主服务 + 管理服务器）"""
    print("🚀 Unimed 分离式服务启动")
    print("=" * 50)
    print(f"📁 配置文件: {args.config}")
    print(f"🎭 主数字人服务: http://localhost:{args.main_port}")
    print(f"⚙️  管理服务器: http://localhost:{args.management_port}")
    print(f"📚 API文档: http://localhost:{args.management_port}/swagger")
    print("=" * 50)
    print("按 Ctrl+C 停止所有服务")
    print()
    
    processes = []
    
    try:
        # 启动管理服务器
        management_cmd = [
            sys.executable, 'management_server.py',
            '--port', str(args.management_port),
            '--config_file', args.config
        ]
        print(f"🔧 启动管理服务器: {' '.join(management_cmd)}")
        management_proc = subprocess.Popen(management_cmd)
        processes.append(management_proc)
        
        # 等待管理服务器启动
        print("⏳ 等待管理服务器启动...")
        time.sleep(3)
        
        # 启动主服务
        main_cmd = [
            sys.executable, 'app.py',
            '--config_file', args.config,
            '--listenport', str(args.main_port)
        ]
        print(f"🎭 启动主数字人服务: {' '.join(main_cmd)}")
        main_proc = subprocess.Popen(main_cmd)
        processes.append(main_proc)
        
        print(f"\n✅ 服务启动完成!")
        print(f"🎭 主数字人服务: http://localhost:{args.main_port}")
        print(f"⚙️  管理服务器: http://localhost:{args.management_port}")
        print(f"📚 API文档: http://localhost:{args.management_port}/swagger")
        print("\n按 Ctrl+C 停止所有服务")
        
        # 等待进程结束
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 正在停止所有服务...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("✅ 所有服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        # 清理进程
        for proc in processes:
            try:
                proc.terminate()
            except:
                pass
        sys.exit(1)

def create_default_config(config_path):
    """创建默认配置文件"""
    default_config = {
        "// 配置文件": "Unimed Dynamic Configuration",
        "// 创建时间": "Auto-generated default config",
        
        "// 基础配置": "Basic Configuration",
        "model": "musetalk",
        "avatar_id": "avator_1", 
        "transport": "webrtc",
        "batch_size": 16,
        "auto_batch_size": true,
        "listenport": 8010,
        "max_session": 1,
        
        "// 界面配置": "UI Configuration",
        "W": 450,
        "H": 450,
        "fps": 50,
        
        "// 滑动窗口": "Sliding Window",
        "l": 10,
        "m": 8,
        "r": 10,
        
        "// TTS配置": "TTS Configuration", 
        "tts": "edgetts",
        "REF_FILE": "zh-CN-XiaoxiaoNeural",
        "REF_TEXT": null,
        "TTS_SERVER": "http://127.0.0.1:9880",
        
        "// LLM配置": "LLM Configuration",
        "llm_provider": "dashscope",
        "llm_model": "qwen-plus",
        "llm_system_prompt": "你是一个友善的AI助手，请用简洁自然的语言回答问题。",
        "ollama_host": "http://localhost:11434",
        
        "// 颜色匹配": "Color Matching",
        "enable_color_matching": true,
        "color_matching_strength": 0.6,
        
        "// 其他配置": "Other Configuration",
        "model_path": "",
        "wav2lip_model_size": "384",
        "customvideo_config": "",
        "push_url": "http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream"
    }
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 默认配置文件已创建: {config_path}")
        print(f"💡 现在可以运行: python run_dynamic.py --config {config_path}")
        
    except Exception as e:
        print(f"❌ 创建配置文件失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()