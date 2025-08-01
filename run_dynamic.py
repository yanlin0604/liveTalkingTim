#!/usr/bin/env python3
"""
Unimed 启动器
支持运行时热更新配置，无需重启服务

使用方法:
    python run_dynamic.py                    # 使用默认config.json
    python run_dynamic.py --config my.json  # 使用指定配置文件
"""

import json
import argparse
import sys
import os
import subprocess
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

访问地址:
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
    
    # 检查app.py
    if not os.path.exists('app.py'):
        print("❌ app.py 不存在，请确保在正确的目录中运行")
        sys.exit(1)
    
    # 打印启动信息
    print("🚀 Unimed 启动")
    print("=" * 50)
    print(f"📁 配置文件: {args.config}")
    print(f"🎭 数字人界面: http://localhost:8010/dashboard.html")
    print(f"⚙️  配置管理: http://localhost:8010/config_manager.html")
    print("=" * 50)
    print("按 Ctrl+C 停止服务")
    print()
    
    # 构建启动命令
    cmd = [
        sys.executable, 'app.py',
        '--config_file', args.config
    ]
    
    # 启动服务
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
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