#!/usr/bin/env python3
"""
LiveTalking 快速启动脚本
"""

import os
import sys

def main():
    print("🚀 LiveTalking 快速启动")
    print("=" * 40)
    
    # 检查必要文件
    if not os.path.exists('app.py'):
        print("❌ app.py 不存在")
        return
    
    if not os.path.exists('config.json'):
        print("📝 创建默认配置文件...")
        from run_dynamic import create_default_config
        create_default_config('config.json')
    
    print("🎭 数字人界面: http://localhost:8010/dashboard.html")
    print("⚙️  配置管理: http://localhost:8010/config_manager.html")
    print("=" * 40)
    print("按 Ctrl+C 停止服务")
    print()
    
    # 启动服务
    import subprocess
    try:
        subprocess.run([sys.executable, 'run_dynamic.py'])
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")

if __name__ == '__main__':
    main()