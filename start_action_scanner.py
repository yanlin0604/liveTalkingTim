#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作视频扫描器启动脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from action_scanner import ActionScanner
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='启动动作视频扫描器')
    parser.add_argument('--config', default='action_scanner_config.json', type=str,
                       help='配置文件路径')
    parser.add_argument('--once', action='store_true',
                       help='只执行一次扫描，不持续运行')
    parser.add_argument('--scan_dir', type=str,
                       help='覆盖配置文件中的扫描目录')
    parser.add_argument('--interval', type=int,
                       help='覆盖配置文件中的扫描间隔')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("动作视频扫描器启动中...")
    print("=" * 50)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件 {args.config} 不存在，将使用默认配置")
        config_file = None
    else:
        config_file = args.config
        print(f"使用配置文件: {args.config}")
    
    # 创建扫描器
    scanner = ActionScanner(
        scan_directory=args.scan_dir,
        scan_interval=args.interval,
        config_file=config_file
    )
    
    try:
        if args.once:
            print("执行单次扫描...")
            scanner.scan_once()
            print("扫描完成！")
        else:
            print("开始持续扫描...")
            print("按 Ctrl+C 停止扫描")
            scanner.start_scanning()
            
            # 等待用户中断
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n收到停止信号，正在停止扫描器...")
                scanner.stop_scanning()
                print("扫描器已停止")
                
    except Exception as e:
        print(f"程序运行出错: {e}")
        scanner.stop_scanning()

if __name__ == "__main__":
    main() 