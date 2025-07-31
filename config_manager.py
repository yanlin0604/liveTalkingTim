#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具 - 管理config.json中的设置
"""

import json
import argparse
from pathlib import Path

def load_config():
    """加载config.json"""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config_data):
    """保存config.json"""
    config_path = Path("config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

def show_config():
    """显示当前配置"""
    config_data = load_config()
    
    if not config_data:
        print("配置文件不存在或为空")
        return
    
    print("\n当前配置:")
    print("-" * 50)
    
    # 显示关键配置项
    key_configs = [
        ("use_custom_silent", "静音时自动使用自定义动作"),
        ("fps", "音频帧率"),
        ("W", "界面宽度"),
        ("H", "界面高度"),
        ("model", "模型类型"),
        ("avatar_id", "数字人ID"),
        ("tts", "TTS服务"),
        ("llm_provider", "LLM提供商"),
        ("transport", "传输方式")
    ]
    
    for key, description in key_configs:
        if key in config_data:
            value = config_data[key]
            status = "✓" if value else "✗" if isinstance(value, bool) else str(value)
            print(f"{description}: {status}")
    
    print("-" * 50)

def set_config(key, value):
    """设置配置项"""
    config_data = load_config()
    
    # 类型转换
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    elif value.isdigit():
        value = int(value)
    elif value.replace('.', '').isdigit():
        value = float(value)
    
    config_data[key] = value
    save_config(config_data)
    
    print(f"已设置 {key} = {value}")

def main():
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('action', choices=['show', 'set'], help='操作类型')
    parser.add_argument('--key', type=str, help='配置项名称')
    parser.add_argument('--value', type=str, help='配置项值')
    
    args = parser.parse_args()
    
    if args.action == 'show':
        show_config()
    
    elif args.action == 'set':
        if not args.key or args.value is None:
            print("错误: 设置配置需要指定 --key 和 --value")
            return
        
        set_config(args.key, args.value)

if __name__ == "__main__":
    main() 