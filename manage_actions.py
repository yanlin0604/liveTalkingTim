#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作管理脚本 - 管理custom_config.json中的动作配置
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime

def load_custom_config():
    """加载custom_config.json"""
    config_path = Path("data/custom_config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_custom_config(config_data):
    """保存custom_config.json"""
    config_path = Path("data/custom_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

def list_actions():
    """列出所有动作"""
    config_data = load_custom_config()
    
    if not config_data:
        print("暂无动作配置")
        return
    
    print(f"\n当前共有 {len(config_data)} 个动作配置:")
    print("-" * 80)
    
    for i, item in enumerate(config_data, 1):
        print(f"{i}. 动作名称: {item.get('audiotype', 'N/A')}")
        print(f"   描述: {item.get('description', 'N/A')}")
        print(f"   图片路径: {item.get('imgpath', 'N/A')}")
        print(f"   音频路径: {item.get('audiopath', 'N/A')}")
        print(f"   创建时间: {item.get('created_time', 'N/A')}")
        
        # 检查文件是否存在
        imgpath = Path(item.get('imgpath', ''))
        audiopath = Path(item.get('audiopath', ''))
        
        img_exists = imgpath.exists() and any(imgpath.glob("*.png")) or any(imgpath.glob("*.jpg"))
        audio_exists = audiopath.exists()
        
        print(f"   状态: 图片{'✓' if img_exists else '✗'} 音频{'✓' if audio_exists else '✗'}")
        print()

def add_action(audiotype, imgpath, audiopath, description=None):
    """添加新动作"""
    config_data = load_custom_config()
    
    # 检查是否已存在
    for item in config_data:
        if item.get('audiotype') == audiotype:
            print(f"错误: 动作 '{audiotype}' 已存在")
            return False
    
    # 检查路径是否存在
    if not Path(imgpath).exists():
        print(f"错误: 图片路径 '{imgpath}' 不存在")
        return False
    
    if not Path(audiopath).exists():
        print(f"错误: 音频路径 '{audiopath}' 不存在")
        return False
    
    # 添加新配置
    new_config = {
        "audiotype": audiotype,
        "imgpath": imgpath,
        "audiopath": audiopath,
        "description": description or f"手动添加的动作配置 - {audiotype}",
        "created_time": datetime.now().isoformat()
    }
    
    config_data.append(new_config)
    save_custom_config(config_data)
    
    print(f"成功添加动作: {audiotype}")
    return True

def remove_action(audiotype):
    """删除动作"""
    config_data = load_custom_config()
    
    # 查找并删除
    original_length = len(config_data)
    config_data = [item for item in config_data if item.get('audiotype') != audiotype]
    
    if len(config_data) < original_length:
        save_custom_config(config_data)
        print(f"成功删除动作: {audiotype}")
        return True
    else:
        print(f"错误: 动作 '{audiotype}' 不存在")
        return False

def update_action(audiotype, imgpath=None, audiopath=None, description=None):
    """更新动作配置"""
    config_data = load_custom_config()
    
    # 查找动作
    for item in config_data:
        if item.get('audiotype') == audiotype:
            if imgpath:
                item['imgpath'] = imgpath
            if audiopath:
                item['audiopath'] = audiopath
            if description:
                item['description'] = description
            
            item['updated_time'] = datetime.now().isoformat()
            
            save_custom_config(config_data)
            print(f"成功更新动作: {audiotype}")
            return True
    
    print(f"错误: 动作 '{audiotype}' 不存在")
    return False

def main():
    parser = argparse.ArgumentParser(description='动作管理脚本')
    parser.add_argument('action', choices=['list', 'add', 'remove', 'update'], 
                       help='操作类型')
    parser.add_argument('--audiotype', type=str, help='动作类型名称')
    parser.add_argument('--imgpath', type=str, help='图片路径')
    parser.add_argument('--audiopath', type=str, help='音频路径')
    parser.add_argument('--description', type=str, help='动作描述')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_actions()
    
    elif args.action == 'add':
        if not all([args.audiotype, args.imgpath, args.audiopath]):
            print("错误: 添加动作需要指定 --audiotype, --imgpath, --audiopath")
            return
        add_action(args.audiotype, args.imgpath, args.audiopath, args.description)
    
    elif args.action == 'remove':
        if not args.audiotype:
            print("错误: 删除动作需要指定 --audiotype")
            return
        remove_action(args.audiotype)
    
    elif args.action == 'update':
        if not args.audiotype:
            print("错误: 更新动作需要指定 --audiotype")
            return
        if not any([args.imgpath, args.audiopath, args.description]):
            print("错误: 更新动作需要指定至少一个参数 (--imgpath, --audiopath, --description)")
            return
        update_action(args.audiotype, args.imgpath, args.audiopath, args.description)

if __name__ == "__main__":
    main() 