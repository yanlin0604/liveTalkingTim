#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作视频扫描器 - 定期扫描视频文件并自动生成动作编排切图
"""

import os
import time
import json
import argparse
import threading
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('action_scanner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ActionScanner:
    def __init__(self, 
                 scan_directory: str = "action_videos",
                 action_base_dir: str = "data/customvideo",
                 scan_interval: int = 60,
                 video_extensions: List[str] = None,
                 video_prefix: str = None,
                 genaction_script: str = "wav2lip/genaction.py",
                 genaction_options: Dict = None,
                 config_file: str = None):
        """
        初始化动作视频扫描器
        
        Args:
            scan_directory: 扫描目录
            action_base_dir: 动作数据存储目录
            scan_interval: 扫描间隔（秒）
            video_extensions: 视频文件扩展名列表
            video_prefix: 视频文件名称前缀，只处理以此前缀开头的文件
            genaction_script: genaction.py脚本路径
            genaction_options: genaction.py的额外选项
            config_file: 配置文件路径
        """
        # 加载配置文件
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        else:
            self.config = {}
        
        # 使用配置文件中的值，如果没有则使用参数值
        self.scan_directory = Path(scan_directory or self.config.get('scan_directory', 'action_videos'))
        self.action_base_dir = Path(action_base_dir or self.config.get('action_base_dir', 'data/customvideo'))
        self.scan_interval = scan_interval or self.config.get('scan_interval', 60)
        self.video_extensions = video_extensions or self.config.get('video_extensions', ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'])
        self.video_prefix = video_prefix or self.config.get('video_prefix', None)
        self.genaction_script = Path(genaction_script or self.config.get('genaction_script', 'wav2lip/genaction.py'))
        self.genaction_options = genaction_options or self.config.get('genaction_options', {})
        
        self.processed_videos = set()
        self.running = False
        self.scan_thread = None
        
        # 确保目录存在
        self.scan_directory.mkdir(parents=True, exist_ok=True)
        self.action_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已处理视频记录
        self.processed_file = self.action_base_dir / "processed_action_videos.json"
        self.load_processed_videos()
        
        logger.info(f"动作视频扫描器初始化完成")
        logger.info(f"扫描目录: {self.scan_directory}")
        logger.info(f"动作目录: {self.action_base_dir}")
        logger.info(f"扫描间隔: {self.scan_interval}秒")
        logger.info(f"视频扩展名: {self.video_extensions}")
        logger.info(f"视频前缀过滤: {self.video_prefix or '无'}")
        logger.info(f"genaction选项: {self.genaction_options}")
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"已加载配置文件: {config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = {}
    
    def calculate_file_md5(self, file_path: Path) -> str:
        """计算文件MD5值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_video_info(self, video_path: Path) -> Dict:
        """获取视频信息"""
        return {
            'path': str(video_path),
            'name': video_path.name,
            'size': video_path.stat().st_size,
            'md5': self.calculate_file_md5(video_path),
            'processed_time': datetime.now().isoformat()
        }
    
    def load_processed_videos(self):
        """加载已处理视频记录"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_videos = data.get('processed_videos', {})
                logger.info(f"加载了 {len(self.processed_videos)} 个已处理动作视频记录")
            except Exception as e:
                logger.error(f"加载已处理动作视频记录失败: {e}")
                self.processed_videos = {}
        else:
            self.processed_videos = {}
    
    def save_processed_videos(self):
        """保存已处理视频记录"""
        try:
            data = {
                'processed_videos': self.processed_videos,
                'last_updated': datetime.now().isoformat(),
                'format_version': '1.0'
            }
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"已保存 {len(self.processed_videos)} 个动作视频处理记录")
        except Exception as e:
            logger.error(f"保存已处理动作视频记录失败: {e}")
    
    def is_video_file(self, file_path: Path) -> bool:
        """判断是否为视频文件"""
        return file_path.suffix.lower() in self.video_extensions
    
    def is_processed(self, video_path: Path) -> bool:
        """判断视频是否已经处理过"""
        video_path_str = str(video_path)
        
        # 检查是否在已处理列表中
        if video_path_str in self.processed_videos:
            recorded_info = self.processed_videos[video_path_str]
            current_md5 = self.calculate_file_md5(video_path)
            
            if current_md5 == recorded_info.get('md5', ''):
                logger.info(f"动作视频 {video_path.name} 已处理过（MD5匹配）")
                return True
            else:
                logger.warning(f"动作视频 {video_path.name} MD5不匹配，可能文件已更改，需要重新处理")
                # 删除旧的记录
                del self.processed_videos[video_path_str]
                self.save_processed_videos()
                return False
        
        # 检查是否存在对应的动作目录
        video_name = video_path.stem
        action_dir = self.action_base_dir / video_name
        
        if action_dir.exists():
            # 检查是否有处理后的图片文件
            image_files = list(action_dir.glob("*.png")) + list(action_dir.glob("*.jpg"))
            if len(image_files) > 0:
                logger.info(f"动作视频 {video_name} 已处理过，找到 {len(image_files)} 张图片")
                # 更新记录，添加MD5信息
                video_info = self.get_video_info(video_path)
                self.processed_videos[video_path_str] = video_info
                self.save_processed_videos()
                return True
        
        return False
    
    def generate_action_id(self, video_path: Path) -> str:
        """生成动作ID"""
        video_name = video_path.stem
        # 清理文件名，移除特殊字符
        clean_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_'))
        return clean_name
    
    def process_video(self, video_path: Path) -> bool:
        """处理动作视频"""
        try:
            video_name = video_path.stem
            action_id = self.generate_action_id(video_path)
            
            logger.info(f"开始处理动作视频: {video_name}")
            logger.info(f"动作ID: {action_id}")
            
            # 构建genaction.py命令
            cmd = [
                sys.executable,
                str(self.genaction_script),
                "--video_path", str(video_path),
                "--avatar_id", action_id
            ]
            
            # 添加配置文件中的选项
            options = self.genaction_options
            if options:
                for key, value in options.items():
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                logger.info(f"动作视频 {video_name} 处理成功")
                
                # 更新处理记录
                video_info = self.get_video_info(video_path)
                self.processed_videos[str(video_path)] = video_info
                self.save_processed_videos()
                
                # 自动更新custom_config.json
                self.update_custom_config(action_id, video_name)
                
                return True
            else:
                logger.error(f"动作视频 {video_name} 处理失败")
                logger.error(f"错误输出: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"处理动作视频 {video_path.name} 时发生错误: {e}")
            return False
    
    def update_custom_config(self, action_id: str, video_name: str):
        """自动更新custom_config.json配置"""
        try:
            custom_config_path = Path("data/custom_config.json")
            
            # 读取现有配置
            if custom_config_path.exists():
                with open(custom_config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                config_data = []
            
            # 检查是否已存在该action_id的配置
            existing_index = None
            for i, item in enumerate(config_data):
                if item.get('audiotype') == action_id:
                    existing_index = i
                    break
            
            # 构建新的配置项
            new_config = {
                "audiotype": action_id,
                "imgpath": f"data/customvideo/{action_id}",
                "audiopath": f"data/customvideo/stat.wav",
                "description": f"自动生成的动作配置 - 来源视频: {video_name}",
                "created_time": datetime.now().isoformat()
            }
            
            # 更新或添加配置
            if existing_index is not None:
                config_data[existing_index] = new_config
                logger.info(f"更新custom_config.json中的配置: audiotype={action_id}")
            else:
                config_data.append(new_config)
                logger.info(f"添加新的custom_config.json配置: audiotype={action_id}")
            
            # 保存配置
            with open(custom_config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"custom_config.json已更新，包含 {len(config_data)} 个动作配置")
            
        except Exception as e:
            logger.error(f"更新custom_config.json失败: {e}")
    
    def remove_custom_config(self, action_id: str):
        """从custom_config.json中移除指定配置"""
        try:
            custom_config_path = Path("data/custom_config.json")
            
            if not custom_config_path.exists():
                return
            
            # 读取现有配置
            with open(custom_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 移除指定配置
            original_length = len(config_data)
            config_data = [item for item in config_data if item.get('audiotype') != action_id]
            
            if len(config_data) < original_length:
                # 保存更新后的配置
                with open(custom_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                logger.info(f"从custom_config.json中移除配置: audiotype={action_id}")
            
        except Exception as e:
            logger.error(f"移除custom_config.json配置失败: {e}")
    
    def scan_and_process(self):
        """扫描目录并处理未处理的视频"""
        try:
            logger.info(f"开始扫描动作视频目录: {self.scan_directory}")
            
            # 获取所有视频文件
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(self.scan_directory.glob(f"*{ext}"))
                video_files.extend(self.scan_directory.glob(f"*{ext.upper()}"))
            
            # 应用前缀过滤
            if self.video_prefix:
                original_count = len(video_files)
                video_files = [f for f in video_files if f.name.startswith(self.video_prefix)]
                logger.info(f"前缀过滤: 从 {original_count} 个文件中筛选出 {len(video_files)} 个符合前缀 '{self.video_prefix}' 的文件")
            
            logger.info(f"找到 {len(video_files)} 个动作视频文件")
            
            # 检查每个视频文件
            for video_path in video_files:
                if not video_path.is_file():
                    continue
                
                logger.info(f"检查动作视频: {video_path.name}")
                
                if self.is_processed(video_path):
                    logger.info(f"动作视频 {video_path.name} 已处理过，跳过")
                    continue
                
                logger.info(f"发现未处理动作视频: {video_path.name}")
                
                # 开始处理
                if self.process_video(video_path):
                    logger.info(f"动作视频 {video_path.name} 处理完成")
                else:
                    logger.error(f"动作视频 {video_path.name} 处理失败")
            
            logger.info("动作视频扫描完成")
            
        except Exception as e:
            logger.error(f"动作视频扫描过程中发生错误: {e}")
    
    def start_scanning(self):
        """开始定期扫描"""
        if self.running:
            logger.warning("动作视频扫描器已在运行")
            return
        
        self.running = True
        logger.info("开始定期扫描动作视频...")
        
        def scan_loop():
            while self.running:
                try:
                    self.scan_and_process()
                except Exception as e:
                    logger.error(f"动作视频扫描循环中发生错误: {e}")
                
                # 等待下次扫描
                for _ in range(self.scan_interval):
                    if not self.running:
                        break
                    time.sleep(1)
        
        self.scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self.scan_thread.start()
    
    def stop_scanning(self):
        """停止扫描"""
        if not self.running:
            logger.warning("动作视频扫描器未在运行")
            return
        
        logger.info("正在停止动作视频扫描器...")
        self.running = False
        
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        
        logger.info("动作视频扫描器已停止")
    
    def scan_once(self):
        """执行一次扫描"""
        logger.info("执行单次动作视频扫描...")
        self.scan_and_process()
        logger.info("单次动作视频扫描完成")

def main():
    parser = argparse.ArgumentParser(description='动作视频扫描器 - 定期扫描视频文件并自动生成动作编排切图')
    parser.add_argument('--scan_dir', type=str, 
                       help='要扫描的动作视频目录')
    parser.add_argument('--action_dir', type=str,
                       help='动作数据存储目录')
    parser.add_argument('--interval', type=int,
                       help='扫描间隔（秒）')
    parser.add_argument('--once', action='store_true',
                       help='只执行一次扫描，不持续运行')
    parser.add_argument('--extensions', nargs='+', 
                       help='视频文件扩展名')
    parser.add_argument('--prefix', type=str,
                       help='视频文件名称前缀，只处理以此前缀开头的文件')
    parser.add_argument('--genaction_script', type=str,
                       help='genaction.py脚本路径')
    parser.add_argument('--config', type=str, default='action_scanner_config.json',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建扫描器
    scanner = ActionScanner(
        scan_directory=args.scan_dir,
        action_base_dir=args.action_dir,
        scan_interval=args.interval,
        video_extensions=args.extensions,
        video_prefix=args.prefix,
        genaction_script=args.genaction_script,
        config_file=args.config
    )
    
    try:
        if args.once:
            scanner.scan_once()
        else:
            scanner.start_scanning()
            
            # 等待用户中断
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n收到停止信号，正在停止动作视频扫描器...")
                scanner.stop_scanning()
                print("动作视频扫描器已停止")
                
    except Exception as e:
        print(f"程序运行出错: {e}")
        scanner.stop_scanning()

if __name__ == "__main__":
    main() 