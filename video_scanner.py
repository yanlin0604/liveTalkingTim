#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频扫描器 - 定期扫描视频文件并自动训练
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
        logging.FileHandler('video_scanner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoScanner:
    def __init__(self, 
                 scan_directory: str = "videos",
                 avatar_base_dir: str = "data/avatars",
                 scan_interval: int = 60,
                 video_extensions: List[str] = None,
                 video_prefix: str = None,
                 genavatar_script: str = "wav2lip/genavatar.py",
                 genavatar_options: Dict = None,
                 config_file: str = None):
        """
        初始化视频扫描器
        
        Args:
            scan_directory: 扫描目录
            avatar_base_dir: 头像数据存储目录
            scan_interval: 扫描间隔（秒）
            video_extensions: 视频文件扩展名列表
            video_prefix: 视频文件名称前缀，只处理以此前缀开头的文件
            genavatar_script: genavatar.py脚本路径
            genavatar_options: genavatar.py的额外选项
            config_file: 配置文件路径
        """
        # 加载配置文件
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        else:
            self.config = {}
        
        # 使用配置文件中的值，如果没有则使用参数值
        self.scan_directory = Path(scan_directory or self.config.get('scan_directory', 'videos'))
        self.avatar_base_dir = Path(avatar_base_dir or self.config.get('avatar_base_dir', 'data/avatars'))
        self.scan_interval = scan_interval or self.config.get('scan_interval', 60)
        self.video_extensions = video_extensions or self.config.get('video_extensions', ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'])
        self.video_prefix = video_prefix or self.config.get('video_prefix', None)
        self.genavatar_script = Path(genavatar_script or self.config.get('genavatar_script', 'wav2lip/genavatar.py'))
        self.genavatar_options = genavatar_options or self.config.get('genavatar_options', {})
        
        self.processed_videos = set()
        self.running = False
        self.scan_thread = None
        
        # 确保目录存在
        self.scan_directory.mkdir(parents=True, exist_ok=True)
        self.avatar_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已处理视频记录
        self.processed_file = self.avatar_base_dir / "processed_videos.json"
        self.load_processed_videos()
        
        logger.info(f"视频扫描器初始化完成")
        logger.info(f"扫描目录: {self.scan_directory}")
        logger.info(f"头像目录: {self.avatar_base_dir}")
        logger.info(f"扫描间隔: {self.scan_interval}秒")
        logger.info(f"视频扩展名: {self.video_extensions}")
        logger.info(f"视频前缀过滤: {self.video_prefix or '无'}")
        logger.info(f"genavatar选项: {self.genavatar_options}")
    
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
        """计算文件的MD5值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件MD5失败 {file_path}: {e}")
            return ""
    
    def get_video_info(self, video_path: Path) -> Dict:
        """获取视频信息，包括MD5和文件大小"""
        try:
            stat = video_path.stat()
            return {
                'path': str(video_path),
                'md5': self.calculate_file_md5(video_path),
                'size': stat.st_size,
                'modified_time': stat.st_mtime
            }
        except Exception as e:
            logger.error(f"获取视频信息失败 {video_path}: {e}")
            return {
                'path': str(video_path),
                'md5': '',
                'size': 0,
                'modified_time': 0
            }
    
    def load_processed_videos(self):
        """加载已处理视频记录"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_videos = data.get('processed_videos', {})
                logger.info(f"加载了 {len(self.processed_videos)} 个已处理视频记录")
            except Exception as e:
                logger.error(f"加载已处理视频记录失败: {e}")
                self.processed_videos = {}
        else:
            self.processed_videos = {}
    
    def save_processed_videos(self):
        """保存已处理视频记录"""
        try:
            data = {
                'processed_videos': self.processed_videos,
                'last_updated': datetime.now().isoformat(),
                'format_version': '2.0'  # 标记新格式
            }
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"已保存 {len(self.processed_videos)} 个处理记录（包含MD5信息）")
        except Exception as e:
            logger.error(f"保存已处理视频记录失败: {e}")
    
    def is_video_file(self, file_path: Path) -> bool:
        """判断是否为视频文件"""
        return file_path.suffix.lower() in self.video_extensions
    
    def is_trained(self, video_path: Path) -> bool:
        """判断视频是否已经训练过"""
        video_path_str = str(video_path)
        
        # 检查是否在已处理列表中
        if video_path_str in self.processed_videos:
            recorded_info = self.processed_videos[video_path_str]
            current_md5 = self.calculate_file_md5(video_path)
            
            if current_md5 == recorded_info.get('md5', ''):
                logger.info(f"视频 {video_path.name} 已训练过（MD5匹配）")
                return True
            else:
                logger.warning(f"视频 {video_path.name} MD5不匹配，可能文件已更改，需要重新训练")
                # 删除旧的记录
                del self.processed_videos[video_path_str]
                self.save_processed_videos()
                return False
        
        # 检查是否存在对应的头像目录
        video_name = video_path.stem
        avatar_dir = self.avatar_base_dir / f"wav2lip_{video_name}"
        
        if avatar_dir.exists():
            # 检查是否有必要的训练文件
            face_imgs_dir = avatar_dir / "face_imgs"
            coords_file = avatar_dir / "coords.pkl"
            
            if face_imgs_dir.exists() and coords_file.exists():
                # 检查是否有足够的人脸图片
                face_images = list(face_imgs_dir.glob("*.png"))
                if len(face_images) > 0:
                    logger.info(f"视频 {video_name} 已训练过，找到 {len(face_images)} 张人脸图片")
                    # 更新记录，添加MD5信息
                    video_info = self.get_video_info(video_path)
                    self.processed_videos[video_path_str] = video_info
                    self.save_processed_videos()
                    return True
        
        return False
    
    def generate_avatar_id(self, video_path: Path) -> str:
        """生成头像ID"""
        video_name = video_path.stem
        # 清理文件名，移除特殊字符
        clean_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_'))
        return f"wav2lip_{clean_name}"
    
    def train_video(self, video_path: Path) -> bool:
        """训练视频"""
        try:
            video_name = video_path.stem
            avatar_id = self.generate_avatar_id(video_path)
            
            logger.info(f"开始训练视频: {video_name}")
            logger.info(f"头像ID: {avatar_id}")
            
            # 构建genavatar.py命令
            cmd = [
                sys.executable,
                str(self.genavatar_script),
                "--video_path", str(video_path),
                "--avatar_id", avatar_id,
                "--avatar_base_dir", str(self.avatar_base_dir)
            ]
            
            # 添加配置文件中的选项
            options = self.genavatar_options
            if options:
                if 'img_size' in options:
                    cmd.extend(["--img_size", str(options['img_size'])])
                if 'batch_frames' in options:
                    cmd.extend(["--batch_frames", str(options['batch_frames'])])
                if options.get('use_video2imgs', False):
                    cmd.append("--use_video2imgs")
                if options.get('nosmooth', False):
                    cmd.append("--nosmooth")
                if 'pads' in options:
                    cmd.extend(["--pads"] + [str(p) for p in options['pads']])
                if 'face_det_batch_size' in options:
                    cmd.extend(["--face_det_batch_size", str(options['face_det_batch_size'])])
                if 'resize_factor' in options:
                    cmd.extend(["--resize_factor", str(options['resize_factor'])])
                if options.get('force_cpu', False):
                    cmd.append("--force_cpu")
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行训练
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                logger.info(f"视频 {video_name} 训练成功")
                # 添加到已处理列表，包含MD5信息
                video_info = self.get_video_info(video_path)
                self.processed_videos[str(video_path)] = video_info
                self.save_processed_videos()
                return True
            else:
                logger.error(f"视频 {video_name} 训练失败")
                logger.error(f"错误输出: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"训练视频 {video_path} 时发生错误: {e}")
            return False
    
    def scan_and_train(self):
        """扫描目录并训练未处理的视频"""
        try:
            logger.info(f"开始扫描目录: {self.scan_directory}")
            
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
            
            logger.info(f"找到 {len(video_files)} 个视频文件")
            
            # 检查每个视频文件
            for video_path in video_files:
                if not video_path.is_file():
                    continue
                
                logger.info(f"检查视频: {video_path.name}")
                
                if self.is_trained(video_path):
                    logger.info(f"视频 {video_path.name} 已训练过，跳过")
                    continue
                
                logger.info(f"发现未训练视频: {video_path.name}")
                
                # 开始训练
                if self.train_video(video_path):
                    logger.info(f"视频 {video_path.name} 训练完成")
                else:
                    logger.error(f"视频 {video_path.name} 训练失败")
            
            logger.info("扫描完成")
            
        except Exception as e:
            logger.error(f"扫描过程中发生错误: {e}")
    
    def start_scanning(self):
        """开始定期扫描"""
        if self.running:
            logger.warning("扫描器已在运行")
            return
        
        self.running = True
        logger.info("开始定期扫描...")
        
        def scan_loop():
            while self.running:
                try:
                    self.scan_and_train()
                except Exception as e:
                    logger.error(f"扫描循环中发生错误: {e}")
                
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
            logger.warning("扫描器未在运行")
            return
        
        logger.info("正在停止扫描器...")
        self.running = False
        
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        
        logger.info("扫描器已停止")
    
    def scan_once(self):
        """执行一次扫描"""
        logger.info("执行单次扫描...")
        self.scan_and_train()
        logger.info("单次扫描完成")

def main():
    parser = argparse.ArgumentParser(description='视频扫描器 - 定期扫描视频文件并自动训练')
    parser.add_argument('--scan_dir', type=str, 
                       help='要扫描的视频目录')
    parser.add_argument('--avatar_dir', type=str,
                       help='头像数据存储目录')
    parser.add_argument('--interval', type=int,
                       help='扫描间隔（秒）')
    parser.add_argument('--once', action='store_true',
                       help='只执行一次扫描，不持续运行')
    parser.add_argument('--extensions', nargs='+', 
                       help='视频文件扩展名')
    parser.add_argument('--prefix', type=str,
                       help='视频文件名称前缀，只处理以此前缀开头的文件')
    parser.add_argument('--genavatar_script', type=str,
                       help='genavatar.py脚本路径')
    parser.add_argument('--config', type=str, default='video_scanner_config.json',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建扫描器
    scanner = VideoScanner(
        scan_directory=args.scan_dir,
        avatar_base_dir=args.avatar_dir,
        scan_interval=args.interval,
        video_extensions=args.extensions,
        video_prefix=args.prefix,
        genavatar_script=args.genavatar_script,
        config_file=args.config
    )
    
    try:
        if args.once:
            # 只执行一次扫描
            scanner.scan_once()
        else:
            # 持续扫描
            scanner.start_scanning()
            
            # 等待用户中断
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("收到中断信号")
                scanner.stop_scanning()
                
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        scanner.stop_scanning()

if __name__ == "__main__":
    main() 