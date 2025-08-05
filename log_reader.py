#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log File Reader Script
读取日志文件的Python脚本，支持自定义路径和行数
"""

import os
import sys
import argparse
import time
import threading
from pathlib import Path
from typing import List, Optional, Callable


class LogReader:
    """日志文件读取器"""
    
    def __init__(self, file_path: str):
        """
        初始化日志读取器
        
        Args:
            file_path (str): 日志文件路径
        """
        self.file_path = Path(file_path)
        
    def validate_file(self) -> bool:
        """
        验证文件是否存在且可读
        
        Returns:
            bool: 文件是否有效
        """
        if not self.file_path.exists():
            print(f"错误: 文件不存在 - {self.file_path}")
            return False
            
        if not self.file_path.is_file():
            print(f"错误: 路径不是文件 - {self.file_path}")
            return False
            
        if not os.access(self.file_path, os.R_OK):
            print(f"错误: 文件不可读 - {self.file_path}")
            return False
            
        return True
    
    def read_lines(self, num_lines: Optional[int] = None, from_end: bool = True) -> List[str]:
        """
        读取指定行数的日志内容
        
        Args:
            num_lines (int, optional): 要读取的行数，None表示读取全部
            from_end (bool): True表示从文件末尾读取，False表示从开头读取
            
        Returns:
            List[str]: 读取的行列表
        """
        if not self.validate_file():
            return []
            
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as file:
                if num_lines is None:
                    # 读取全部行
                    lines = file.readlines()
                elif from_end:
                    # 从末尾读取指定行数
                    lines = file.readlines()
                    lines = lines[-num_lines:] if len(lines) > num_lines else lines
                else:
                    # 从开头读取指定行数
                    lines = []
                    for i, line in enumerate(file):
                        if i >= num_lines:
                            break
                        lines.append(line)
                        
            return [line.rstrip('\n\r') for line in lines]
            
        except UnicodeDecodeError:
            try:
                # 尝试使用gbk编码
                with open(self.file_path, 'r', encoding='gbk', errors='ignore') as file:
                    if num_lines is None:
                        lines = file.readlines()
                    elif from_end:
                        lines = file.readlines()
                        lines = lines[-num_lines:] if len(lines) > num_lines else lines
                    else:
                        lines = []
                        for i, line in enumerate(file):
                            if i >= num_lines:
                                break
                            lines.append(line)
                return [line.rstrip('\n\r') for line in lines]
            except Exception as e:
                print(f"读取文件时出错: {e}")
                return []
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return []
    
    def get_file_info(self) -> dict:
        """
        获取文件信息
        
        Returns:
            dict: 文件信息字典
        """
        if not self.validate_file():
            return {}
            
        stat = self.file_path.stat()
        return {
            'path': str(self.file_path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'exists': True
        }
    
    def tail_follow(self, callback: Callable[[str], None], interval: float = 1.0, stop_event: Optional[threading.Event] = None) -> None:
        """
        实时跟踪文件变化（类似tail -f）
        
        Args:
            callback: 处理新行的回调函数
            interval: 检查间隔（秒）
            stop_event: 停止事件
        """
        if not self.validate_file():
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as file:
                # 移动到文件末尾
                file.seek(0, 2)
                
                while True:
                    if stop_event and stop_event.is_set():
                        break
                        
                    line = file.readline()
                    if line:
                        callback(line.rstrip('\n\r'))
                    else:
                        time.sleep(interval)
                        
        except UnicodeDecodeError:
            try:
                with open(self.file_path, 'r', encoding='gbk', errors='ignore') as file:
                    file.seek(0, 2)
                    
                    while True:
                        if stop_event and stop_event.is_set():
                            break
                            
                        line = file.readline()
                        if line:
                            callback(line.rstrip('\n\r'))
                        else:
                            time.sleep(interval)
            except Exception as e:
                print(f"实时读取文件时出错: {e}")
        except Exception as e:
            print(f"实时读取文件时出错: {e}")
    
    def get_recent_lines(self, num_lines: int = 10) -> List[str]:
        """
        获取文件最近的几行（用于实时读取的初始显示）
        
        Args:
            num_lines: 要获取的行数
            
        Returns:
            List[str]: 最近的行列表
        """
        return self.read_lines(num_lines=num_lines, from_end=True)


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description='日志文件读取器 - 读取并显示日志文件内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python log_reader.py /mnt/disk1/ftp/data/60397193/logs/start.log
  python log_reader.py /path/to/log.txt -n 50
  python log_reader.py /path/to/log.txt -n 100 --from-start
        """
    )
    
    parser.add_argument(
        'file_path',
        help='日志文件路径'
    )
    
    parser.add_argument(
        '-n', '--lines',
        type=int,
        default=50,
        help='要读取的行数 (默认: 50行)'
    )
    
    parser.add_argument(
        '--from-start',
        action='store_true',
        help='从文件开头读取 (默认从末尾读取)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='读取全部内容'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='显示文件信息'
    )
    
    parser.add_argument(
        '-f', '--follow',
        action='store_true',
        help='实时跟踪文件变化（类似tail -f）'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='实时读取检查间隔（秒，默认1.0）'
    )
    
    args = parser.parse_args()
    
    # 创建日志读取器
    reader = LogReader(args.file_path)
    
    # 显示文件信息
    if args.info:
        info = reader.get_file_info()
        if info:
            print(f"文件路径: {info['path']}")
            print(f"文件大小: {info['size']} 字节")
            print(f"修改时间: {info['modified']}")
            print("-" * 50)
    
    # 实时跟踪模式
    if args.follow:
        print(f"开始实时跟踪文件: {args.file_path}")
        print(f"检查间隔: {args.interval} 秒")
        print("按 Ctrl+C 停止跟踪")
        print("=" * 80)
        
        # 先显示最近的几行作为上下文
        recent_lines = reader.get_recent_lines(args.lines)
        if recent_lines:
            print("最近的日志内容:")
            for i, line in enumerate(recent_lines, 1):
                print(f"{i:4d}: {line}")
            print("-" * 80)
            print("实时日志 (新增内容):")
        
        # 创建停止事件
        stop_event = threading.Event()
        
        def print_new_line(line: str):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {line}")
        
        # 启动实时跟踪线程
        follow_thread = threading.Thread(
            target=reader.tail_follow,
            args=(print_new_line, args.interval, stop_event)
        )
        follow_thread.daemon = True
        follow_thread.start()
        
        try:
            # 主线程等待用户中断
            while follow_thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n停止实时跟踪...")
            stop_event.set()
            follow_thread.join(timeout=2)
            print("已停止")
        
        return
    
    # 普通读取模式
    if args.all:
        lines = reader.read_lines(num_lines=None)
        print(f"读取全部内容 ({len(lines)} 行):")
    else:
        lines = reader.read_lines(
            num_lines=args.lines,
            from_end=not args.from_start
        )
        direction = "开头" if args.from_start else "末尾"
        print(f"从文件{direction}读取 {len(lines)} 行:")
    
    print("=" * 80)
    
    # 显示内容
    for i, line in enumerate(lines, 1):
        print(f"{i:4d}: {line}")
    
    if not lines:
        print("没有读取到任何内容")


if __name__ == "__main__":
    main()
