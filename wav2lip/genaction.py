from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
import shutil 

parser = argparse.ArgumentParser(description='动作编排视频切图生成程序')
parser.add_argument('--avatar_id', default='action_video1', type=str, help='动作视频名称')
parser.add_argument('--video_path', default='', type=str, help='输入视频路径')
parser.add_argument('--batch_frames', type=int, default=150,
                    help='每批处理的帧数，内存不足时可调小此值')
parser.add_argument('--use_video2imgs', action='store_true',
                    help='使用原来的video2imgs方式处理视频')
parser.add_argument('--resize_factor', type=float, default=1.0,
                    help='调整输入图像大小的因子，值小于1可减少内存使用')
parser.add_argument('--force_cpu', action='store_true',
                    help='强制使用CPU进行推理，解决CUDA兼容性问题')
args = parser.parse_args()

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext='.jpg', cut_frame=10000000):
    print(f"即将使用OpenCV将视频: {vid_path} 转换为图片")
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"检测到视频帧率: {fps}, 总帧数: {frame_count}, 预计时长: {frame_count/fps if fps else 0:.2f}秒")
    count = 0
    while True:
        if count >= frame_count or count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break
    print("视频转换完成")

def clean_directory(directory):
    """清空目录中的所有文件和子目录"""
    if os.path.exists(directory):
        print(f"清理目录: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)

if __name__ == "__main__":
    try:
        # 创建temp目录（如果不存在）
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        # 使用data/customvideo作为基础目录
        action_path = f"data/customvideo/{args.avatar_id}"
        
        print(f"progress:动作视频路径: {action_path}")

        # 确保目录存在
        osmakedirs([action_path])

        # 清理旧文件
        print("progress:开始清理旧文件...")
        clean_directory(action_path)
        
        print("progress:当前参数设置:", args)

        if args.use_video2imgs:
            # 使用原来的处理方式
            print("progress:使用video2imgs方式处理视频...")
            video2imgs(args.video_path, action_path, ext='.jpg')
            print(f"progress:处理完成!")
        else:
            # 使用新的分批处理方式
            cap = cv2.VideoCapture(args.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"progress:视频总帧数: {total_frames}")
            
            frame_count = 0
            
            print("progress:开始处理视频帧...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 如果指定了resize_factor，调整图像大小
                if args.resize_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * args.resize_factor), int(w * args.resize_factor)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # 保存完整帧
                cv2.imwrite(f"{action_path}/{frame_count:08d}.png", frame)
                frame_count += 1
                
                # 输出读取进度
                if frame_count % 30 == 0:  # 每30帧输出一次进度
                    progress = int((frame_count / total_frames) * 100)
                    print(f"progress:{progress}%")
            
            cap.release()
            
            print(f"progress:处理完成! 共处理了 {frame_count} 帧")
            print("progress:100%")
            
    except Exception as e:
        print(f"progress:处理失败: {e}")
        raise e 