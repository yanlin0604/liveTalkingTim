from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
import shutil 
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Wav2Lip模型视频处理程序')
parser.add_argument('--img_size', default=256, type=int, help='输出人脸图片尺寸')
parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str, help='角色名')
parser.add_argument('--avatar_base_dir', default='../data/avatars', type=str, help='头像数据存储基础目录')
parser.add_argument('--video_path', default='', type=str, help='输入视频路径')
parser.add_argument('--batch_frames', type=int, default=150,
                    help='每批处理的帧数，内存不足时可调小此值')
parser.add_argument('--use_video2imgs', action='store_true',
                    help='使用原来的video2imgs方式处理视频')
parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='是否关闭人脸框平滑处理')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='人脸区域填充值 (上, 下, 左, 右)，建议调整以包含下巴')
parser.add_argument('--face_det_batch_size', type=int, 
                    help='人脸检测的批处理大小', default=6)
parser.add_argument('--resize_factor', type=float, default=1.0,
                    help='调整输入图像大小的因子，值小于1可减少内存使用')
parser.add_argument('--force_cpu', action='store_true',
                    help='强制使用CPU进行推理，解决CUDA兼容性问题')
args = parser.parse_args()

# 确定使用的设备
if args.force_cpu:
    device = 'cpu'
    print('强制使用CPU进行推理')
else:
    # 即使有CUDA可用，也尝试检测是否有兼容性问题
    try:
        print('检测CUDA兼容性...')
        # 尝试在CUDA上进行一个简单操作
        if torch.cuda.is_available():
            test_tensor = torch.zeros(1).cuda()
            test_tensor = test_tensor + 1
            # 尝试进行NMS操作，可能会失败
            try:
                from torchvision.ops import nms
                boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]).cuda().float()
                scores = torch.tensor([0.9, 0.8]).cuda()
                nms(boxes, scores, 0.5)
                device = 'cuda'
                print('使用CUDA进行推理')
            except Exception as e:
                print(f'CUDA NMS操作不支持: {e}')
                device = 'cpu'
                print('由于NMS兼容性问题，切换到CPU进行推理')
        else:
            device = 'cpu'
            print('未检测到CUDA，使用CPU进行推理')
    except Exception as e:
        device = 'cpu'
        print(f'CUDA测试失败: {e}，使用CPU进行推理')

# 初始化YOLO模型 - 始终在CPU上加载模型，需要时再转移到CUDA
# 使用绝对路径确保模型文件能被找到
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
yolo_model_path = os.path.join(project_root, 'models', 'yolo', 'yolov8n-face.pt')

if not os.path.exists(yolo_model_path):
    # 尝试其他可能的路径
    alternative_paths = [
        os.path.join(project_root, 'models', 'yolo', 'yolov8n-face.pt'),
        os.path.join(os.getcwd(), 'models', 'yolo', 'yolov8n-face.pt'),
        'models/yolo/yolov8n-face.pt',
        '../models/yolo/yolov8n-face.pt',
        '../../models/yolo/yolov8n-face.pt'
    ]
    
    for path in alternative_paths:
        if os.path.exists(path):
            yolo_model_path = path
            break
    else:
        raise FileNotFoundError(f"找不到YOLO模型文件。尝试过的路径: {alternative_paths}")

print(f"使用YOLO模型文件: {yolo_model_path}")
face_det = YOLO(yolo_model_path)
if device == 'cpu':
    # 确保模型在CPU上
    face_det.to('cpu')
else:
    # 尝试将模型转移到CUDA
    try:
        face_det.to('cuda')
    except Exception as e:
        print(f'将模型转移到CUDA失败: {e}，使用CPU代替')
        device = 'cpu'
        face_det.to('cpu')

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

def read_imgs(img_list):
    frames = []
    print('读取图片到内存...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
    print('即将开始人脸检测...')
    predictions = []
    global device, face_det  # 全局变量，可能需要修改
    total_batches = len(images) // args.face_det_batch_size + (1 if len(images) % args.face_det_batch_size != 0 else 0)
    
    print(f"progress:总共需要处理 {total_batches} 批次")
    
    for batch_idx in range(0, len(images), args.face_det_batch_size):
        try:
            current_batch = images[batch_idx:batch_idx + args.face_det_batch_size]
            print(f"progress:正在处理第 {batch_idx//args.face_det_batch_size + 1}/{total_batches} 批次，共 {len(current_batch)} 帧")
            
            # 使用YOLO处理当前批次
            batch_predictions = []
            for img in current_batch:
                # 如果指定了resize_factor，调整图像大小
                if args.resize_factor != 1.0:
                    h, w = img.shape[:2]
                    new_h, new_w = int(h * args.resize_factor), int(w * args.resize_factor)
                    resized_img = cv2.resize(img, (new_w, new_h))
                    # 在缩小图像上检测人脸
                    results = face_det(resized_img, conf=0.01, iou=0.5, device=device)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    # 将坐标映射回原始大小
                    if len(boxes) > 0:
                        # 确保是可修改的数组
                        boxes = boxes.copy()
                        boxes[:, [0, 2]] = boxes[:, [0, 2]] / args.resize_factor
                        boxes[:, [1, 3]] = boxes[:, [1, 3]] / args.resize_factor
                        # 确保坐标是有效的
                        h, w = img.shape[:2]
                        box = boxes[0]  # 取第一个检测到的人脸
                        # 确保框在图像范围内
                        box[0] = max(0, min(box[0], w-1))
                        box[1] = max(0, min(box[1], h-1))
                        box[2] = max(box[0]+1, min(box[2], w))
                        box[3] = max(box[1]+1, min(box[3], h))
                        batch_predictions.append(box)  # 取第一个检测到的人脸
                    else:
                        batch_predictions.append(None)
                else:
                    # 在原图上检测
                    results = face_det(img, conf=0.01, iou=0.5, device=device)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    if len(boxes) > 0:
                        batch_predictions.append(boxes[0])  # 取第一个检测到的人脸
                    else:
                        batch_predictions.append(None)
            
            predictions.extend(batch_predictions)
            
            # 输出进度
            progress = int((batch_idx + len(current_batch)) / len(images) * 100)
            print(f"progress:人脸检测进度: {progress}%")
            
        except Exception as e:
            error_str = str(e)
            print(f"progress:处理出现错误: {error_str}")
            
            # 检查是否是CUDA兼容性问题
            if "torchvision::nms" in error_str or "CUDA" in error_str:
                if device == "cuda":
                    print("progress:检测到CUDA兼容性问题，切换到CPU")
                    device = 'cpu'
                    face_det.to('cpu')
                    # 重试当前批次
                    batch_idx -= args.face_det_batch_size
                    continue
            
            # 内存不足问题处理
            if args.face_det_batch_size > 1:
                # 减小批次大小并重试
                args.face_det_batch_size = max(1, args.face_det_batch_size // 2)
                print(f'progress:检测到内存溢出，减小批处理大小为: {args.face_det_batch_size}')
                predictions = predictions[:batch_idx]  # 保留已处理的结果
                batch_idx -= args.face_det_batch_size  # 回退到上一个批次
                continue
            elif args.resize_factor > 0.1:
                # 批处理大小已为1，尝试减小图像大小
                old_factor = args.resize_factor
                args.resize_factor *= 0.5
                print(f"progress:批处理大小已为1，尝试将resize_factor从{old_factor}减小到{args.resize_factor}")
                batch_idx -= args.face_det_batch_size
                continue
            else:
                # 所有方法都失败了
                print(f"progress:处理失败，已尝试所有可能的优化方法")
                raise RuntimeError(f'处理失败: {error_str}。请尝试使用更小的视频或更低分辨率的视频。')

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    print('progress:开始处理检测结果...')
    for i, (rect, image) in enumerate(zip(predictions, images)):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('未检测到人脸！请确保视频中所有帧都包含人脸。')

        # 确保坐标是浮点数，以便于平滑处理
        y1 = float(max(0, rect[1] - pady1))
        y2 = float(min(image.shape[0], rect[3] + pady2))
        x1 = float(max(0, rect[0] - padx1))
        x2 = float(min(image.shape[1], rect[2] + padx2))
        
        # 检查框是否有效
        if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
            print(f"警告：检测到无效的人脸框 [{x1},{y1},{x2},{y2}]，调整到有效范围")
            y1 = float(max(0, min(y1, image.shape[0]-1)))
            y2 = float(max(y1+1, min(y2, image.shape[0])))
            x1 = float(max(0, min(x1, image.shape[1]-1)))
            x2 = float(max(x1+1, min(x2, image.shape[1])))
        
        results.append([x1, y1, x2, y2])
        
        if i % 10 == 0:  # 每处理10帧输出一次进度
            progress = int((i + 1) / len(predictions) * 100)
            print(f"progress:结果处理进度: {progress}%")

    boxes = np.array(results)
    if not args.nosmooth: 
        print('progress:正在平滑处理人脸框...')
        boxes = get_smoothened_boxes(boxes, T=5)
    
    print('progress:正在生成最终结果...')
    # 确保坐标都是整数
    results = []
    for image, (x1, y1, x2, y2) in zip(images, boxes):
        # 将浮点数转换为整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 确保边界不超出图像范围
        y1 = max(0, y1)
        y2 = min(image.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(image.shape[1], x2)
        # 确保区域有效（宽高大于0）
        if y2 <= y1 or x2 <= x1:
            print(f"警告：检测到无效的人脸框 [{x1},{y1},{x2},{y2}]，使用整个图像")
            y1, x1 = 0, 0
            y2, x2 = image.shape[0], image.shape[1]
        
        try:
            face_region = image[y1:y2, x1:x2]
            results.append([face_region, (y1, y2, x1, x2)])
        except Exception as e:
            print(f"警告：裁剪人脸区域时出错: {e}，坐标: [{x1},{y1},{x2},{y2}]，图像大小: {image.shape}")
            # 使用整个图像作为备选
            results.append([image, (0, image.shape[0], 0, image.shape[1])])

    print('progress:人脸检测完成')
    return results

def process_video_batch(frames, face_imgs_path, start_idx):
    print(f"progress:正在处理第 {start_idx} - {start_idx + len(frames)} 帧...")
    face_det_results = face_detect(frames)
    coord_list = []
    idx = start_idx
    print(f"progress:本批次检测到{len(face_det_results)}张人脸")
    for i, (frame, coords) in enumerate(face_det_results):
        resized_crop_frame = cv2.resize(frame, (args.img_size, args.img_size))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append(coords)
        idx += 1
        if i % 10 == 0:  # 每处理10帧输出一次进度
            progress = int((i / len(face_det_results)) * 100)
            print(f"progress:{progress}%")
    return coord_list

def clean_directory(directory):
    """清空目录中的所有文件和子目录"""
    if os.path.exists(directory):
        print(f"清理目录: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)

if __name__ == "__main__":
    try:
        # 打印所有参数到日志
        print("=" * 80)
        print("📋 Wav2Lip视频处理参数配置:")
        print("=" * 80)
        print(f"  🎭 角色名称 (avatar_id): {args.avatar_id}")
        print(f"  📁 头像基础目录 (avatar_base_dir): {args.avatar_base_dir}")
        print(f"  🎬 输入视频路径 (video_path): {args.video_path}")
        print(f"  📐 输出图片尺寸 (img_size): {args.img_size}x{args.img_size}")
        print(f"  📦 批处理帧数 (batch_frames): {args.batch_frames}")
        print(f"  🔍 人脸检测批大小 (face_det_batch_size): {args.face_det_batch_size}")
        print(f"  📏 图像缩放因子 (resize_factor): {args.resize_factor}")
        print(f"  🔲 人脸填充值 (pads): 上={args.pads[0]}, 下={args.pads[1]}, 左={args.pads[2]}, 右={args.pads[3]}")
        print(f"  🎯 使用video2imgs模式 (use_video2imgs): {args.use_video2imgs}")
        print(f"  🔄 关闭平滑处理 (nosmooth): {args.nosmooth}")
        print(f"  💻 强制使用CPU (force_cpu): {args.force_cpu}")
        print(f"  ⚙️  实际使用设备 (device): {device}")
        print("=" * 80)
        
        # 创建temp目录（如果不存在）
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        avatar_path = f"{args.avatar_base_dir}/{args.avatar_id}"
        full_imgs_path = f"{avatar_path}/full_imgs" 
        face_imgs_path = f"{avatar_path}/face_imgs" 
        coords_path = f"{avatar_path}/coords.pkl"
        
        print(f"progress:形象路径: {avatar_path}，切图路径: {full_imgs_path}，人脸切图路径: {face_imgs_path}，坐标路径: {coords_path}")

        # 确保目录存在
        osmakedirs([avatar_path, full_imgs_path, face_imgs_path])

        # 清理旧文件
        print("progress:开始清理旧文件...")
        clean_directory(full_imgs_path)
        clean_directory(face_imgs_path)
        if os.path.exists(coords_path):
            os.remove(coords_path)
            print(f"progress:已删除旧的坐标文件: {coords_path}")

        if args.use_video2imgs:
            # 使用原来的处理方式
            print("progress:使用video2imgs方式处理视频...")
            video2imgs(args.video_path, full_imgs_path, ext='.jpg')
            input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.jpg')))
            frames = read_imgs(input_img_list)
            face_det_results = face_detect(frames)
            coord_list = []
            idx = 0
            print(f"progress:共检测到{len(face_det_results)}张人脸")
            total = len(face_det_results)
            for i, (frame, coords) in enumerate(face_det_results):
                resized_crop_frame = cv2.resize(frame, (args.img_size, args.img_size))
                cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
                coord_list.append(coords)
                idx += 1
                if i % 10 == 0:  # 每处理10帧输出一次进度
                    progress = int((i / total) * 100)
                    print(f"progress:{progress}%")
            
            print(f"progress:正在写入坐标数据到文件: {coords_path}")
            with open(coords_path, 'wb') as f:
                pickle.dump(coord_list, f)
            print(f"progress:处理完成! 共处理了 {idx} 帧")
        else:
            # 使用新的分批处理方式
            cap = cv2.VideoCapture(args.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"progress:视频总帧数: {total_frames}")
            
            all_coords = []
            batch_frames = []
            frame_count = 0
            batch_count = 0
            
            print("progress:开始分批处理视频...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存完整帧
                cv2.imwrite(f"{full_imgs_path}/{frame_count:08d}.png", frame)
                
                batch_frames.append(frame)
                frame_count += 1
                
                # 输出读取进度
                if frame_count % 30 == 0:  # 每30帧输出一次进度
                    progress = int((frame_count / total_frames) * 50)  # 前半部分进度(0-50%)
                    print(f"progress:{progress}%")
                
                if len(batch_frames) >= args.batch_frames or frame_count == total_frames:
                    print(f"\nprogress:正在处理第 {batch_count + 1} 批, 帧数范围: {batch_count * args.batch_frames} - {frame_count}")
                    coords = process_video_batch(batch_frames, face_imgs_path, batch_count * args.batch_frames)
                    all_coords.extend(coords)
                    batch_frames = []
                    batch_count += 1
                    
                    # 输出批次处理进度
                    progress = 50 + int((batch_count * args.batch_frames / total_frames) * 50)  # 后半部分进度(50-100%)
                    print(f"progress:{progress}%")
            
            cap.release()
            
            print(f"progress:正在写入坐标数据到文件: {coords_path}")
            with open(coords_path, 'wb') as f:
                pickle.dump(all_coords, f)
            
            print(f"progress:处理完成! 共处理了 {frame_count} 帧，分为 {batch_count} 个批次")
            print("progress:100%")
    except Exception as e:
        print(f"progress:处理失败: {e}")
        raise e

