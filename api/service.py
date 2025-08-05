import json
import asyncio
import threading
import subprocess
import psutil
import os
import time
from aiohttp import web
from typing import Dict, Optional
from logger import logger


class ServiceAPI:
    """服务管理API类"""
    
    def __init__(self):
        self.service_status = "stopped"  # running, stopped, starting, stopping
        self.service_process = None
        self.service_pid = None
        self.service_lock = threading.Lock()
        self.start_time = None
        self.stop_time = None
        
        # 服务配置
        self.service_script = "start.py"  # 启动脚本
        self.target_processes = ["app.py", "start.py", "run_dynamic.py"]  # 要管理的进程
        self.log_dir = "/mnt/disk1/ftp/data/60397193/logs"  # 日志目录
        self.conda_env = "nerfstream"  # conda环境
        
        # 初始化时检查服务状态
        self._check_service_status()
    
    def _check_service_status(self):
        """检查当前服务状态"""
        try:
            # 检查是否有目标进程在运行
            running_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    for target in self.target_processes:
                        if target in cmdline:
                            running_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline
                            })
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            with self.service_lock:
                if running_processes:
                    self.service_status = "running"
                    self.service_pid = running_processes[0]['pid']
                    # 尝试获取启动时间
                    try:
                        proc = psutil.Process(self.service_pid)
                        self.start_time = proc.create_time()
                    except:
                        self.start_time = time.time()
                    logger.info(f"检测到服务正在运行，PID: {self.service_pid}")
                else:
                    self.service_status = "stopped"
                    self.service_pid = None
                    self.start_time = None
                    logger.info("检测到服务已停止")
                    
        except Exception as e:
            logger.error(f"检查服务状态失败: {e}")
            self.service_status = "stopped"
    
    def _kill_service_processes(self):
        """杀掉服务相关进程"""
        try:
            killed_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    for target in self.target_processes:
                        if target in cmdline:
                            pid = proc.info['pid']
                            proc.terminate()
                            killed_pids.append(pid)
                            logger.info(f"终止进程: {pid} ({target})")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 等待进程退出
            if killed_pids:
                time.sleep(2)
                
                # 强制杀掉仍在运行的进程
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        for target in self.target_processes:
                            if target in cmdline:
                                pid = proc.info['pid']
                                proc.kill()
                                logger.info(f"强制终止进程: {pid} ({target})")
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            return True
        except Exception as e:
            logger.error(f"杀掉服务进程失败: {e}")
            return False
    
    def _start_service_process(self):
        """启动服务进程"""
        try:
            # 确保日志目录存在
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 构建启动命令
            conda_init = "/mnt/disk1/ftp/data/60397193/miniconda3/etc/profile.d/conda.sh"
            log_file = os.path.join(self.log_dir, "start.log")
            
            # 使用bash启动，包含conda环境激活
            cmd = [
                "bash", "-c",
                f"source {conda_init} && conda activate {self.conda_env} && python {self.service_script} > {log_file} 2>&1"
            ]
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            self.service_process = process
            self.service_pid = process.pid
            self.start_time = time.time()
            
            logger.info(f"启动服务进程，PID: {self.service_pid}")
            return True
            
        except Exception as e:
            logger.error(f"启动服务进程失败: {e}")
            return False
    
    async def get_status(self, request: web.Request) -> web.Response:
        """查询服务状态接口"""
        try:
            # 实时检查服务状态
            self._check_service_status()
            
            with self.service_lock:
                status_info = {
                    "status": self.service_status,
                    "pid": self.service_pid,
                    "start_time": self.start_time,
                    "stop_time": self.stop_time,
                    "uptime": None
                }
                
                # 计算运行时间
                if self.start_time and self.service_status == "running":
                    status_info["uptime"] = time.time() - self.start_time
                
                # 获取进程详细信息
                if self.service_pid:
                    try:
                        proc = psutil.Process(self.service_pid)
                        status_info["cpu_percent"] = proc.cpu_percent()
                        status_info["memory_mb"] = proc.memory_info().rss / 1024 / 1024
                        status_info["cmdline"] = ' '.join(proc.cmdline())
                    except psutil.NoSuchProcess:
                        status_info["pid"] = None
                        status_info["status"] = "stopped"
                
                logger.info(f"查询服务状态: {self.service_status}, PID: {self.service_pid}")
                return web.json_response({
                    "success": True,
                    "message": "查询服务状态成功",
                    "data": status_info
                })
                
        except Exception as e:
            logger.error(f"查询服务状态失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"查询服务状态失败: {str(e)}",
                "data": None
            }, status=500)
    
    async def start_service(self, request: web.Request) -> web.Response:
        """启动服务接口"""
        try:
            with self.service_lock:
                if self.service_status == "running":
                    return web.json_response({
                        "success": False,
                        "message": "服务已在运行中",
                        "data": None
                    }, status=400)
                
                if self.service_status == "starting":
                    return web.json_response({
                        "success": False,
                        "message": "服务正在启动中，请稍候",
                        "data": None
                    }, status=400)
                
                # 设置状态为启动中
                self.service_status = "starting"
                self.stop_time = None
                
                logger.info("开始启动服务...")
                
                # 先杀掉可能存在的旧进程
                self._kill_service_processes()
                
                # 启动新进程
                if self._start_service_process():
                    # 等待一下确保进程启动
                    await asyncio.sleep(2)
                    
                    # 再次检查状态
                    self._check_service_status()
                    
                    if self.service_status == "running":
                        logger.info("服务启动成功")
                        return web.json_response({
                            "success": True,
                            "message": "服务启动成功",
                            "data": {
                                "status": self.service_status,
                                "pid": self.service_pid,
                                "start_time": self.start_time
                            }
                        })
                    else:
                        self.service_status = "stopped"
                        return web.json_response({
                            "success": False,
                            "message": "服务启动失败，请检查日志",
                            "data": None
                        }, status=500)
                else:
                    self.service_status = "stopped"
                    return web.json_response({
                        "success": False,
                        "message": "服务启动失败",
                        "data": None
                    }, status=500)
                
        except Exception as e:
            with self.service_lock:
                self.service_status = "stopped"
            logger.error(f"启动服务失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"启动服务失败: {str(e)}",
                "data": None
            }, status=500)
    
    async def stop_service(self, request: web.Request) -> web.Response:
        """停止服务接口"""
        try:
            with self.service_lock:
                if self.service_status == "stopped":
                    return web.json_response({
                        "success": False,
                        "message": "服务已停止",
                        "data": None
                    }, status=400)
                
                if self.service_status == "stopping":
                    return web.json_response({
                        "success": False,
                        "message": "服务正在停止中，请稍候",
                        "data": None
                    }, status=400)
                
                # 设置状态为停止中
                self.service_status = "stopping"
                self.stop_time = time.time()
                
                logger.info("开始停止服务...")
                
                # 杀掉服务进程
                if self._kill_service_processes():
                    # 等待进程完全退出
                    await asyncio.sleep(2)
                    
                    # 再次检查状态
                    self._check_service_status()
                    
                    if self.service_status == "stopped":
                        logger.info("服务停止成功")
                        return web.json_response({
                            "success": True,
                            "message": "服务停止成功",
                            "data": {
                                "status": self.service_status,
                                "stop_time": self.stop_time
                            }
                        })
                    else:
                        # 如果进程仍在运行，强制停止
                        self._kill_service_processes()
                        self.service_status = "stopped"
                        return web.json_response({
                            "success": True,
                            "message": "服务已强制停止",
                            "data": {
                                "status": self.service_status,
                                "stop_time": self.stop_time
                            }
                        })
                else:
                    self.service_status = "running"  # 恢复为运行状态
                    return web.json_response({
                        "success": False,
                        "message": "停止服务失败",
                        "data": None
                    }, status=500)
                
        except Exception as e:
            with self.service_lock:
                self.service_status = "running"  # 恢复为运行状态
            logger.error(f"停止服务失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"停止服务失败: {str(e)}",
                "data": None
            }, status=500) 