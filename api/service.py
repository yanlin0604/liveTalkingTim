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
    """主数字人服务管理API类"""
    
    def __init__(self):
        self.service_status = "stopped"  # running, stopped, starting, stopping
        self.service_process = None
        self.service_pid = None
        self.service_lock = threading.Lock()
        self.start_time = None
        self.stop_time = None
        
        # 服务配置 - 专门管理主数字人服务
        self.service_script = "start_main_service.sh"  # 使用专门的主服务启动脚本
        self.stop_script = "stop_main_service.sh"      # 使用专门的主服务停止脚本
        self.target_processes = ["app.py"]  # 只管理主数字人服务进程（app.py）
        self.log_dir = "/mnt/disk1/ftp/data/60397193/logs"  # 日志目录
        self.conda_env = "nerfstream"  # conda环境
        self.script_dir = "/mnt/disk1/ftp/file/60397193/LiveTalking"  # 脚本目录
        self.main_port = 8010  # 主服务端口
        
        # 初始化时检查服务状态
        self._check_service_status()
    
    def _check_service_status(self):
        """检查当前服务状态"""
        try:
            logger.info("检查主服务状态...")
            
            # 简化检查：只通过PID文件检查
            pid_file = f"{self.log_dir}/main.pid"
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        main_pid = int(f.read().strip())
                    
                    if psutil.pid_exists(main_pid):
                        proc = psutil.Process(main_pid)
                        cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                        
                        # 确认是主服务进程
                        if proc.cmdline() and 'app.py' in cmdline and 'management_server.py' not in cmdline:
                            with self.service_lock:
                                self.service_status = "running"
                                self.service_pid = main_pid
                                try:
                                    self.start_time = proc.create_time()
                                except Exception:
                                    self.start_time = time.time()
                            logger.info(f"主服务运行中，PID: {main_pid}")
                            return
                except Exception as e:
                    logger.warning(f"读取PID文件失败: {e}")
            
            # 如果PID文件不存在或进程不存在，则认为服务已停止
            with self.service_lock:
                self.service_status = "stopped"
                self.service_pid = None
                self.stop_time = time.time()
            logger.info("主服务已停止")
            
        except Exception as e:
            logger.error(f"检查服务状态失败: {e}")
            with self.service_lock:
                    self.service_status = "stopped"
                    self.service_pid = None
                    self.start_time = None
                    logger.info("检测到主服务已停止，状态: stopped")
                    
        except Exception as e:
            logger.error(f"检查服务状态失败: {e}")
            self.service_status = "stopped"
    
    def _kill_service_processes(self):
        """停止主数字人服务进程"""
        try:
            logger.info("开始执行停止主服务进程操作...")
            
            # 使用专门的停止脚本
            cmd = ["bash", self.stop_script]
            logger.info(f"停止命令: {' '.join(cmd)}")
            
            # 异步执行停止脚本，避免阻塞当前管理服务
            logger.info("开始异步执行停止脚本...")
            process = subprocess.Popen(
                cmd,
                cwd=self.script_dir,  # 设置工作目录
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"停止脚本已开始执行，进程ID: {process.pid}")
            logger.info("停止脚本正在后台运行，管理服务继续运行")
            
            # 返回成功，让调用方知道脚本已开始执行
            return True
                
        except Exception as e:
            logger.error(f"停止主数字人服务进程失败: {e}")
            return False
    
    def _start_service_process(self):
        """启动主数字人服务进程"""
        try:
            logger.info("开始启动主数字人服务进程...")
            logger.info(f"当前工作目录: {os.getcwd()}")
            logger.info(f"脚本目录: {self.script_dir}")
            logger.info(f"日志目录: {self.log_dir}")
            
            # 确保日志目录存在
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info("日志目录已确保存在")
            
            # 使用专门的启动脚本
            cmd = ["bash", self.service_script]
            logger.info(f"启动命令: {' '.join(cmd)}")
            
            # 异步启动进程，避免阻塞当前管理服务
            logger.info("开始异步执行启动脚本...")
            process = subprocess.Popen(
                cmd,
                cwd=self.script_dir,  # 设置工作目录
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"启动脚本已开始执行，进程ID: {process.pid}")
            logger.info("启动脚本正在后台运行，管理服务继续运行")
            
            # 记录启动时间
            self.start_time = time.time()
            
            # 返回成功，让调用方知道脚本已开始执行
            return True
            
        except Exception as e:
            logger.error(f"启动主数字人服务进程失败: {e}")
            return False
    
    async def get_status(self, request: web.Request) -> web.Response:
        """查询主数字人服务状态接口"""
        try:
            # 实时检查服务状态
            self._check_service_status()
            
            with self.service_lock:
                status_info = {
                    "service_type": "main_digital_human",  # 明确标识这是主数字人服务
                    "status": self.service_status,
                    "pid": self.service_pid,
                    "start_time": self.start_time,
                    "stop_time": self.stop_time,
                    "uptime": None,
                    "port": self.main_port,  # 主服务端口
                    "endpoints": {
                        "webrtc": f"http://localhost:{self.main_port}/offer",
                        "chat": f"http://localhost:{self.main_port}/human",
                        "audio": f"http://localhost:{self.main_port}/humanaudio",
                        "dashboard": f"http://localhost:{self.main_port}/dashboard.html",
                        "config_manager": f"http://localhost:{self.main_port}/config_manager.html"
                    }
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
                
                # 检查端口监听状态
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(('localhost', self.main_port))
                    sock.close()
                    status_info["port_listening"] = (result == 0)
                except:
                    status_info["port_listening"] = False
                
                logger.info(f"查询主数字人服务状态: {self.service_status}, PID: {self.service_pid}")
                return web.json_response({
                    "success": True,
                    "message": "查询主数字人服务状态成功",
                    "data": status_info
                })
                
        except Exception as e:
            logger.error(f"查询主数字人服务状态失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"查询主数字人服务状态失败: {str(e)}",
                "data": None
            }, status=500)
    
    async def start_service(self, request: web.Request) -> web.Response:
        """启动主数字人服务接口"""
        try:
            logger.info("收到启动主数字人服务请求")
            with self.service_lock:
                logger.info(f"当前服务状态: {self.service_status}")
                if self.service_status == "running":
                    logger.info("主数字人服务已在运行中，拒绝启动请求")
                    return web.json_response({
                        "success": False,
                        "message": "主数字人服务已在运行中",
                        "data": None
                    }, status=400)
                
                if self.service_status == "starting":
                    logger.info("主数字人服务正在启动中，拒绝重复启动请求")
                    return web.json_response({
                        "success": False,
                        "message": "主数字人服务正在启动中，请稍候",
                        "data": None
                    }, status=400)
                
                # 设置状态为启动中
                self.service_status = "starting"
                self.stop_time = None
                
                logger.info("设置服务状态为启动中，开始启动主数字人服务...")
                
                # 先停止可能存在的旧进程
                logger.info("开始停止可能存在的旧进程...")
                self._kill_service_processes()
                
                # 启动新进程
                logger.info("开始启动新进程...")
                if self._start_service_process():
                    # 脚本已开始执行，直接设置为运行状态
                    self.service_status = "running"
                    logger.info("启动脚本已执行，主数字人服务启动中")
                    return web.json_response({
                        "success": True,
                        "message": "主数字人服务启动成功",
                        "data": {
                            "service_type": "main_digital_human",
                            "status": self.service_status,
                            "start_time": self.start_time,
                            "port": self.main_port,
                            "note": "启动脚本已执行，管理服务器继续运行"
                        }
                    })
                else:
                    logger.error("启动脚本执行失败")
                    self.service_status = "stopped"
                    return web.json_response({
                        "success": False,
                        "message": "主数字人服务启动失败",
                        "data": None
                    }, status=500)
                
        except Exception as e:
            with self.service_lock:
                self.service_status = "stopped"
            logger.error(f"启动主数字人服务失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"启动主数字人服务失败: {str(e)}",
                "data": None
            }, status=500)
    
    async def stop_service(self, request: web.Request) -> web.Response:
        """停止主数字人服务接口"""
        try:
            logger.info("收到停止主数字人服务请求")
            with self.service_lock:
                logger.info(f"当前服务状态: {self.service_status}")
                if self.service_status == "stopped":
                    logger.info("主数字人服务已停止，拒绝停止请求")
                    return web.json_response({
                        "success": False,
                        "message": "主数字人服务已停止",
                        "data": None
                    }, status=400)
                
                if self.service_status == "stopping":
                    logger.info("主数字人服务正在停止中，拒绝重复停止请求")
                    return web.json_response({
                        "success": False,
                        "message": "主数字人服务正在停止中，请稍候",
                        "data": None
                    }, status=400)
                
                # 设置状态为停止中
                self.service_status = "stopping"
                self.stop_time = time.time()
                
                logger.info("设置服务状态为停止中，开始停止主数字人服务...")
                
                # 停止服务进程
                logger.info("开始停止服务进程...")
                if self._kill_service_processes():
                    # 脚本已开始执行，直接设置为已停止状态
                    self.service_status = "stopped"
                    logger.info("停止脚本已执行，主数字人服务停止中")
                    return web.json_response({
                        "success": True,
                        "message": "主数字人服务停止成功",
                        "data": {
                            "service_type": "main_digital_human",
                            "status": self.service_status,
                            "stop_time": self.stop_time,
                            "note": "停止脚本已执行，管理服务器继续运行"
                        }
                    })
                else:
                    logger.error("停止脚本执行失败")
                    self.service_status = "running"  # 恢复为运行状态
                    return web.json_response({
                        "success": False,
                        "message": "停止主数字人服务失败",
                        "data": None
                    }, status=500)
                
        except Exception as e:
            with self.service_lock:
                self.service_status = "running"  # 恢复为运行状态
            logger.error(f"停止主数字人服务失败: {e}")
            return web.json_response({
                "success": False,
                "message": f"停止主数字人服务失败: {str(e)}",
                "data": None
            }, status=500) 