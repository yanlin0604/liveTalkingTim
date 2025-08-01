"""
鉴权相关API接口
"""
import json
import time
import uuid
import hashlib
import hmac
import ipaddress
from typing import Dict, Optional, Union
from aiohttp import web
from logger import logger


class AuthAPI:
    """鉴权相关API接口类"""
    
    def __init__(self):
        # 存储有效的token和对应的uuid
        self.tokens: Dict[str, Dict] = {}  # token -> {uuid, created_time, expires_time}
        self.token_secret = "Unimed_Token_Secret_2025"  # 用于签名验证的密钥
        self.token_expire_hours = 24  # token有效期（小时）
        
        # IP白名单 - 写死在代码中
        # 支持单个IP地址和IP段（CIDR格式）
        self.ip_whitelist = [
            "127.0.0.1",      # 本地回环地址
            "localhost",      # 本地主机名
            "::1",           # IPv6本地回环地址
            "192.168.1.0/24", # 192.168.1.0 - 192.168.1.255
            "10.0.0.0/16",   # 10.0.0.0 - 10.0.255.255
            "172.16.0.0/12", # 172.16.0.0 - 172.31.255.255
            # 可以在这里添加更多允许的IP地址或IP段
        ]
    
    def generate_token(self, client_uuid: str) -> str:
        """生成token"""
        # 生成随机token
        token = str(uuid.uuid4())
        
        # 计算过期时间
        created_time = time.time()
        expires_time = created_time + (self.token_expire_hours * 3600)
        
        # 存储token信息
        self.tokens[token] = {
            'uuid': client_uuid,
            'created_time': created_time,
            'expires_time': expires_time
        }
        
        logger.info(f"为客户端 {client_uuid} 生成token: {token[:8]}...")
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """验证token，返回对应的uuid，如果无效返回None"""
        if token not in self.tokens:
            return None
        
        token_info = self.tokens[token]
        
        # 检查是否过期
        if time.time() > token_info['expires_time']:
            # 删除过期token
            del self.tokens[token]
            logger.warning(f"Token已过期: {token[:8]}...")
            return None
        
        return token_info['uuid']
    
    def revoke_token(self, token: str) -> bool:
        """撤销token"""
        if token in self.tokens:
            del self.tokens[token]
            logger.info(f"撤销token: {token[:8]}...")
            return True
        return False
    
    def cleanup_expired_tokens(self):
        """清理过期的token"""
        current_time = time.time()
        expired_tokens = []
        
        for token, token_info in self.tokens.items():
            if current_time > token_info['expires_time']:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.tokens[token]
        
        if expired_tokens:
            logger.info(f"清理了 {len(expired_tokens)} 个过期token")
    
    def check_ip_whitelist(self, request) -> bool:
        """检查请求IP是否在白名单中"""
        # 获取客户端IP地址
        client_ip = request.remote
        
        # 检查是否在白名单中（支持单个IP和IP段）
        if self._is_ip_allowed(client_ip):
            logger.info(f"IP {client_ip} 在白名单中，允许访问")
            return True
        
        # 检查X-Forwarded-For头（如果使用代理）
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # 取第一个IP地址
            first_ip = forwarded_for.split(',')[0].strip()
            if self._is_ip_allowed(first_ip):
                logger.info(f"代理IP {first_ip} 在白名单中，允许访问")
                return True
        
        # 检查X-Real-IP头（Nginx代理）
        real_ip = request.headers.get('X-Real-IP')
        if real_ip and self._is_ip_allowed(real_ip):
            logger.info(f"真实IP {real_ip} 在白名单中，允许访问")
            return True
        
        logger.warning(f"IP {client_ip} 不在白名单中，拒绝访问")
        return False
    
    def _is_ip_allowed(self, ip_str: str) -> bool:
        """检查IP是否在白名单中（支持单个IP和IP段）"""
        try:
            # 处理特殊的主机名
            if ip_str in ["localhost", "::1"]:
                return ip_str in self.ip_whitelist
            
            # 解析客户端IP
            client_ip = ipaddress.ip_address(ip_str)
            
            # 检查白名单中的每个条目
            for whitelist_entry in self.ip_whitelist:
                try:
                    # 如果是CIDR格式（IP段）
                    if '/' in whitelist_entry:
                        network = ipaddress.ip_network(whitelist_entry, strict=False)
                        if client_ip in network:
                            return True
                    # 如果是单个IP地址
                    else:
                        # 处理特殊的主机名
                        if whitelist_entry in ["localhost", "::1"]:
                            continue
                        
                        whitelist_ip = ipaddress.ip_address(whitelist_entry)
                        if client_ip == whitelist_ip:
                            return True
                            
                except ValueError:
                    # 如果白名单条目格式无效，跳过
                    logger.warning(f"无效的白名单条目: {whitelist_entry}")
                    continue
            
            return False
            
        except ValueError as e:
            logger.warning(f"无效的IP地址格式: {ip_str}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"检查IP白名单时发生错误: {e}")
            return False
    
    async def get_token_api(self, request):
        """
        获取访问token
        
        ---
        tags:
          - Authentication
        summary: 获取访问token
        description: 根据客户端UUID获取访问token，用于后续API调用（需要IP白名单授权）
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                uuid:
                  type: string
                  description: 客户端唯一标识符
                  required: true
        responses:
          200:
            description: 获取成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                token:
                  type: string
                  description: 访问token
                expires_in:
                  type: integer
                  description: token有效期（秒）
                message:
                  type: string
                  description: 状态消息
          400:
            description: 请求参数错误
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
          403:
            description: IP地址未授权
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        try:
            # 检查IP白名单
            if not self.check_ip_whitelist(request):
                return web.json_response({
                    "success": False,
                    "error": "IP地址未授权",
                    "message": "您的IP地址不在白名单中，无法访问此接口"
                }, status=403)
            
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.json_response({
                    "success": False,
                    "error": "请求体为空",
                    "message": "请提供有效的JSON数据"
                }, status=400)
            
            data = await request.json()
            client_uuid = data.get('uuid')
            
            # 参数验证
            if not client_uuid:
                return web.json_response({
                    "success": False,
                    "error": "缺少必要参数",
                    "message": "请提供 uuid 参数"
                }, status=400)
            
            # 验证UUID格式
            try:
                uuid.UUID(client_uuid)
            except ValueError:
                return web.json_response({
                    "success": False,
                    "error": "无效的UUID格式",
                    "message": "请提供有效的UUID格式"
                }, status=400)
            
            # 清理过期token
            self.cleanup_expired_tokens()
            
            # 生成token
            token = self.generate_token(client_uuid)
            expires_in = self.token_expire_hours * 3600
            
            return web.json_response({
                "success": True,
                "token": token,
                "expires_in": expires_in,
                "message": "token获取成功"
            })
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return web.json_response({
                "success": False,
                "error": "JSON格式错误",
                "message": f"无效的JSON格式: {str(e)}"
            }, status=400)
        except Exception as e:
            logger.error(f"获取token失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "获取token时发生错误"
            }, status=500)
    
    async def revoke_token_api(self, request):
        """
        撤销访问token
        
        ---
        tags:
          - Authentication
        summary: 撤销访问token
        description: 撤销指定的访问token
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                token:
                  type: string
                  description: 要撤销的token
                  required: true
        responses:
          200:
            description: 撤销成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                message:
                  type: string
                  description: 状态消息
          400:
            description: 请求参数错误
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.json_response({
                    "success": False,
                    "error": "请求体为空",
                    "message": "请提供有效的JSON数据"
                }, status=400)
            
            data = await request.json()
            token = data.get('token')
            
            # 参数验证
            if not token:
                return web.json_response({
                    "success": False,
                    "error": "缺少必要参数",
                    "message": "请提供 token 参数"
                }, status=400)
            
            # 撤销token
            if self.revoke_token(token):
                return web.json_response({
                    "success": True,
                    "message": "token撤销成功"
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "token不存在",
                    "message": "指定的token不存在或已过期"
                }, status=404)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return web.json_response({
                "success": False,
                "error": "JSON格式错误",
                "message": f"无效的JSON格式: {str(e)}"
            }, status=400)
        except Exception as e:
            logger.error(f"撤销token失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "撤销token时发生错误"
            }, status=500)
    
    async def verify_token_api(self, request):
        """
        验证访问token
        
        ---
        tags:
          - Authentication
        summary: 验证访问token
        description: 验证指定的访问token是否有效
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                token:
                  type: string
                  description: 要验证的token
                  required: true
        responses:
          200:
            description: 验证成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                valid:
                  type: boolean
                  description: token是否有效
                uuid:
                  type: string
                  description: 客户端UUID（仅当token有效时）
                expires_in:
                  type: integer
                  description: 剩余有效期（秒，仅当token有效时）
                message:
                  type: string
                  description: 状态消息
          400:
            description: 请求参数错误
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.json_response({
                    "success": False,
                    "error": "请求体为空",
                    "message": "请提供有效的JSON数据"
                }, status=400)
            
            data = await request.json()
            token = data.get('token')
            
            # 参数验证
            if not token:
                return web.json_response({
                    "success": False,
                    "error": "缺少必要参数",
                    "message": "请提供 token 参数"
                }, status=400)
            
            # 验证token
            client_uuid = self.verify_token(token)
            
            if client_uuid:
                # token有效
                token_info = self.tokens[token]
                expires_in = max(0, int(token_info['expires_time'] - time.time()))
                
                return web.json_response({
                    "success": True,
                    "valid": True,
                    "uuid": client_uuid,
                    "expires_in": expires_in,
                    "message": "token有效"
                })
            else:
                # token无效
                return web.json_response({
                    "success": True,
                    "valid": False,
                    "message": "token无效或已过期"
                })
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return web.json_response({
                "success": False,
                "error": "JSON格式错误",
                "message": f"无效的JSON格式: {str(e)}"
            }, status=400)
        except Exception as e:
            logger.error(f"验证token失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "验证token时发生错误"
            }, status=500)


def require_auth(auth_api: AuthAPI):
    """装饰器：要求token验证"""
    def decorator(func):
        async def wrapper(request):
            # 从请求头获取token
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return web.json_response({
                    "success": False,
                    "error": "缺少认证信息",
                    "message": "请在请求头中提供 Authorization: Bearer <token>"
                }, status=401)
            
            # 检查Authorization格式
            if not auth_header.startswith('Bearer '):
                return web.json_response({
                    "success": False,
                    "error": "认证格式错误",
                    "message": "Authorization格式应为: Bearer <token>"
                }, status=401)
            
            # 提取token
            token = auth_header[7:]  # 去掉 "Bearer " 前缀
            
            # 验证token
            client_uuid = auth_api.verify_token(token)
            if not client_uuid:
                return web.json_response({
                    "success": False,
                    "error": "token无效",
                    "message": "token无效或已过期，请重新获取"
                }, status=401)
            
            # 将client_uuid添加到请求中，供后续使用
            request['client_uuid'] = client_uuid
            
            # 调用原始函数
            return await func(request)
        
        return wrapper
    return decorator 