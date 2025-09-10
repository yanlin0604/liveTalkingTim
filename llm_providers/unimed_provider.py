import time
import json
import requests
from basereal import BaseReal
from logger import logger
from .base_provider import BaseLLMProvider

# Unimed 会话缓存：sessionid -> {chatId, tokenValue} 映射
_unimed_chat_cache = {}


class UnimedProvider(BaseLLMProvider):
    """Unimed知识库提供商"""
    
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """Unimed知识库响应处理"""
        # 获取Unimed配置
        ollama_host = getattr(nerfreal.opt, "ollama_host", "http://localhost:8080")
        llm_api_key = getattr(nerfreal.opt, "llm_api_key", "")
        
        if not llm_api_key:
            self.llm_logger.error("Unimed API Key未配置")
            nerfreal.put_msg_txt("抱歉，Unimed服务配置不完整。")
            return
        
        # 获取 sessionid 用于会话缓存，确保为字符串类型
        sessionid = str(getattr(nerfreal, 'sessionid', 'default'))
        self.llm_logger.info("使用Unimed host=%s sessionid=%s msg_len=%d", ollama_host, sessionid, len(message or ""))
        
        try:
            # 获取或创建会话（基于 sessionid 缓存）
            chat_info = self._get_or_create_chat(ollama_host, llm_api_key, sessionid)
            if not chat_info:
                nerfreal.put_msg_txt("抱歉，无法创建Unimed会话。")
                return
            
            end = time.perf_counter()
            logger.info(f"llm Time init (unimed): {end-start_time}s")
            
            # 进行对话
            self._chat_message(ollama_host, chat_info['tokenValue'], chat_info['chatId'], message, nerfreal, start_time)
            
        except Exception as e:
            logger.error(f"Unimed request failed: {e}")
            self.llm_logger.exception("Unimed 请求失败: %s", e)
            nerfreal.put_msg_txt("抱歉，Unimed服务暂时不可用。")

    def _get_or_create_chat(self, host: str, api_key: str, sessionid: str) -> dict:
        """获取或创建Unimed会话，基于sessionid缓存复用"""
        global _unimed_chat_cache
        
        # 检查缓存中是否已有该 sessionid 的会话
        if sessionid in _unimed_chat_cache:
            self.llm_logger.debug("复用已缓存的Unimed会话 sessionid=%s", sessionid)
            return _unimed_chat_cache[sessionid]
        
        # 缓存中没有，创建新会话
        chat_info = self._create_new_chat(host, api_key)
        if chat_info:
            _unimed_chat_cache[sessionid] = chat_info
            self.llm_logger.info("新建Unimed会话并缓存 sessionid=%s", sessionid)
        
        return chat_info

    def _create_new_chat(self, host: str, api_key: str) -> dict:
        """创建新的Unimed会话，返回包含chatId和tokenValue的字典"""
        url = f"{host.rstrip('/')}/application/chat/api/open"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") == 200:
                chat_data = data.get("data", {})
                chat_info = {
                    "chatId": chat_data.get("chatId"),
                    "tokenValue": chat_data.get("tokenValue")
                }
                self.llm_logger.info("Unimed会话创建成功")
                return chat_info
            else:
                self.llm_logger.error("Unimed会话创建失败 code=%s message=%s", data.get("code"), data.get("message"))
                return None
                
        except Exception as e:
            self.llm_logger.exception("Unimed会话创建异常: %s", e)
            return None

    def _chat_message(self, host: str, token_value: str, chat_id: str, message: str, nerfreal: BaseReal, start_time: float):
        """Unimed对话消息处理"""
        url = f"{host.rstrip('/')}/application/chat_message/{chat_id}"
        headers = {
            "Authorization": f"Bearer {token_value}",
            "Content-Type": "application/json"
        }
        
        # 构建Unimed API要求的特殊格式
        payload = {
            "message": message,
            "re_chat": False,
            "form_data": {}
        }
        
        # 添加字符串索引映射（根据API要求）
        for i, char in enumerate(message):
            payload[str(i)] = char
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            # 设置响应编码为UTF-8
            response.encoding = 'utf-8'
            
            self._process_unimed_stream_response(response, nerfreal, start_time)
            
        except Exception as e:
            self.llm_logger.exception("Unimed对话请求异常: %s", e)
            raise

    def _process_unimed_stream_response(self, response, nerfreal: BaseReal, start_time: float):
        """处理Unimed的流式响应"""
        result = ""
        first = True
        total_chars = 0
        
        try:
            current_data = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    # 空行表示一个SSE事件结束，处理当前数据块
                    if current_data:
                        data, result, total_chars = self._process_json_data(current_data, nerfreal, start_time, first, result, total_chars)
                        if first:
                            first = False
                        if data and data.get("end", False):
                            break
                        current_data = ""
                    continue
                
                # 跳过事件类型行
                if line.startswith("event:"):
                    continue
                
                # 处理数据行
                if line.startswith("data:"):
                    current_data = line[5:]  # 去掉 "data:" 前缀
                else:
                    current_data += line
            
            # 处理最后一个数据块
            if current_data:
                self._process_json_data(current_data, nerfreal, start_time, first, result, total_chars)
                    
        except Exception as e:
            self.llm_logger.exception("处理Unimed流式响应异常: %s", e)
        
        end = time.perf_counter()
        logger.info(f"LLM响应总时间: {end-start_time:.2f}s")
        
        if result:
            nerfreal.put_msg_txt(result)

    def _process_json_data(self, json_str: str, nerfreal: BaseReal, start_time: float, is_first: bool, result: str, total_chars: int) -> tuple:
        """处理JSON数据块，返回(data, result, total_chars)"""
        try:
            data = json.loads(json_str)
            
            if is_first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
            
            content = data.get("content", "")
            if content:
                result = self._process_message_chunk(content, result, nerfreal)
                total_chars += len(content)
            
            return data, result, total_chars
            
        except json.JSONDecodeError as e:
            self.llm_logger.warning("JSON解析失败: %s", e)
            return None, result, total_chars

    @staticmethod
    def clear_chat_cache(sessionid=None):
        """清除Unimed会话缓存
        
        Args:
            sessionid: 指定要清除的sessionid，为None时清除所有缓存
        """
        global _unimed_chat_cache
        
        if sessionid is None:
            _unimed_chat_cache.clear()
            logger.info("已清除所有Unimed会话缓存")
        else:
            if sessionid in _unimed_chat_cache:
                del _unimed_chat_cache[sessionid]
                logger.info("已清除Unimed会话缓存 sessionid=%s", sessionid)
