import time
import json
import requests
from basereal import BaseReal
from logger import logger
from .base_provider import BaseLLMProvider

# Dify 会话缓存：sessionid -> conversation_id 映射
_dify_chat_cache = {}


class DifyProvider(BaseLLMProvider):
    """Dify提供商"""
    
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """Dify响应处理"""
        # 获取Dify配置
        ollama_host = getattr(nerfreal.opt, "ollama_host", "http://127.0.0.1")
        llm_api_key = getattr(nerfreal.opt, "llm_api_key", "")
        
        if not llm_api_key:
            self.llm_logger.error("Dify API Key未配置")
            nerfreal.put_msg_txt("抱歉，Dify服务配置不完整。")
            return
        
        # 获取 sessionid 用于会话缓存，确保为字符串类型
        sessionid = str(getattr(nerfreal, 'sessionid', 'default'))
        self.llm_logger.info("使用Dify host=%s sessionid=%s msg_len=%d", ollama_host, sessionid, len(message or ""))
        
        try:
            # 获取或创建会话ID（基于 sessionid 缓存）
            conversation_id = self._get_or_create_conversation(sessionid)
            
            end = time.perf_counter()
            logger.info(f"llm Time init (dify): {end-start_time}s")
            
            # 进行对话
            self._chat_message(ollama_host, llm_api_key, conversation_id, message, nerfreal, start_time)
            
        except Exception as e:
            logger.error(f"Dify request failed: {e}")
            self.llm_logger.exception("Dify 请求失败: %s", e)
            nerfreal.put_msg_txt("抱歉，Dify服务暂时不可用。")

    def _get_or_create_conversation(self, sessionid: str) -> str:
        """获取或创建Dify会话ID，基于sessionid缓存复用"""
        global _dify_chat_cache
        
        # 检查缓存中是否已有该 sessionid 的会话
        if sessionid in _dify_chat_cache:
            conversation_id = _dify_chat_cache[sessionid]
            self.llm_logger.debug("复用已缓存的Dify会话 sessionid=%s conversation_id=%s", sessionid, conversation_id)
            return conversation_id
        
        # 缓存中没有，使用空字符串作为新会话（Dify会自动生成新的conversation_id）
        conversation_id = ""
        _dify_chat_cache[sessionid] = conversation_id
        self.llm_logger.info("新建Dify会话 sessionid=%s", sessionid)
        
        return conversation_id

    def _chat_message(self, host: str, api_key: str, conversation_id: str, message: str, nerfreal: BaseReal, start_time: float):
        """Dify对话消息处理"""
        url = f"{host.rstrip('/')}/v1/chat-messages"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建Dify API要求的格式
        payload = {
            "inputs": {},
            "query": message,
            "response_mode": "streaming",
            "conversation_id": conversation_id,
            "user": "abc-123",
            "files": []
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            # 设置响应编码为UTF-8
            response.encoding = 'utf-8'
            
            self._process_dify_stream_response(response, nerfreal, start_time)
            
        except Exception as e:
            self.llm_logger.exception("Dify对话请求异常: %s", e)
            raise

    def _process_dify_stream_response(self, response, nerfreal: BaseReal, start_time: float):
        """处理Dify的流式响应"""
        result = ""
        first = True
        total_chars = 0
        current_conversation_id = None
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                
                # 处理data:开头的行
                if line.startswith("data:"):
                    json_str = line[5:].strip()  # 去掉 "data:" 前缀
                    
                    if json_str:
                        data, result, total_chars, conv_id = self._process_json_data(
                            json_str, nerfreal, start_time, first, result, total_chars
                        )
                        
                        if first:
                            first = False
                        
                        # 更新会话ID到缓存
                        if conv_id and not current_conversation_id:
                            current_conversation_id = conv_id
                            self._update_conversation_cache(nerfreal, conv_id)
                        
                        # 检查是否结束
                        if data and data.get("event") == "message_end":
                            break
                    
        except Exception as e:
            self.llm_logger.exception("处理Dify流式响应异常: %s", e)
        
        end = time.perf_counter()
        logger.info(f"LLM响应总时间: {end-start_time:.2f}s")
        
        if result:
            nerfreal.put_msg_txt(result)

    def _process_json_data(self, json_str: str, nerfreal: BaseReal, start_time: float, is_first: bool, result: str, total_chars: int) -> tuple:
        """处理JSON数据块，返回(data, result, total_chars, conversation_id)"""
        try:
            data = json.loads(json_str)
            
            if is_first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
            
            # 只处理message事件
            if data.get("event") == "message":
                content = data.get("answer", "")
                conversation_id = data.get("conversation_id", "")
                
                if content:
                    result = self._process_message_chunk(content, result, nerfreal)
                    total_chars += len(content)
                
                return data, result, total_chars, conversation_id
            
            return data, result, total_chars, data.get("conversation_id", "")
            
        except json.JSONDecodeError as e:
            self.llm_logger.warning("JSON解析失败: %s", e)
            return None, result, total_chars, ""

    def _update_conversation_cache(self, nerfreal: BaseReal, conversation_id: str):
        """更新会话缓存中的conversation_id"""
        global _dify_chat_cache
        sessionid = str(getattr(nerfreal, 'sessionid', 'default'))
        
        if conversation_id and sessionid in _dify_chat_cache:
            _dify_chat_cache[sessionid] = conversation_id
            self.llm_logger.info("更新Dify会话缓存 sessionid=%s conversation_id=%s", sessionid, conversation_id)

    @staticmethod
    def clear_chat_cache(sessionid=None):
        """清除Dify会话缓存
        
        Args:
            sessionid: 指定要清除的sessionid，为None时清除所有缓存
        """
        global _dify_chat_cache
        
        if sessionid is None:
            _dify_chat_cache.clear()
            logger.info("已清除所有Dify会话缓存")
        else:
            if sessionid in _dify_chat_cache:
                del _dify_chat_cache[sessionid]
                logger.info("已清除Dify会话缓存 sessionid=%s", sessionid)
