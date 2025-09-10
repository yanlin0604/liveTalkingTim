import time
import json
import requests
from basereal import BaseReal
from logger import logger
from .base_provider import BaseLLMProvider

# MaxKB 会话缓存：sessionid -> chat_id 映射
_maxkb_chat_cache = {}


class MaxKBProvider(BaseLLMProvider):
    """MaxKB知识库提供商"""
    
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """MaxKB知识库响应处理"""
        # 获取MaxKB配置
        ollama_host = getattr(nerfreal.opt, "ollama_host", "http://localhost:8080")
        llm_api_key = getattr(nerfreal.opt, "llm_api_key", "")
        
        if not llm_api_key:
            logger.error("MaxKB API Key未配置")
            self.llm_logger.error("MaxKB API Key未配置")
            nerfreal.put_msg_txt("抱歉，MaxKB服务配置不完整。")
            return
        
        # 获取 sessionid 用于会话缓存，确保为字符串类型
        sessionid = str(getattr(nerfreal, 'sessionid', 'default'))
        self.llm_logger.info("使用MaxKB host=%s api_key=%s sessionid=%s msg_len=%d", ollama_host, llm_api_key[:10] + "...", sessionid, len(message or ""))
        
        try:
            # 第一步：获取或创建会话（基于 sessionid 缓存）
            chat_id = self._get_or_create_chat(ollama_host, llm_api_key, sessionid)
            if not chat_id:
                nerfreal.put_msg_txt("抱歉，无法创建MaxKB会话。")
                return
            
            end = time.perf_counter()
            logger.info(f"llm Time init (maxkb): {end-start_time}s")
            self.llm_logger.debug("maxkb 会话获取耗时=%.3fs chat_id=%s sessionid=%s", end - start_time, chat_id, sessionid)
            
            # 第二步：进行对话
            self._chat_message(ollama_host, llm_api_key, chat_id, message, nerfreal, start_time)
            
        except Exception as e:
            logger.error(f"MaxKB request failed: {e}")
            self.llm_logger.exception("MaxKB 请求失败: %s", e)
            nerfreal.put_msg_txt("抱歉，MaxKB服务暂时不可用。")

    def _get_or_create_chat(self, host: str, api_key: str, sessionid: str) -> str:
        """获取或创建MaxKB会话，基于sessionid缓存复用"""
        global _maxkb_chat_cache
        
        # 检查缓存中是否已有该 sessionid 的会话
        if sessionid in _maxkb_chat_cache:
            cached_chat_id = _maxkb_chat_cache[sessionid]
            self.llm_logger.info("复用已缓存的MaxKB会话 sessionid=%s chat_id=%s", sessionid, cached_chat_id)
            return cached_chat_id
        
        # 缓存中没有，创建新会话
        chat_id = self._create_new_chat(host, api_key)
        if chat_id:
            _maxkb_chat_cache[sessionid] = chat_id
            self.llm_logger.info("新建MaxKB会话并缓存 sessionid=%s chat_id=%s", sessionid, chat_id)
        
        return chat_id

    def _create_new_chat(self, host: str, api_key: str) -> str:
        """创建新的MaxKB会话，返回chat_id"""
        url = f"{host.rstrip('/')}/chat/api/open"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") == 200:
                chat_id = data.get("data")
                self.llm_logger.info("MaxKB会话创建成功 chat_id=%s", chat_id)
                return chat_id
            else:
                self.llm_logger.error("MaxKB会话创建失败 code=%s message=%s", data.get("code"), data.get("message"))
                return None
                
        except Exception as e:
            self.llm_logger.exception("MaxKB会话创建异常: %s", e)
            return None

    def _chat_message(self, host: str, api_key: str, chat_id: str, message: str, nerfreal: BaseReal, start_time: float):
        """MaxKB对话消息处理"""
        url = f"{host.rstrip('/')}/chat/api/chat_message/{chat_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": message,
            "stream": True,
            "re_chat": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            self.llm_logger.info("maxkb 流式响应开始")
            self._process_maxkb_stream_response(response, nerfreal, start_time)
            
        except Exception as e:
            self.llm_logger.exception("MaxKB对话请求异常: %s", e)
            raise

    def _process_maxkb_stream_response(self, response, nerfreal: BaseReal, start_time: float):
        """处理MaxKB的流式响应"""
        result = ""
        first = True
        total_chars = 0
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                    
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start_time}s")
                    self.llm_logger.info("首包到达耗时=%.3fs", end - start_time)
                    first = False
                
                try:
                    # 解析流式数据
                    json_str = line[6:]  # 去掉 "data: " 前缀
                    data = json.loads(json_str)
                    
                    content = data.get("content", "")
                    if content:
                        self.llm_logger.debug("收到delta长度=%d 预览='%s'", len(content), content[:120])
                        result = self._process_message_chunk(content, result, nerfreal)
                        total_chars += len(content)
                    
                    # 检查是否结束
                    if data.get("is_end", False):
                        self.llm_logger.info("MaxKB响应结束标志")
                        break
                        
                except json.JSONDecodeError as e:
                    self.llm_logger.warning("解析MaxKB流式数据失败: %s, line: %s", e, line[:200])
                    continue
                    
        except Exception as e:
            self.llm_logger.exception("处理MaxKB流式响应异常: %s", e)
        
        end = time.perf_counter()
        logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
        self.llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
        
        if result:
            logger.info(f" 发送最终TTS文本: '{result}'")
            self.llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(result), result[:200])
            nerfreal.put_msg_txt(result)
        
        logger.info(" === LLM响应完成 ===")
        self.llm_logger.info("=== LLM响应完成（maxkb）===")

    @staticmethod
    def clear_chat_cache(sessionid=None):
        """清除MaxKB会话缓存
        
        Args:
            sessionid: 指定要清除的sessionid，为None时清除所有缓存
        """
        global _maxkb_chat_cache
        
        if sessionid is None:
            # 清除所有缓存
            cleared_count = len(_maxkb_chat_cache)
            _maxkb_chat_cache.clear()
            logger.info("已清除所有MaxKB会话缓存，共%d个", cleared_count)
        else:
            # 清除指定sessionid的缓存
            if sessionid in _maxkb_chat_cache:
                del _maxkb_chat_cache[sessionid]
                logger.info("已清除MaxKB会话缓存 sessionid=%s", sessionid)
            else:
                logger.warning("指定的sessionid不在缓存中 sessionid=%s", sessionid)
