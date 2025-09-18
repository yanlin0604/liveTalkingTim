import time
import os
import logging
from logging.handlers import RotatingFileHandler
from basereal import BaseReal
from logger import logger
from llm_providers import PROVIDERS

# === LLM 专用日志（独立文件，便于排查是否有发送文本以触发TTS）===
# 日志位置: logs/llm.log，大小轮转
_LLM_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_LLM_LOG_PATH = os.path.join(_LLM_LOG_DIR, "llm.log")


def _setup_llm_logger():
    """初始化 LLM 专用日志记录器（单例）。
    - 文件: logs/llm.log
    - 轮转: 2MB, 保留 5 个历史
    - 编码: utf-8
    """
    ll = logging.getLogger("llm")
    if ll.handlers:
        return ll

    ll.setLevel(logging.DEBUG)
    try:
        os.makedirs(_LLM_LOG_DIR, exist_ok=True)
    except Exception:
        # 目录创建失败不应中断主流程，回退为仅控制台
        pass

    try:
        fh = RotatingFileHandler(_LLM_LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        ll.addHandler(fh)
    except Exception:
        # 文件句柄失败时不抛出，至少保留一个空记录器
        pass

    ll.propagate = False  # 不向上冒泡，避免与全局logger重复
    return ll


llm_logger = _setup_llm_logger()


def llm_response(message, nerfreal: BaseReal):
    """
    LLM响应函数，支持多种LLM提供商
    支持的提供商：dashscope（阿里云）、ollama（本地）、maxkb（MaxKB知识库）、unimed（Unimed知识库）、dify（Dify知识库）
    """
    start = time.perf_counter()
    logger.info(f" === LLM响应开始 ===")
    logger.info(f" 用户消息: '{message[:50]}{'...' if len(message) > 50 else ''}'")
    llm_logger.info("=== LLM响应开始 ===")
    llm_logger.info("用户消息长度=%d 预览='%s'", len(message or ""), (message or "")[:200])

    # 获取LLM配置
    llm_provider = getattr(nerfreal.opt, "llm_provider", "dashscope")
    logger.info(f" LLM提供商: {llm_provider}")
    llm_logger.info("提供商=%s", llm_provider)

    # 获取对应的提供商类
    provider_class = PROVIDERS.get(llm_provider)
    if not provider_class:
        logger.error(f" 未知的LLM提供商: {llm_provider}")
        llm_logger.error("未知的LLM提供商: %s", llm_provider)
        nerfreal.put_msg_txt("抱歉，不支持的LLM提供商。")
        return

    try:
        # 创建提供商实例并获取响应
        provider = provider_class(llm_logger)
        logger.info(f" 使用{llm_provider}提供商")
        llm_logger.info("使用%s提供商", llm_provider)
        provider.get_response(message, nerfreal, start)
    except Exception as e:
        logger.error(f"LLM提供商 {llm_provider} 处理失败: {e}")
        llm_logger.exception("LLM提供商 %s 处理失败: %s", llm_provider, e)
        nerfreal.put_msg_txt("抱歉，LLM服务暂时不可用。")


# 兼容性函数：保持向后兼容
def clear_maxkb_chat_cache(sessionid=None):
    """清除MaxKB会话缓存（兼容性函数）"""
    from llm_providers.maxkb_provider import MaxKBProvider
    MaxKBProvider.clear_chat_cache(sessionid)


def clear_unimed_chat_cache(sessionid=None):
    """清除Unimed会话缓存（兼容性函数）"""
    from llm_providers.unimed_provider import UnimedProvider
    UnimedProvider.clear_chat_cache(sessionid)


def clear_dify_chat_cache(sessionid=None):
    """清除Dify会话缓存（兼容性函数）"""
    from llm_providers.dify_provider import DifyProvider
    DifyProvider.clear_chat_cache(sessionid)
