"""
LLM提供商模块

支持的提供商：
- DashScope: 阿里云DashScope
- Ollama: 本地Ollama模型
- MaxKB: MaxKB知识库
- Unimed: Unimed知识库
"""

from .base_provider import BaseLLMProvider
from .dashscope_provider import DashScopeProvider
from .ollama_provider import OllamaProvider
from .maxkb_provider import MaxKBProvider
from .unimed_provider import UnimedProvider

# 提供商映射
PROVIDERS = {
    "dashscope": DashScopeProvider,
    "ollama": OllamaProvider,
    "maxkb": MaxKBProvider,
    "unimed": UnimedProvider,
}

__all__ = [
    "BaseLLMProvider",
    "DashScopeProvider", 
    "OllamaProvider",
    "MaxKBProvider",
    "UnimedProvider",
    "PROVIDERS",
]
