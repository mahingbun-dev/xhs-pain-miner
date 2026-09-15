"""LLM 协议适配层。

对外暴露统一入口 :func:`build_provider`，以及协议无关的消息 / 响应类型。
"""

from xhs_pain_miner.llm.base import (
    BaseLLMProvider,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    build_data_url,
    extract_json,
    split_data_url,
)
from xhs_pain_miner.llm.factory import build_provider, describe_protocol

__all__ = [
    "BaseLLMProvider",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "build_data_url",
    "build_provider",
    "describe_protocol",
    "extract_json",
    "split_data_url",
]
