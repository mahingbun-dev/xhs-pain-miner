"""按配置构造 LLM / VLM 供应商实例。

**延迟导入**：``openai`` 与 ``anthropic`` 的导入各需数百毫秒，因此只在真正构造
供应商时才导入，避免 ``xhs-pain-miner doctor`` 这类轻量命令被拖慢。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from xhs_pain_miner.llm.base import BaseLLMProvider

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.config import Settings

Purpose = Literal["text", "vision"]


def _provider_class(protocol: str) -> type[BaseLLMProvider]:
    """按协议名返回实现类。"""
    from xhs_pain_miner.llm.anthropic_messages import AnthropicMessagesProvider
    from xhs_pain_miner.llm.openai_chat import OpenAIChatProvider
    from xhs_pain_miner.llm.openai_responses import OpenAIResponsesProvider

    registry: dict[str, type[BaseLLMProvider]] = {
        "chat": OpenAIChatProvider,
        "responses": OpenAIResponsesProvider,
        "messages": AnthropicMessagesProvider,
    }
    try:
        return registry[protocol]
    except KeyError:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"不支持的 LLM 协议 {protocol!r}，可选: {supported}") from None


def build_provider(settings: Settings, *, purpose: Purpose = "text") -> BaseLLMProvider:
    """按用途构造供应商。

    Args:
        settings: 全局配置。
        purpose: ``text`` 走 LLM 配置；``vision`` 走 VLM 配置（未单独配置时继承 LLM）。

    Returns:
        已初始化的供应商实例，调用方负责 ``close()``。

    Raises:
        ValueError: 协议名非法。
        LLMError: 缺少 API Key。
    """
    if purpose == "vision":
        protocol = settings.effective_vlm_protocol
        api_key = settings.effective_vlm_api_key
        base_url = settings.effective_vlm_base_url
        model = settings.effective_vlm_model
    else:
        protocol = settings.llm_protocol
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        model = settings.llm_model

    provider = _provider_class(protocol)(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
    )
    provider.require_api_key()
    return provider


def describe_protocol(protocol: str) -> str:
    """返回协议的简短说明，供 ``doctor`` 展示。"""
    return {
        "chat": "OpenAI Chat Completions（兼容 DeepSeek / Qwen / Kimi / GLM）",
        "responses": "OpenAI Responses API",
        "messages": "Anthropic Messages API",
    }.get(protocol, f"未知协议: {protocol}")
