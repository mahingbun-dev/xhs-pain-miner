"""Anthropic Messages API 协议适配。

与 OpenAI 系协议的差异：

* system 提示通过顶层 ``system`` 参数传入。
* ``max_tokens`` 是**必填**参数，因此这里需要给出默认值。
* 图片既可用 ``{"type": "url", ...}`` 引用远程地址，也可用 base64 内联；
  内联能让「本地压缩后再上传」这条降本链路对 Claude 同样生效。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from xhs_pain_miner.llm.base import (
    BaseLLMProvider,
    LLMError,
    LLMResponse,
    Message,
    split_data_url,
)

DEFAULT_MAX_TOKENS = 4096
"""Messages API 的 ``max_tokens`` 为必填项，未指定时使用该值。"""


class AnthropicMessagesProvider(BaseLLMProvider):
    """基于 ``/v1/messages`` 的补全实现。"""

    name = "messages"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
        )
        self._client = anthropic.Anthropic(
            api_key=api_key or "not-set",
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------ 内部 --
    @staticmethod
    def _split_messages(
        messages: Sequence[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """把 system 消息剥离到顶层 ``system`` 参数。

        Messages API 要求 ``messages`` 中只允许 user/assistant，且必须以 user 开头。
        """
        system_parts: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            items.append({"role": message.role, "content": message.content})

        if items and items[0]["role"] != "user":
            items.insert(0, {"role": "user", "content": "(继续)"})

        return ("\n\n".join(system_parts) or None), items

    @staticmethod
    def _to_image_block(image: str) -> dict[str, Any]:
        """把图片地址转成 Messages API 的 image block。"""
        if image.startswith("data:"):
            media_type, data = split_data_url(image)
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        return {"type": "image", "source": {"type": "url", "url": image}}

    def _create(self, **kwargs: Any) -> LLMResponse:
        """执行调用并把 Anthropic 异常统一转成 :class:`LLMError`。"""
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise LLMError(f"[messages] 调用 {self.model} 失败: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(response, "model", "") or self.model,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw=response,
        )

    # ------------------------------------------------------------------ 文本 --
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """文本补全。"""
        system, items = self._split_messages(messages)
        if not items:
            raise LLMError("[messages] 至少需要一条非 system 消息")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": items,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system:
            kwargs["system"] = system

        return self._record(self._create(**kwargs))

    # ------------------------------------------------------------------ 视觉 --
    def complete_vision(
        self,
        prompt: str,
        images: Sequence[str],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """多模态补全。图片地址支持 ``http(s)://`` 与 Data URL。"""
        if not images:
            raise LLMError("[messages] complete_vision 至少需要一张图片")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(self._to_image_block(image) for image in images)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system:
            kwargs["system"] = system

        return self._record(self._create(**kwargs), is_vision=True)

    def close(self) -> None:
        """关闭底层 HTTP 连接。"""
        self._client.close()
