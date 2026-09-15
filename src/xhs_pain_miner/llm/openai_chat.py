"""OpenAI Chat Completions 协议适配。

这一套实现同时覆盖所有 OpenAI 兼容服务：OpenAI、DeepSeek、Qwen（DashScope 兼容模式）、
Kimi、GLM、硅基流动等 —— 只需改 ``LLM_BASE_URL`` 与 ``LLM_MODEL``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import openai

from xhs_pain_miner.llm.base import BaseLLMProvider, LLMError, LLMResponse, Message


class OpenAIChatProvider(BaseLLMProvider):
    """基于 ``/v1/chat/completions`` 的补全实现。"""

    name = "chat"

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
        self._client = openai.OpenAI(
            api_key=api_key or "not-set",
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
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
        payload: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(f"[chat] 调用 {self.model} 失败: {exc}") from exc

        if not response.choices:
            raise LLMError(f"[chat] {self.model} 返回了空的 choices")

        usage = response.usage
        return self._record(
            LLMResponse(
                text=response.choices[0].message.content or "",
                model=response.model or self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                raw=response,
            )
        )

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
        """多模态补全，支持 ``http(s)://`` 与 Data URL 两种图片地址。"""
        if not images:
            raise LLMError("[chat] complete_vision 至少需要一张图片")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image}})

        payload: list[dict[str, Any]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.append({"role": "user", "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(f"[chat] 视觉调用 {self.model} 失败: {exc}") from exc

        if not response.choices:
            raise LLMError(f"[chat] {self.model} 视觉返回了空的 choices")

        usage = response.usage
        return self._record(
            LLMResponse(
                text=response.choices[0].message.content or "",
                model=response.model or self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                raw=response,
            ),
            is_vision=True,
        )

    def close(self) -> None:
        """关闭底层 HTTP 连接。"""
        self._client.close()
