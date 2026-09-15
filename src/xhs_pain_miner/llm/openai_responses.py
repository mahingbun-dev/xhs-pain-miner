"""OpenAI Responses API 协议适配。

与 Chat Completions 的差异：

* system 提示通过顶层 ``instructions`` 参数传入，而不是放进 ``input`` 数组。
* 输入项使用 ``input_text`` / ``input_image`` 块类型。
* 文本输出通过 ``response.output_text`` 便捷属性获取。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import openai

from xhs_pain_miner.llm.base import BaseLLMProvider, LLMError, LLMResponse, Message


class OpenAIResponsesProvider(BaseLLMProvider):
    """基于 ``/v1/responses`` 的补全实现。"""

    name = "responses"

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

    # ------------------------------------------------------------------ 内部 --
    @staticmethod
    def _split_messages(
        messages: Sequence[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """把 system 消息剥离到 ``instructions``，其余转成 Responses 输入项。"""
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                instructions.append(message.content)
                continue
            items.append({"role": message.role, "content": message.content})
        return ("\n\n".join(instructions) or None), items

    def _create(self, **kwargs: Any) -> LLMResponse:
        """执行调用并把 OpenAI 异常统一转成 :class:`LLMError`。"""
        try:
            response = self._client.responses.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(f"[responses] 调用 {self.model} 失败: {exc}") from exc

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=getattr(response, "output_text", "") or "",
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
        instructions, items = self._split_messages(messages)
        if not items:
            raise LLMError("[responses] 至少需要一条非 system 消息")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": items,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

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
            raise LLMError("[responses] complete_vision 至少需要一张图片")

        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image in images:
            content.append({"type": "input_image", "image_url": image})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system:
            kwargs["instructions"] = system
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        return self._record(self._create(**kwargs), is_vision=True)

    def close(self) -> None:
        """关闭底层 HTTP 连接。"""
        self._client.close()
