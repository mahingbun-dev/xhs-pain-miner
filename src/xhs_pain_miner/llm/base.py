"""LLM 协议适配层 —— 统一的文本与视觉补全接口。

支持三种协议（由 ``Settings.llm_protocol`` 选择）：

=====================  ==========================================================
``chat``               OpenAI Chat Completions。一套即可覆盖 OpenAI、DeepSeek、
                       Qwen（DashScope 兼容模式）、Kimi、GLM 等绝大多数服务。
``responses``          OpenAI Responses API。
``messages``           Anthropic Messages API。
=====================  ==========================================================

所有实现都必须：

1. 把调用次数与 token 用量累加到 ``self.usage``（:class:`~xhs_pain_miner.models.RunCost`），
   供成本报告使用。
2. 在网络/鉴权/限流错误时抛出 :class:`LLMError`，**不得**静默返回空字符串 ——
   静默失败会让下游把「调用失败」误判成「没有痛点」。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from xhs_pain_miner.models import RunCost

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """LLM 调用失败（网络 / 鉴权 / 限流 / 内容过滤 / 响应格式异常）。"""


@dataclass(slots=True)
class Message:
    """一条对话消息。"""

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        """构造 system 消息。"""
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        """构造 user 消息。"""
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        """构造 assistant 消息。"""
        return cls("assistant", content)


@dataclass(slots=True)
class LLMResponse:
    """一次补全的结果。"""

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)
    """原始响应对象，仅供调试，不参与序列化。"""


@runtime_checkable
class LLMProvider(Protocol):
    """LLM 供应商的统一接口。"""

    name: str
    """协议名，用于 doctor 展示。"""

    usage: RunCost
    """累计用量。"""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """文本补全。"""
        ...

    def complete_vision(
        self,
        prompt: str,
        images: Sequence[str],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """带图片的多模态补全。

        Args:
            images: 图片地址列表，支持 ``https://`` URL 或 ``data:image/...;base64,...``
                Data URL。
        """
        ...

    def close(self) -> None:
        """释放底层连接。"""
        ...


class BaseLLMProvider:
    """提供用量累计与通用校验的基类。

    子类只需实现 :meth:`complete` 与 :meth:`complete_vision`，并在拿到响应后调用
    :meth:`_record`。
    """

    name = "base"

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
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.usage = RunCost()

    # ------------------------------------------------------------- 内部工具 --
    def require_api_key(self) -> str:
        """确认 API Key 已配置，否则抛出可读的错误。"""
        if not self.api_key:
            raise LLMError(
                f"{self.name} 协议需要 API Key。请设置环境变量 LLM_API_KEY"
                "（或 DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY），"
                "或在 .env 中配置。"
            )
        return self.api_key

    def _record(self, response: LLMResponse, *, is_vision: bool = False) -> LLMResponse:
        """把一次调用的用量累加进 ``self.usage``。"""
        self.usage.llm_calls += 1
        self.usage.llm_input_tokens += response.input_tokens
        self.usage.llm_output_tokens += response.output_tokens
        if is_vision:
            self.usage.vlm_calls += 1
            self.usage.vlm_images += 1
        return response

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """文本补全（子类实现）。"""
        raise NotImplementedError

    def complete_vision(
        self,
        prompt: str,
        images: Sequence[str],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """多模态补全（子类实现）。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放底层连接。基类无连接需要释放。"""


# --------------------------------------------------------------------------- #
# 通用辅助
# --------------------------------------------------------------------------- #

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从 LLM 回复中稳健地抽取 JSON。

    LLM 常见的三种返回形态都要能处理：裸 JSON、`````json`` 代码块包裹、正文中夹带 JSON。

    Args:
        text: LLM 的原始回复。

    Returns:
        解析后的 Python 对象。

    Raises:
        LLMError: 完全无法解析出 JSON 时抛出，并保留原文片段便于排查。
    """
    candidates: list[str] = []

    stripped = text.strip()
    candidates.append(stripped)

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # 退而求其次：截取第一个 { 或 [ 到最后一个 } 或 ]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    preview = text[:200].replace("\n", " ")
    raise LLMError(f"无法从模型回复中解析出 JSON。回复开头：{preview!r}")


def build_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """把图片字节编码为 Data URL。

    Anthropic Messages API 不接受远程 URL 直接作为 ``source``，统一转成 Data URL
    可以让三种协议共用同一条图片处理链路。
    """
    import base64

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def split_data_url(data_url: str) -> tuple[str, str]:
    """把 Data URL 拆成 ``(media_type, base64_data)``。

    Raises:
        ValueError: 传入的不是 Data URL。
    """
    if not data_url.startswith("data:"):
        raise ValueError("不是合法的 Data URL")
    header, _, payload = data_url.partition(",")
    media_type = header[len("data:") :].split(";")[0] or "image/jpeg"
    return media_type, payload
