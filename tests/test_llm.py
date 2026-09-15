"""LLM 适配层测试。

不发起任何真实网络请求 —— 只验证协议分发、参数装配、JSON 解析与错误处理。
真实的模型调用验证放在 M1 的成本验收中。
"""

from __future__ import annotations

import pytest

from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.base import (
    LLMError,
    Message,
    build_data_url,
    extract_json,
    split_data_url,
)
from xhs_pain_miner.llm.factory import build_provider, describe_protocol

_MANAGED_ENV = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理可能让「缺少 Key」测试意外通过的 provider 环境变量。"""
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)


def _settings(**overrides: object) -> Settings:
    """构造不读取 .env 的配置对象。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestMessage:
    """消息构造。"""

    def test_convenience_constructors(self):
        assert Message.system("s").role == "system"
        assert Message.user("u").role == "user"
        assert Message.assistant("a").role == "assistant"

    def test_content_preserved(self):
        assert Message.user("你好").content == "你好"


class TestExtractJson:
    """从模型回复中抽取 JSON —— 提示词工程的现实：模型不会只回纯 JSON。"""

    def test_plain_object(self):
        assert extract_json('{"label": "假白"}') == {"label": "假白"}

    def test_fenced_block(self):
        text = '这是分析结果：\n```json\n{"label": "搓泥"}\n```\n以上。'
        assert extract_json(text) == {"label": "搓泥"}

    def test_fenced_without_language_tag(self):
        text = '```\n{"label": "闷痘"}\n```'
        assert extract_json(text) == {"label": "闷痘"}

    def test_json_embedded_in_prose(self):
        text = '好的，结果如下 {"label": "不防水"} 希望有帮助'
        assert extract_json(text) == {"label": "不防水"}

    def test_json_array(self):
        assert extract_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]

    def test_nested_object(self):
        text = '```json\n{"clusters": [{"label": "x", "size": 3}]}\n```'
        assert extract_json(text) == {"clusters": [{"label": "x", "size": 3}]}

    def test_unparsable_raises_with_preview(self):
        """解析失败必须抛错并带上原文片段，便于排查是提示词问题还是模型抽风。"""
        with pytest.raises(LLMError) as exc_info:
            extract_json("我无法完成这个任务。")
        assert "我无法完成这个任务" in str(exc_info.value)


class TestDataUrl:
    """Data URL 编解码。"""

    def test_roundtrip(self):
        raw = b"\xff\xd8\xff\xe0 fake jpeg bytes"
        data_url = build_data_url(raw, "image/jpeg")
        assert data_url.startswith("data:image/jpeg;base64,")
        media_type, payload = split_data_url(data_url)
        assert media_type == "image/jpeg"
        assert payload

    def test_split_requires_data_url(self):
        with pytest.raises(ValueError, match="Data URL"):
            split_data_url("https://example.invalid/a.jpg")

    def test_png_media_type(self):
        data_url = build_data_url(b"png-bytes", "image/png")
        assert split_data_url(data_url)[0] == "image/png"


class TestProviderFactory:
    """协议分发。"""

    def test_chat_protocol(self):
        provider = build_provider(_settings(llm_protocol="chat", llm_api_key="sk-test"))
        assert provider.name == "chat"
        provider.close()

    def test_responses_protocol(self):
        provider = build_provider(_settings(llm_protocol="responses", llm_api_key="sk-test"))
        assert provider.name == "responses"
        provider.close()

    def test_messages_protocol(self):
        provider = build_provider(_settings(llm_protocol="messages", llm_api_key="sk-test"))
        assert provider.name == "messages"
        provider.close()

    def test_vision_uses_vlm_settings(self):
        """``purpose="vision"`` 时必须走 VLM 那套配置，否则 --deep 会错误地打到文本模型。"""
        settings = _settings(
            llm_model="deepseek-chat",
            llm_api_key="sk-text",
            vlm_model="qwen-vl-max",
            vlm_api_key="sk-vision",
        )
        provider = build_provider(settings, purpose="vision")
        assert provider.model == "qwen-vl-max"
        assert provider.api_key == "sk-vision"
        provider.close()

    def test_missing_api_key_raises_readable_error(self):
        with pytest.raises(LLMError, match="API Key"):
            build_provider(_settings(llm_api_key=None))

    def test_invalid_protocol_raises(self):
        """绕过 pydantic 校验注入非法协议，验证工厂自身的兜底分支。"""
        settings = _settings(llm_api_key="sk-test")
        settings.llm_protocol = "carrier-pigeon"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="不支持"):
            build_provider(settings)

    def test_usage_starts_empty(self):
        provider = build_provider(_settings(llm_api_key="sk-test"))
        assert provider.usage.total_calls == 0
        provider.close()

    @pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
    def test_describe_protocol_covers_all(self, protocol: str):
        assert describe_protocol(protocol) != f"未知协议: {protocol}"
