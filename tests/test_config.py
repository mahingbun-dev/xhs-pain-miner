"""配置层测试。

测试统一传 ``_env_file=None`` 关闭 ``.env`` 加载，并用 monkeypatch 清理环境变量，
保证结果不受开发机上的本地配置影响。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xhs_pain_miner.config import Settings, _mask, load_settings

_MANAGED_ENV = (
    "LLM_API_KEY",
    "LLM_PROTOCOL",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "VLM_MODEL",
    "VLM_API_KEY",
    "VLM_BASE_URL",
    "VLM_PROTOCOL",
    "COLLECTOR_BACKEND",
    "COLLECTOR_PLUGIN",
    "XHS_COLLECTOR_PLUGIN",
    "SHARE_RESULTS",
    "MAX_NOTES",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理可能影响断言的环境变量。"""
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)


def _settings(**overrides: object) -> Settings:
    """构造不读取 .env 的配置对象。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestDefaults:
    """默认值。"""

    def test_default_protocol_is_chat(self):
        assert _settings().llm_protocol == "chat"

    def test_default_embedding_is_local(self):
        """默认本地 embedding 是成本控制的关键，不应被改回远程 API。"""
        settings = _settings()
        assert settings.embedding_provider == "local"
        assert "bge" in settings.embedding_model

    def test_default_collector_is_fixture(self):
        """默认采集后端必须是内置样例 —— 不能开箱就去打真实平台。"""
        assert _settings().collector_backend == "fixture"

    def test_default_sharing_is_off(self):
        """★ 隐私默认值：结果共享必须默认关闭。"""
        assert _settings().share_results is False

    def test_default_notes_limit(self):
        assert _settings().max_notes == 100


class TestEnvOverrides:
    """环境变量覆盖。"""

    def test_llm_api_key_from_generic_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-generic")
        assert _settings().llm_api_key == "sk-generic"

    @pytest.mark.parametrize(
        "env_name",
        ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DASHSCOPE_API_KEY"],
    )
    def test_llm_api_key_falls_back_to_provider_vars(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str
    ):
        """各家 SDK 惯用的环境变量名都要能识别，否则用户会一脸茫然。"""
        monkeypatch.setenv(env_name, "sk-from-provider")
        assert _settings().llm_api_key == "sk-from-provider"

    def test_backend_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COLLECTOR_BACKEND", "plugin")
        assert _settings().collector_backend == "plugin"

    def test_plugin_path_accepts_prefixed_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """★ 文档里写的是 XHS_COLLECTOR_PLUGIN。

        pydantic-settings **不会**自动加 ``XHS_`` 前缀 —— 曾经因为只声明了裸名，
        用户照文档设置环境变量会被静默忽略，插件永远加载不上。
        """
        monkeypatch.setenv("XHS_COLLECTOR_PLUGIN", "/tmp/my_backend.py")
        assert _settings().collector_plugin == "/tmp/my_backend.py"

    def test_plugin_path_accepts_bare_env_var(self, monkeypatch: pytest.MonkeyPatch):
        """裸名保持向后兼容。"""
        monkeypatch.setenv("COLLECTOR_PLUGIN", "/tmp/my_backend.py")
        assert _settings().collector_plugin == "/tmp/my_backend.py"

    def test_invalid_protocol_is_rejected(self):
        """非法协议名必须立即报错，而不是等到调用时才发现。"""
        with pytest.raises(ValueError):
            _settings(llm_protocol="grpc")


class TestVlmInheritance:
    """VLM 配置的继承规则 —— 让用户只配一个多模态模型即可。"""

    def test_inherits_when_unset(self):
        settings = _settings(llm_model="deepseek-chat", llm_api_key="sk-1")
        assert settings.effective_vlm_model == "deepseek-chat"
        assert settings.effective_vlm_api_key == "sk-1"

    def test_own_values_win(self):
        settings = _settings(
            llm_model="deepseek-chat",
            llm_api_key="sk-text",
            vlm_model="qwen-vl-max",
            vlm_api_key="sk-vision",
        )
        assert settings.effective_vlm_model == "qwen-vl-max"
        assert settings.effective_vlm_api_key == "sk-vision"

    def test_empty_base_url_does_not_fall_back(self):
        """显式设为空串表示「用协议默认端点」，不应继承 LLM 的 base_url。"""
        settings = _settings(llm_base_url="https://api.deepseek.com", vlm_base_url="")
        assert settings.effective_vlm_base_url == ""

    def test_embedding_falls_back_to_llm_key(self):
        settings = _settings(llm_api_key="sk-1", embedding_provider="api")
        assert settings.effective_embedding_api_key == "sk-1"


class TestMask:
    """密钥脱敏 —— 保证终端输出与日志里不会出现完整密钥。"""

    def test_masks_long_secret(self):
        masked = _mask("sk-abcdefghijklmnop")
        assert masked.startswith("sk-abc")
        assert masked.endswith("mnop")
        assert "defghijkl" not in masked

    def test_handles_none(self):
        assert _mask(None) == "(未设置)"

    def test_handles_short_secret(self):
        assert _mask("abc") == "ab***"

    def test_settings_mask_helpers(self):
        settings = _settings(llm_api_key="sk-abcdefghijklmnop")
        assert "defghijkl" not in settings.masked_llm_key()
        assert "defghijkl" not in settings.masked_vlm_key()


class TestLoadSettings:
    """``load_settings`` 便捷函数。"""

    def test_ignores_none_overrides(self, monkeypatch: pytest.MonkeyPatch):
        """``None`` 表示「不覆盖」，否则 CLI 的默认选项会把环境变量清空。"""
        monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
        settings = load_settings(llm_model=None)
        assert settings.llm_model == "deepseek-chat"

    def test_applies_overrides(self):
        settings = load_settings(llm_model="gpt-4o", llm_api_key="sk-x")
        assert settings.llm_model == "gpt-4o"
        assert settings.llm_api_key == "sk-x"


class TestPaths:
    """路径处理。"""

    def test_ensure_dirs_creates_both(self, tmp_path: Path):
        settings = _settings(
            output_dir=tmp_path / "out",
            db_path=tmp_path / "db" / "db.sqlite",
        )
        settings.ensure_dirs()
        assert (tmp_path / "out").is_dir()
        assert (tmp_path / "db").is_dir()

    def test_max_notes_hard_limit_is_at_least_one(self):
        assert _settings(max_notes=0).max_notes_hard_limit == 1
