"""门面与 CLI 测试。

CLI 测试全部走内置样例后端，不联网、不需要 API Key。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from xhs_pain_miner import PainMiner, PipelineNotAvailableError
from xhs_pain_miner.cli import main
from xhs_pain_miner.config import Settings
from xhs_pain_miner.pain_miner import normalize_keyword

_MANAGED_ENV = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "COLLECTOR_BACKEND",
    "COLLECTOR_PLUGIN",
    "XHS_COLLECTOR_PLUGIN",
    "SHARE_RESULTS",
    "MAX_NOTES",
    "DB_PATH",
    "OUTPUT_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """隔离环境变量，并把输出路径重定向到临时目录。

    环境变量必须隔离 —— 否则开发机上的真实配置（API Key、后端选择）会让断言失真；
    路径重定向则保证没有任何测试会写入用户主目录。
    """
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "db" / "db.sqlite"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))


@pytest.fixture
def runner() -> CliRunner:
    """CLI 测试运行器。"""
    return CliRunner()


def _settings(**overrides: object) -> Settings:
    """构造不读取 .env 的配置对象。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestPainMinerFacade:
    """门面类。"""

    def test_collect_returns_corpus(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        corpus = miner.collect("防晒霜", limit=5)
        assert corpus.keyword == "防晒霜"
        assert len(corpus.notes) == 5
        assert corpus.backend == "fixture"

    def test_collect_uses_settings_defaults(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture", max_notes=4))
        assert len(miner.collect("防晒霜").notes) == 4

    def test_collect_limit_overrides_settings(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture", max_notes=4))
        assert len(miner.collect("防晒霜", limit=7).notes) == 7

    def test_mine_reports_pending_milestone(self):
        """M1 之前，mine() 必须抛出明确的异常而不是静默返回空结果。"""
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        with pytest.raises(PipelineNotAvailableError, match="M1"):
            miner.mine("防晒霜")

    def test_context_manager(self):
        with PainMiner(settings=_settings(collector_backend="fixture")) as miner:
            assert miner.settings.collector_backend == "fixture"

    def test_api_key_convenience_argument(self):
        """便捷参数只覆盖指定字段，不破坏其它配置。"""
        miner = PainMiner(api_key="sk-inline", model="deepseek-chat")
        assert miner.settings.llm_api_key == "sk-inline"
        assert miner.settings.llm_model == "deepseek-chat"


class TestCollectCommand:
    """``collect`` 子命令。"""

    def test_runs_with_fixture_backend(self, runner: CliRunner):
        result = runner.invoke(main, ["collect", "-k", "防晒霜", "--backend", "fixture"])
        assert result.exit_code == 0, result.output
        assert "采集完成" in result.output
        assert "防晒霜" in result.output

    def test_respects_note_limit(self, runner: CliRunner):
        result = runner.invoke(main, ["collect", "-k", "防晒霜", "--backend", "fixture", "-n", "3"])
        assert result.exit_code == 0, result.output
        assert "3 篇笔记" in result.output

    def test_saves_corpus_as_json(self, runner: CliRunner, tmp_path: Path):
        target = tmp_path / "corpus.json"
        result = runner.invoke(
            main,
            ["collect", "-k", "防晒霜", "--backend", "fixture", "--save", str(target)],
        )
        assert result.exit_code == 0, result.output
        assert target.is_file()
        assert "防晒霜" in target.read_text(encoding="utf-8")

    def test_mcp_backend_fails_with_guidance(self, runner: CliRunner):
        result = runner.invoke(main, ["collect", "-k", "防晒霜", "--backend", "mcp"])
        assert result.exit_code != 0
        assert "M3" in result.output

    def test_missing_keyword_is_usage_error(self, runner: CliRunner):
        result = runner.invoke(main, ["collect"])
        assert result.exit_code != 0


class TestMineCommand:
    """``mine`` 子命令。"""

    def test_reports_pending_pipeline(self, runner: CliRunner):
        """M1 之前必须给出「开发中」提示与退出码 2，而不是崩溃。"""
        result = runner.invoke(main, ["mine", "-k", "防晒霜", "--backend", "fixture"])
        assert result.exit_code == 2
        assert "M1" in result.output
        assert "collect" in result.output

    def test_accepts_deep_flag(self, runner: CliRunner):
        result = runner.invoke(main, ["mine", "-k", "防晒霜", "--backend", "fixture", "--deep"])
        assert result.exit_code == 2
        assert "--deep" in result.output


class TestDoctorCommand:
    """``doctor`` 子命令。"""

    def test_fails_without_api_key(self, runner: CliRunner):
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 2
        assert "LLM" in result.output

    def test_passes_with_api_key(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-test-1234567890")
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0, result.output
        assert "环境可用" in result.output

    def test_never_leaks_full_key(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch):
        """★ 合规：doctor 输出里不得出现完整密钥。"""
        secret = "sk-abcdefghijklmnopqrstuvwxyz"
        monkeypatch.setenv("LLM_API_KEY", secret)
        result = runner.invoke(main, ["doctor"])
        assert secret not in result.output

    def test_warns_when_sharing_enabled(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch):
        """开启结果共享时必须给出醒目提示，避免用户不知情地上传数据。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-test-1234567890")
        monkeypatch.setenv("SHARE_RESULTS", "true")
        result = runner.invoke(main, ["doctor"])
        assert "结果共享已开启" in result.output


class TestCliBasics:
    """CLI 基础行为。"""

    def test_version(self, runner: CliRunner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_lists_all_commands(self, runner: CliRunner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for command in ("mine", "collect", "doctor"):
            assert command in result.output


def _combined(result: object) -> str:
    """合并 stdout 与 stderr。

    click 8.2 起 CliRunner 把两者分开，而 `_fail` 与 click 的参数错误都写 stderr。
    """
    output = getattr(result, "output", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return output + stderr


class TestRichMarkupSafety:
    """用户输入与采集内容里的方括号必须被转义。

    rich 默认把 ``[...]`` 当样式标签解析：轻则崩溃（``[/]`` 没有可闭合的标签），
    重则采集到的内容可注入 ``[link=...]`` 渲染出可点击的终端超链接。

    **断言必须是「关键字面量完整出现」，不能只是「没崩溃」** ——
    ``[bold]`` / ``a[b]c`` 这类载荷即使完全不转义也不会让程序崩溃
    （rich 会静默把它们当标签吃掉），所以「没崩溃」是恒真断言，守不住任何东西。
    """

    @pytest.mark.parametrize(
        "keyword",
        [
            "[/]",
            "[bold]",
            "a[b]c",
            "[red]红[/red]",
            "[link=http://evil.example]x[/link]",
            "[[[]]]",
            "[on red]",
            "[blink]",
            "防晒[霜]",
        ],
    )
    def test_keyword_appears_verbatim(self, runner: CliRunner, keyword: str):
        """转义生效时关键词以字面量完整出现；转义失效时方括号会被当标签吃掉。"""
        result = runner.invoke(main, ["collect", "-k", keyword, "--backend", "fixture", "-n", "1"])
        assert result.exit_code == 0, _combined(result)

        output = _combined(result)
        assert keyword in output, (
            f"关键词 {keyword!r} 未以字面量出现在输出中 —— rich markup 转义可能失效"
        )

    def test_link_markup_does_not_produce_osc8_hyperlink(self, runner: CliRunner):
        """link 标签不得被渲染成终端超链接（OSC-8）。

        仅断言 "clickme 出现在输出里" 是**无效**的：渲染成超链接时它同样出现。
        必须断言 link 标签以字面量完整出现。
        """
        payload = "[link=http://evil.example]clickme[/link]"
        result = runner.invoke(main, ["collect", "-k", payload, "--backend", "fixture", "-n", "1"])
        assert result.exit_code == 0, _combined(result)

        assert "[link=http://evil.example]" in _combined(result), "link 标签被当成样式解析了"

    def test_bare_escape_sequences_are_stripped(self, runner: CliRunner):
        """★ 裸 ANSI / OSC 控制序列必须被剥离 —— 只防 rich markup 是不够的。

        ``rich.escape()`` 只处理 ``[``，不管控制字符。当输出被重定向到文件、管道
        或 CI 日志（非 TTY）时，采集数据里的裸 ``ESC`` 会被逐字节写出 ——
        包括可点击的 OSC-8 超链接、``ESC[2J`` 清屏、甚至写剪贴板（OSC-52）。
        """
        payload = "\x1b[2J\x1b]8;;http://evil.example\x1b\\click\x1b]8;;\x1b\\"
        result = runner.invoke(main, ["collect", "-k", payload, "--backend", "fixture", "-n", "1"])
        assert result.exit_code == 0, _combined(result)

        output = _combined(result)
        assert "\x1b" not in output, "输出里出现了裸 ESC 控制字符"
        assert "click" in output, "可读部分不应被一并删除"

    def test_control_chars_in_collected_content_are_stripped(
        self, runner: CliRunner, tmp_path: Path
    ):
        """采集内容同样不可信 —— 插件返回的 title 里塞 ANSI 也必须被清洗。"""
        plugin = tmp_path / "ansi_plugin.py"
        plugin.write_text(
            "from xhs_pain_miner.models import RawCorpus, RawNote\n"
            "\n"
            "\n"
            "class B:\n"
            "    name = 'ansi'\n"
            "\n"
            "    def collect(self, keyword, *, limit, max_comments_per_note=20):\n"
            "        return RawCorpus(keyword=keyword, backend=self.name, notes=[\n"
            "            RawNote(note_id='n1', title='\\x1b[31mRED\\x1b[0m 正常标题'),\n"
            "        ])\n"
            "\n"
            "    def available(self):\n"
            "        return True\n"
            "\n"
            "\n"
            "BACKEND = B()\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            main,
            ["collect", "-k", "防晒霜", "--backend", "plugin", "-n", "1"],
            env={"XHS_COLLECTOR_PLUGIN": str(plugin)},
        )
        assert result.exit_code == 0, _combined(result)

        output = _combined(result)
        assert "\x1b" not in output, "采集内容里的 ANSI 序列未被清洗"
        assert "正常标题" in output

    def test_mine_also_escapes_keyword(self, runner: CliRunner):
        result = runner.invoke(main, ["mine", "-k", "[/]", "--backend", "fixture"])
        # 走到 M1 未实现的分支，但不应该因 markup 而崩
        assert result.exit_code == 2, _combined(result)
        assert "[/]" in _combined(result), "mine 的关键词未被转义"
        assert "M1" in _combined(result)


class TestKeywordValidation:
    """关键词校验 —— 空关键词会让样例后端静默回退到它自带的关键词。"""

    @pytest.mark.parametrize("keyword", ["", "   ", "\t"])
    def test_blank_keyword_is_rejected(self, runner: CliRunner, keyword: str):
        result = runner.invoke(main, ["collect", "-k", keyword, "--backend", "fixture"])
        assert result.exit_code != 0
        assert "不能为空" in _combined(result)

    def test_surrounding_whitespace_is_stripped(self, runner: CliRunner):
        result = runner.invoke(
            main, ["collect", "-k", "  防晒霜  ", "--backend", "fixture", "-n", "1"]
        )
        assert result.exit_code == 0, _combined(result)
        assert "防晒霜" in _combined(result)

    def test_zero_width_keyword_is_rejected(self, runner: CliRunner):
        """零宽字符肉眼不可见、``strip()`` 也不会去掉，必须被显式过滤。"""
        result = runner.invoke(main, ["collect", "-k", "\u200b", "--backend", "fixture", "-n", "1"])
        assert result.exit_code != 0
        assert "不能为空" in _combined(result)


class TestKeywordValidationAtApiLevel:
    """关键词校验必须**同时**在 API 层生效。

    只挂在 click 的 callback 上是不够的：库使用者（脚本、M4 的 Skill）会绕过 CLI，
    拿到一个被后端静默替换过的关键词 —— 而 ``corpus.keyword`` 与传入值不一致
    是个没人会注意到的数据错误。
    """

    @pytest.mark.parametrize("keyword", ["", "   ", "\t", "\n", "\u200b", "\u200b\u200b", "　"])
    def test_collect_rejects_blank_keyword(self, keyword: str):
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        with pytest.raises(ValueError, match="不能为空"):
            miner.collect(keyword)

    def test_collect_rejects_non_string(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        with pytest.raises(ValueError, match="字符串"):
            miner.collect(123)  # type: ignore[arg-type]

    def test_keyword_is_stripped(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        assert miner.collect("  防晒霜  ", limit=1).keyword == "防晒霜"

    def test_zero_width_chars_are_stripped(self):
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        assert miner.collect("\u200b防晒霜\u200b", limit=1).keyword == "防晒霜"

    def test_keyword_never_silently_falls_back(self):
        """★ 核心回归：这曾经会静默变成样例数据自带的 '防晒霜'。"""
        miner = PainMiner(settings=_settings(collector_backend="fixture"))
        with pytest.raises(ValueError):
            miner.collect("")


class TestNormalizeKeyword:
    """关键词规范化函数本身的契约。"""

    def test_strips_whitespace(self):
        assert normalize_keyword("  防晒霜 ") == "防晒霜"

    def test_strips_zero_width(self):
        assert normalize_keyword("\u200b\u200c防晒霜\ufeff") == "防晒霜"

    @pytest.mark.parametrize("value", ["", "   ", "\u200b", "\u2060"])
    def test_rejects_empty_after_normalization(self, value: str):
        with pytest.raises(ValueError, match="不能为空"):
            normalize_keyword(value)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="字符串"):
            normalize_keyword(None)  # type: ignore[arg-type]


class TestSaveErrors:
    """``--save`` 的失败路径必须是可读错误，而不是裸 traceback。"""

    def test_unwritable_path_reports_readable_error(self, runner: CliRunner, tmp_path: Path):
        target = tmp_path / "no-such-dir" / "corpus.json"  # 父目录不存在
        result = runner.invoke(
            main,
            ["collect", "-k", "防晒霜", "--backend", "fixture", "-n", "1", "--save", str(target)],
        )
        assert result.exit_code == 1
        assert "无法写入" in _combined(result)


class TestDoctorHasNoSideEffects:
    """``doctor`` 被文档描述为"只做本地检查"，不该在用户主目录创建目录。"""

    def test_does_not_create_db_directory(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        db_dir = tmp_path / "brand-new-dir"
        monkeypatch.setenv("DB_PATH", str(db_dir / "db.sqlite"))

        result = runner.invoke(main, ["doctor"])

        assert result.exit_code in (0, 2), _combined(result)
        assert not db_dir.exists(), "doctor 不应创建任何目录"


_PLUGIN_SOURCE = (
    "from xhs_pain_miner.models import RawCorpus\n"
    "\n"
    "\n"
    "class B:\n"
    "    name = 'env-plugin'\n"
    "\n"
    "    def collect(self, keyword, *, limit, max_comments_per_note=20):\n"
    "        return RawCorpus(keyword=keyword, backend=self.name)\n"
    "\n"
    "    def available(self):\n"
    "        return True\n"
    "\n"
    "\n"
    "BACKEND = B()\n"
)


class TestCollectorPluginEnvVar:
    """插件环境变量的名字必须与文档一致。

    pydantic-settings **不会**自动加 ``XHS_`` 前缀。文档里写的
    ``XHS_COLLECTOR_PLUGIN`` 曾经被静默忽略 —— 用户照文档操作永远加载不上插件，
    而报错信息还把他引回同一个错的变量名。
    """

    @pytest.fixture
    def plugin_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "ok_plugin.py"
        path.write_text(_PLUGIN_SOURCE, encoding="utf-8")
        return path

    def test_prefixed_name_works(
        self, runner: CliRunner, plugin_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """文档推荐的 XHS_COLLECTOR_PLUGIN 必须真的生效。"""
        monkeypatch.setenv("XHS_COLLECTOR_PLUGIN", str(plugin_file))

        result = runner.invoke(main, ["collect", "-k", "防晒霜", "--backend", "plugin", "-n", "1"])

        assert result.exit_code == 0, _combined(result)
        assert "env-plugin" in _combined(result)

    def test_bare_name_still_works(
        self, runner: CliRunner, plugin_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """裸名 COLLECTOR_PLUGIN 保持向后兼容。"""
        monkeypatch.setenv("COLLECTOR_PLUGIN", str(plugin_file))

        result = runner.invoke(main, ["collect", "-k", "防晒霜", "--backend", "plugin", "-n", "1"])

        assert result.exit_code == 0, _combined(result)


class TestDoctorShowsCollectorReason:
    """``doctor`` 必须展示插件 ``available()`` 失败的原因。

    否则「cookie 已失效，请重新登录」这类最有可操作性的信息会被完全吞掉，
    用户只看到一句无用的「当前不可用」。
    """

    def test_reason_appears_in_output(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        plugin = tmp_path / "flaky.py"
        plugin.write_text(
            "class B:\n"
            "    name = 'flaky-plugin'\n"
            "\n"
            "    def collect(self, *args, **kwargs):\n"
            "        raise RuntimeError('不该被调用')\n"
            "\n"
            "    def available(self):\n"
            "        raise RuntimeError('cookie 已失效，请重新登录')\n"
            "\n"
            "\n"
            "BACKEND = B()\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("LLM_API_KEY", "sk-test-1234567890")
        monkeypatch.setenv("COLLECTOR_BACKEND", "plugin")
        monkeypatch.setenv("XHS_COLLECTOR_PLUGIN", str(plugin))

        result = runner.invoke(main, ["doctor"])

        # rich 表格会把长文本折行并在中间插入边框字符（│），
        # 所以不能用连续子串断言，改为断言关键片段
        output = _combined(result)
        assert "cookie" in output, "插件的异常消息未展示给用户"
        assert "RuntimeError" in output, "异常类型未展示"


class TestSavePathValidation:
    """``--save`` 的路径校验。"""

    def test_empty_save_path_is_rejected(self, runner: CliRunner):
        """``--save ""`` 曾经静默什么都不做且 exit 0。"""
        result = runner.invoke(
            main, ["collect", "-k", "防晒霜", "--backend", "fixture", "-n", "1", "--save", ""]
        )
        assert result.exit_code == 2, _combined(result)
        assert "不能为空" in _combined(result)

    def test_whitespace_save_path_is_rejected(self, runner: CliRunner):
        result = runner.invoke(
            main, ["collect", "-k", "防晒霜", "--backend", "fixture", "-n", "1", "--save", "   "]
        )
        assert result.exit_code == 2, _combined(result)
