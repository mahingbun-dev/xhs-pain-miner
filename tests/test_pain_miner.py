"""门面与 CLI 测试。

CLI 测试全部走内置样例后端，不联网、不需要 API Key。

``mine`` 的端到端测试通过**注入假 LLM 与假编码器**完成 —— 这正是
:class:`~xhs_pain_miner.pain_miner.PainMiner` 留出 ``llm`` / ``embedder``
参数的原因：整条流水线必须能在没有网络、没有模型、没有 API Key 的环境里跑完，
否则 CI 就永远测不到它。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from xhs_pain_miner import PainMiner
from xhs_pain_miner.cli import main
from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.base import LLMError, LLMResponse
from xhs_pain_miner.models import RunCost
from xhs_pain_miner.pain_miner import normalize_keyword
from xhs_pain_miner.research import appstore as appstore_module
from xhs_pain_miner.research import github as github_module
from xhs_pain_miner.research import query, relevance

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


class _FakeEmbedder:
    """确定性假编码器 —— 让"哪些文本该聚在一起"完全可控。

    按字符分布构造向量：共享字符多的文本向量方向接近，因此同一痛点的不同措辞
    会聚到一起，不同痛点则分开。这不是好的语义编码，但它是**确定性的**，
    而测试要的正是确定性 —— 真实模型在 CI 里既慢又不可复现。

    刻意不实现 ``Embedder`` 协议的运行时检查（那会触发 ``dimension`` 读取），
    只提供协议要求的成员。
    """

    name = "fake"
    is_local = True

    def __init__(self, dimension: int = 12) -> None:
        self._dimension = dimension
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts, *, batch_size: int = 64):  # type: ignore[no-untyped-def]
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * self._dimension
            for index, char in enumerate(text):
                vector[index % self._dimension] += (ord(char) % 17) / 17.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def close(self) -> None:
        pass


_LABEL_JSON = (
    '{"label": "防晒霜搓泥", "summary": "上妆后起白条。", "category": "体验粗糙", '
    '"sentiment": -0.8, "stage": "growing", "difficulty": 2, '
    '"feasibility": "个人可做 / 1-2 周"}'
)

_TAXONOMY_NAMES = ("搓泥", "假白", "闷痘", "难卸", "价格")
"""假模型归纳出的痛点清单 —— 固定 5 个，与 ``MIN_PAINS`` 一致。"""


def _taxonomy_reply(prompt: str) -> str:
    """按提示词里的样本内容生成归纳结果。

    样本数必须从提示词里数出来（``[0]``、``[1]`` …）：``labels`` 的长度必须严格
    等于样本数，写死一个长度会在采样数变化时静默错位。
    """
    sample_lines = [line for line in prompt.splitlines() if line.startswith("[")]
    labels: list[str] = []
    for line in sample_lines:
        matched = next((name for name in _TAXONOMY_NAMES if name in line), None)
        labels.append(matched or "其他")

    pains = [
        {"name": name, "summary": f"{name}相关抱怨。", "category": "体验粗糙"}
        for name in _TAXONOMY_NAMES
    ]
    return json.dumps({"pains": pains, "labels": labels}, ensure_ascii=False)


_SOLUTION_QUERIES_JSON = json.dumps(
    {
        "queries": [
            {"text": "美妆 成分查询", "channel": "appstore"},
            {"text": "cosmetic ingredient lookup", "channel": "github"},
            {"text": "web clipper", "channel": "chrome"},
        ]
    },
    ensure_ascii=False,
)
"""解法检索词 —— 刻意带上一条 **未接入渠道**（chrome）的词。

生产环境里 T1 的提示词会让模型为四个渠道都出词，而本版本只实现了两个。假实现
必须复现这个形状，否则"未接入的渠道被跳过"这条约束在端到端测试里永远测不到。
"""

_RELEVANCE_JSON = '{"relevant": [], "rejected": [0, 1]}'


class _FakeLLM:
    """假 LLM —— 按提示词类型返回归纳 / 标注 / 检索词 / 相关性判定结果。

    M2 之后一个簇会走四类提示词，都从同一个 provider 出去，所以这里必须按**提示词
    的类型**分派。用系统提示词做判据（而不是"回复里有没有某个词"）：系统提示词是
    各模块自己定义的常量，改措辞时测试跟着一起变，不会静默失配。
    """

    name = "fake"

    def __init__(
        self,
        *,
        solution_reply: str = _SOLUTION_QUERIES_JSON,
        relevance_reply: str = _RELEVANCE_JSON,
    ) -> None:
        self.usage = RunCost()
        self.prompts: list[str] = []
        self.solution_reply = solution_reply
        self.relevance_reply = relevance_reply

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        prompt = messages[-1].content
        self.prompts.append(prompt)
        self.usage.llm_calls += 1
        self.usage.llm_input_tokens += 100
        self.usage.llm_output_tokens += 50
        joined = "".join(getattr(message, "content", "") for message in messages)
        if query.SYSTEM_PROMPT in joined:
            return LLMResponse(
                text=self.solution_reply, model="fake", input_tokens=100, output_tokens=50
            )
        if relevance.SYSTEM_PROMPT in joined:
            return LLMResponse(
                text=self.relevance_reply, model="fake", input_tokens=100, output_tokens=50
            )
        # 归纳阶段的提示词要求输出 pains + labels；标注阶段要求输出单个痛点属性
        text = _taxonomy_reply(prompt) if '"pains"' in prompt else _LABEL_JSON
        return LLMResponse(text=text, model="fake", input_tokens=100, output_tokens=50)

    def complete_vision(self, prompt, images, **kwargs):  # type: ignore[no-untyped-def]
        self.usage.vlm_calls += 1
        return LLMResponse(
            text='{"description": "对比图", "pain_hints": ["宣传图与实际不符"]}',
            model="fake-vl",
        )

    def close(self) -> None:
        pass


def _miner(*, llm: _FakeLLM | None = None, **setting_overrides: object) -> PainMiner:
    """构造一台全程离线的 PainMiner（假 LLM + 假编码器 + 样例语料）。

    ``vlm`` 也要注入：不注入的话 ``--deep`` 会去构造真实 provider 并因缺少
    API Key 失败 —— 那测的就不是流水线，而是"CI 里没有 Key"这件事。
    两个 provider 用各自的实例，与生产路径（文本/视觉两条链路）保持一致。

    ``llm`` 可替换成配置过回复的假模型（竞品调研那几条用例要按提示词类型给出
    不同回复）；不传则用默认的。

    默认 ``research_enabled=False``：只有明确针对竞品调研的用例才打开它，并把两个
    渠道的 HTTP 都接到假传输层上 —— 打开调研就会真的发请求，默认值必须是关的。
    """
    settings = _settings(
        collector_backend="fixture",
        **{"research_enabled": False, **setting_overrides},
    )
    return PainMiner(
        settings=settings,
        llm=llm if llm is not None else _FakeLLM(),
        vlm=_FakeLLM(),
        embedder=_FakeEmbedder(),
    )


def _miner_with_broken_llm() -> PainMiner:
    """构造一台 LLM 完全不可用的 PainMiner —— 用于验证降级路径。

    归纳阶段失败会让 ``_group_pains`` 退回聚类路径；而聚类路径的命名同样依赖
    LLM，所以这里代表的是"LLM 彻底宕机"这一最坏情形。
    """

    class _BrokenLLM(_FakeLLM):
        def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise LLMError("模拟服务不可用")

    settings = _settings(collector_backend="fixture", research_enabled=False)
    return PainMiner(
        settings=settings,
        llm=_BrokenLLM(),
        vlm=_BrokenLLM(),
        embedder=_FakeEmbedder(),
    )


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

    def test_mine_runs_end_to_end(self):
        """★ M1 核心：整条链路能在无网络、无 API Key、无模型的环境下跑完。"""
        result = _miner().mine("防晒霜", notes_count=40)

        assert result.keyword == "防晒霜"
        assert result.total_notes == 40
        assert result.cards, "没有产出任何机会卡片"
        assert result.clusters, "没有产出任何痛点簇"

    def test_mine_rejects_blank_keyword(self):
        with pytest.raises(ValueError, match="不能为空"):
            _miner().mine("   ")

    def test_mine_without_notes_returns_empty_result(self):
        """采集不到笔记时返回**空结果**而不是抛异常 —— 搜不到结果是正常情况。"""
        result = _miner().mine("防晒霜", notes_count=0)

        assert result.cards == []
        assert result.total_notes == 0
        assert any("没有采集到" in note for note in result.notes)

    def test_mine_reports_cost(self):
        """成本统计必须反映真实调用，否则验收门④的成本报告毫无意义。"""
        result = _miner().mine("防晒霜", notes_count=20)

        assert result.cost.llm_calls > 0
        assert result.cost.elapsed_seconds > 0

    def test_mine_survives_total_llm_failure(self):
        """★ 命名全部失败时必须降级出卡片，而不是让整次运行白跑。"""

        class _BrokenLLM(_FakeLLM):
            def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                raise LLMError("模拟服务不可用")

        settings = _settings(collector_backend="fixture", research_enabled=False)
        miner = PainMiner(settings=settings, llm=_BrokenLLM(), embedder=_FakeEmbedder())

        result = miner.mine("防晒霜", notes_count=20)

        assert result.cards, "LLM 全挂时仍应产出卡片（用降级的占位名）"
        assert any("降级" in note for note in result.notes), "降级必须如实告知用户"

    def test_mine_never_uses_raw_text_as_degraded_label(self):
        """★ 合规：降级占位名绝不能取自证据原文（那会绕过结构性脱敏）。"""
        from xhs_pain_miner.models import find_verbatim_overlap

        class _BrokenLLM(_FakeLLM):
            def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                raise LLMError("模拟服务不可用")

        settings = _settings(collector_backend="fixture", research_enabled=False)
        miner = PainMiner(settings=settings, llm=_BrokenLLM(), embedder=_FakeEmbedder())
        result = miner.mine("防晒霜", notes_count=20)

        for card in result.cards:
            sources = [ev.text for ev in card.pain.evidences]
            overlap = find_verbatim_overlap(card.pain.label, sources, min_len=6)
            assert overlap is None, f"降级标签回抄了原文：{overlap!r}"

    def test_degraded_to_clustering_still_fills_names(self):
        """★ 降级到聚类路径后，簇名仍必须被填上。

        这里曾经有个真 bug：``keep_labels`` 判的是**配置**（``taxonomy``）而不是
        **实际走的路径**。归纳失败降级到聚类后，簇上本来就没有名字，而
        ``keep_labels=True`` 让标注阶段也不去填 —— 实测 34/34 张卡片退化成占位名
        甚至空名。

        修法是让 ``label.py`` 按**数据**判断（簇上有没有名字），而不是按配置。
        """
        result = _miner_with_broken_llm().mine("防晒霜", notes_count=40)

        assert result.cards
        assert any("已降级为聚类模式" in note for note in result.notes)
        assert all(card.pain.label for card in result.cards), "降级后仍有簇没有名字"

    def test_all_unnamed_warns_against_picking_topics(self):
        """★ 所有痛点都没命名时，必须明确劝阻用户据此选题。

        降级到聚类后命名同样依赖 LLM；LLM 依然不可用时每个簇只剩占位名，而占位名
        生成不出方向标题（不变式 5），于是「方向」一列全是"待命名方向"。此时
        证据链仍有价值，但方向毫无意义 —— 不告知的话用户会以为真有几十个叫
        "待命名方向"的机会。
        """
        result = _miner_with_broken_llm().mine("防晒霜", notes_count=40)

        first = result.notes[0] if result.notes else ""
        assert "未能命名" in first, f"警告必须排在最前面，实际首条是: {first!r}"
        assert "请不要据此选题" in first

    def test_mine_announces_disabled_research(self):
        """关闭竞品调研必须留下痕迹 —— 静默关闭会让用户以为查过了。"""
        result = _miner().mine("防晒霜", notes_count=20)
        assert any("竞品调研已关闭" in note for note in result.notes)

    def test_disabled_research_never_claims_no_competitors(self):
        """★ 关闭调研时，空白度必须是中性值，不能是「查证过没有竞品」。

        这是不变式 3 点名的危险实例：把"没查成"当成"没有"。它会同时污染两处 ——
        机会分每张卡虚高 12.5 分（`competitor_gap` 从 0.5 变成 1.0），报告上还会
        印出一句"✅ 未发现竞品 —— 查证过，目前没有可查到的成熟实现"。

        这条测试是回归守卫：M1 交付时的确存在这个缺陷，且它出现在真实产物里。
        """
        result = _miner().mine("防晒霜", notes_count=20)

        assert result.cards
        for card in result.cards:
            gap = card.score_breakdown["competitor_gap"]
            assert gap == pytest.approx(0.5), (
                f"卡片「{card.title}」的竞品空白度是 {gap}，"
                "但这次运行根本没查过竞品 —— 它必须是中性值 0.5"
            )
            assert card.research_failed, f"卡片「{card.title}」未标记调研失败"
            assert card.research_status == "unsearchable", (
                f"卡片「{card.title}」的结论类别是 {card.research_status} —— "
                "没查过只能是 unsearchable（渲染层据此说「没查成」，而不是「没有竞品」）"
            )
            assert not card.competitors

    def test_mine_progress_callback_is_called(self):
        """进度回调按阶段推进，且比例单调不减。"""
        stages: list[tuple[str, float]] = []
        _miner().mine("防晒霜", notes_count=20, progress=lambda s, r: stages.append((s, r)))

        assert stages, "进度回调从未被调用"
        ratios = [ratio for _, ratio in stages]
        assert ratios == sorted(ratios), f"进度比例出现回退: {ratios}"
        assert ratios[-1] == 1.0

    def test_deep_without_pillow_does_not_break_text_analysis(self):
        """图片分析失败不该拖垮文本分析。"""
        result = _miner().mine("防晒霜", notes_count=20, deep=True)

        # 样例语料的图片是 synthetic:// 合成图，能正常走完；
        # 无论成功与否，文本分析的卡片都必须产出
        assert result.cards

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
    """``mine`` 子命令。

    这里只覆盖 CLI 层的参数处理与错误提示。**完整链路的跑通由
    :class:`TestPainMinerFacade` 的假组件测试负责** —— CLI 不支持注入假
    provider，而 CI 里没有 API Key。
    """

    def test_missing_api_key_gives_readable_error(self, runner: CliRunner):
        """没有 API Key 时必须**立刻**给出可操作的提示。

        必须是快速失败：先跑一遍采集再报缺 Key，用户会白等一场，还会误以为
        是采集环节出了问题。
        """
        result = runner.invoke(main, ["mine", "-k", "防晒霜", "--backend", "fixture", "-n", "5"])

        assert result.exit_code == 2, _combined(result)
        output = _combined(result)
        assert "API Key" in output
        assert "Traceback" not in output
        assert "采集完成" not in output, "缺 API Key 时不该先跑完采集"

    def test_rejects_unknown_weight_name(self, runner: CliRunner):
        """★ 拼错的权重名必须报错 —— 静默忽略会让用户以为调整生效了。"""
        result = runner.invoke(
            main,
            ["mine", "-k", "防晒霜", "--weights", "gapp=0.4", "--backend", "fixture"],
        )
        assert result.exit_code == 2
        assert "未知的权重名" in _combined(result)

    def test_rejects_malformed_weights(self, runner: CliRunner):
        result = runner.invoke(
            main,
            ["mine", "-k", "防晒霜", "--weights", "gap", "--backend", "fixture"],
        )
        assert result.exit_code == 2
        assert "key=value" in _combined(result)

    def test_rejects_non_numeric_weight(self, runner: CliRunner):
        result = runner.invoke(
            main,
            ["mine", "-k", "防晒霜", "--weights", "gap=高", "--backend", "fixture"],
        )
        assert result.exit_code == 2
        assert "不是数字" in _combined(result)

    def test_help_lists_cost_and_weight_options(self, runner: CliRunner):
        result = runner.invoke(main, ["mine", "--help"])
        assert result.exit_code == 0
        for flag in ("--deep", "--weights", "--no-save"):
            assert flag in result.output


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
        """``mine`` 会在多处回显关键词，每一处都必须转义。"""
        result = runner.invoke(main, ["mine", "-k", "[/]", "--backend", "fixture"])

        # 没有 API Key 会走到失败分支，但关键词必须先被完整打印出来
        assert "[/]" in _combined(result), "mine 的关键词未被转义"


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


class TestCliSummaryMatchesReport:
    """终端摘要必须与报告展示同一批卡片。

    噪声簇（长尾低频）的 ``label`` 是空的，会被 ``_direction_title`` 渲染成
    "待命名方向"；而它又常常因为 ``size`` 大而排在第一位 —— 于是终端的第一行
    变成「最大的机会：待命名方向」，而 HTML 报告里根本没有这张卡片（渲染层默认
    ``include_noise=False``）。两处不一致会让用户以为自己看漏了。
    """

    def test_noise_is_not_listed_as_a_card(self, monkeypatch: pytest.MonkeyPatch):
        """★ 噪声簇不得作为机会卡片出现在终端摘要里，但要如实告知条数。"""
        import io

        from rich.console import Console

        from xhs_pain_miner import cli as cli_module
        from xhs_pain_miner.models import MiningResult, OpportunityCard, PainCluster

        buffer = io.StringIO()
        monkeypatch.setattr(cli_module, "console", Console(file=buffer, width=200))

        noise = PainCluster(id="pain-noise", size=173, is_noise=True)
        real = PainCluster(id="pain-1", label="搓泥", size=237)
        result = MiningResult(
            keyword="防晒霜",
            cards=[
                OpportunityCard(id="c1", title="防晒霜 · 待命名方向", pain=noise, score=88.0),
                OpportunityCard(
                    id="c2", title="防晒霜 · 解决「搓泥」的工具", pain=real, score=80.0
                ),
            ],
            clusters=[noise, real],
        )

        cli_module._render_result(result)
        output = buffer.getvalue()

        assert "待命名方向" not in output, "噪声簇被当成机会卡片显示了"
        assert "解决「搓泥」的工具" in output, "真实卡片应当显示"
        assert "173" in output, "未归类的条数必须如实告知，不能悄悄丢掉"
        assert "1 张机会卡片" in output, "卡片数不该把噪声簇算进去"


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


# --------------------------------------------------------------------------- #
# 竞品调研端到端（打开调研 + 两个渠道的假传输层）
# --------------------------------------------------------------------------- #

Handler = Callable[[httpx.Request], httpx.Response]


def _github_repo(name: str, *, description: str = "解决防晒问题的工具") -> dict[str, Any]:
    return {
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "stargazers_count": 120,
        "pushed_at": "2026-08-01T00:00:00Z",
        "description": description,
    }


def _appstore_app(name: str) -> dict[str, Any]:
    return {
        "trackName": name,
        "trackViewUrl": f"https://apps.apple.com/cn/app/{name}",
        "userRatingCount": 900,
        "currentVersionReleaseDate": "2026-08-01T00:00:00Z",
        "description": "帮你解决这个问题的 App",
    }


@pytest.fixture
def research_channels(monkeypatch: pytest.MonkeyPatch):
    """把两个渠道的 HTTP 都接到假传输层（不联网、不等限速）。"""

    def install(*, github: Handler, appstore: Handler) -> None:
        monkeypatch.setattr(github_module, "_transport", httpx.MockTransport(github))
        monkeypatch.setattr(appstore_module, "_transport", httpx.MockTransport(appstore))

    monkeypatch.setattr(github_module, "_sleep", lambda _seconds: None)
    return install


class TestResearchEnabledEndToEnd:
    """★ 打开竞品调研的整条链路 —— 真正的接线验收。

    这里跑的是 ``mine()``：检索词生成 → 渠道分发 → 相关性判定 → 结论 → 卡片 →
    产物。前面那些单点用例守的是每一步的边界，这一组守的是它们**连起来**之后
    用户拿到的东西。
    """

    def test_cards_carry_competitors_from_both_channels(self, research_channels):
        research_channels(
            github=lambda request: httpx.Response(
                200,
                json={
                    "total_count": 2,
                    "items": [_github_repo(f"a/{request.url.params['q']}")],
                },
            ),
            appstore=lambda request: httpx.Response(
                200, json={"resultCount": 1, "results": [_appstore_app("成分查询助手")]}
            ),
        )
        llm = _FakeLLM(relevance_reply='{"relevant": [0, 1], "rejected": []}')
        result = _miner(llm=llm, research_enabled=True).mine("防晒霜", notes_count=20)

        assert result.cards
        for card in result.cards:
            assert card.research_status == "ok"
            assert {finding.source for finding in card.competitors} == {"github", "appstore"}
            assert card.score_breakdown["competitor_gap"] < 1.0

    def test_zero_hits_everywhere_never_claim_no_competitor(self, research_channels):
        """★ 0 命中（平台检索不到）不是"没有竞品" —— 空白度必须落回中性值。

        这是 M2 要修的那个假空白：M1 把 0 命中读成"查证过确实没有竞品"，
        给出空白度 1.0（机会分里最强的正面信号），用户于是去做一个实际很拥挤的
        方向。这里同时钉住分数、结论类别与告诉用户的那句话。
        """
        research_channels(
            github=lambda request: httpx.Response(200, json={"total_count": 0, "items": []}),
            appstore=lambda request: httpx.Response(200, json={"resultCount": 0, "results": []}),
        )
        result = _miner(llm=_FakeLLM(), research_enabled=True).mine("防晒霜", notes_count=20)

        assert result.cards
        for card in result.cards:
            assert card.research_status == "unsearchable"
            assert card.research_failed is True
            assert card.score_breakdown["competitor_gap"] == pytest.approx(0.5)
        assert any("检索不到" in note for note in result.notes)

    def test_unimplemented_channel_words_do_not_downgrade_the_conclusion(self, research_channels):
        """★ chrome / xhs 的词被跳过，且**没有**因此把结论压回中性值。

        假模型每次都给出 4 条词（其中 chrome 那条永远不该被发出）。若未接入的渠道
        也被当成"这次没查成"参与合并，每张卡片都会从"查证过没有竞品"（1.0）
        掉到中性值（0.5）—— 一个必然失败的渠道会系统性抹掉 M2 的正面信号。
        """
        research_channels(
            github=lambda request: httpx.Response(
                200, json={"total_count": 7, "items": [_github_repo("someone/books")]}
            ),
            appstore=lambda request: httpx.Response(
                200, json={"resultCount": 5, "results": [_appstore_app("无关应用")]}
            ),
        )
        llm = _FakeLLM(relevance_reply='{"relevant": [], "rejected": [0, 1]}')
        result = _miner(llm=llm, research_enabled=True).mine("防晒霜", notes_count=20)

        assert result.cards
        for card in result.cards:
            assert card.research_status == "no_competitor", (
                f"卡片「{card.title}」的结论是 {card.research_status} —— "
                "平台搜得到、只是不相关，这正是「查证过确实没有」"
            )
            assert card.score_breakdown["competitor_gap"] == 1.0
        assert not any("web clipper" in prompt for prompt in llm.prompts)

    def test_warning_text_stays_out_of_the_public_payload(self, research_channels):
        """★ 约束 6：判定失败的警告含 LLM 回复预览，绝不能跟着出网。"""
        research_channels(
            github=lambda request: httpx.Response(
                200, json={"total_count": 3, "items": [_github_repo("a/one")]}
            ),
            appstore=lambda request: httpx.Response(
                200, json={"resultCount": 1, "results": [_appstore_app("成分查询助手")]}
            ),
        )
        leak = "模型今天不想输出 JSON —— 这段是自由文本"
        result = _miner(llm=_FakeLLM(relevance_reply=leak), research_enabled=True).mine(
            "防晒霜", notes_count=20
        )

        assert any(leak in note for note in result.notes), "警告必须先出现在本地产物里"
        payload = json.dumps(result.to_public_dict(), ensure_ascii=False)
        assert leak not in payload
        assert "notes" not in result.to_public_dict()
        for card in result.cards:
            card_payload = json.dumps(card.to_public_dict(), ensure_ascii=False)
            assert leak not in card_payload
            assert "warning" not in card_payload
