"""门面层的 VLM 缓存接线测试。

``pipeline/vlm.py`` 实现了「省钱七条」的第 4 条（结果按图片内容哈希缓存到
SQLite），但那条策略只有在 :class:`~xhs_pain_miner.pain_miner.PainMiner` 把
:class:`~xhs_pain_miner.pipeline.vlm.SqliteVlmCache` **注入**
:class:`~xhs_pain_miner.pipeline.vlm.VlmAnalyzer` 之后才真的生效。

缺了 ``cache=`` 那一行时，一切都还是「看起来正常」的：报告照出、卡片照产、
成本摘要里也只有真实发生的调用 —— 唯一的变化是第二次跑同一个品类要**全额重复
计费**，而模块文档明明白白写着成本近乎归零。这类缺陷不会被任何"跑得通"的测试
发现，只能靠在门面这层断言「调用次数」「落盘位置」「生命周期」「降级」。

因此本文件**不重复** vlm.py 自身的缓存逻辑测试（那是 ``test_vlm.py`` 的职责），
只测门面到缓存这条线有没有接上。

全部离线：语料用内置 fixture，图片是 ``synthetic://`` 合成图，LLM 与编码器都是
假实现 —— 不联网、不需要 API Key。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xhs_pain_miner import PainMiner
from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.base import LLMResponse
from xhs_pain_miner.models import MiningResult, RunCost

_ENV_KEYS = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "COLLECTOR_BACKEND",
    "MAX_NOTES",
    "RESEARCH_ENABLED",
    "DB_PATH",
    "OUTPUT_DIR",
)

_NOTES = 12
"""每次运行的笔记数：够产生几十张图，又不至于让 CI 变慢。"""

_VLM_REPLY = '{"description": "一张对比图", "pain_hints": ["宣传图与实际不符"]}'

_LABEL_JSON = (
    '{"label": "防晒霜搓泥", "summary": "上妆后起白条。", "category": "体验粗糙", '
    '"sentiment": -0.8, "stage": "growing", "difficulty": 2, '
    '"feasibility": "个人可做 / 1-2 周"}'
)

_TAXONOMY_NAMES = ("搓泥", "假白", "闷痘", "难卸", "价格")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离环境变量 —— 开发机上的真实配置（API Key、后端、库路径）会让断言失真。"""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class _FakeVision:
    """假 VLM provider —— 只数调用次数，不联网。"""

    name = "fake-vision"

    def __init__(self) -> None:
        self.usage = RunCost()
        self.call_count = 0

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("VLM 链路不该调用文本补全接口")

    def complete_vision(self, prompt, images, **kwargs):  # type: ignore[no-untyped-def]
        self.call_count += 1
        self.usage.vlm_calls += 1
        return LLMResponse(text=_VLM_REPLY, model="fake-vision")

    def close(self) -> None:
        pass


class _FakeTextLLM:
    """假文本 LLM —— 按提示词类型返回归纳结果或标注结果。

    与 ``test_pain_miner.py`` 的同名替身保持同样的契约：提示词里要求
    ``pains`` 时返回归纳清单，否则返回标注结果。文案与形状必须与真实提示词一致，
    否则测的就不是"缓存接线"而是"假模型会不会被解析"。
    """

    name = "fake-text"

    def __init__(self) -> None:
        self.usage = RunCost()

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        prompt = messages[-1].content
        self.usage.llm_calls += 1
        self.usage.llm_input_tokens += 100
        self.usage.llm_output_tokens += 50
        return LLMResponse(
            text=_taxonomy_reply(prompt) if '"pains"' in prompt else _LABEL_JSON,
            model="fake",
            input_tokens=100,
            output_tokens=50,
        )

    def close(self) -> None:
        pass


class _FakeEmbedder:
    """确定性假编码器 —— 不加载模型，也不联网。"""

    name = "fake"
    is_local = True

    def __init__(self, dimension: int = 12) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts, *, batch_size: int = 64):  # type: ignore[no-untyped-def]
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


def _taxonomy_reply(prompt: str) -> str:
    """按提示词里的样本条数生成归纳结果（``labels`` 长度必须严格等于样本数）。"""
    sample_lines = [line for line in prompt.splitlines() if line.startswith("[")]
    labels = [
        next((name for name in _TAXONOMY_NAMES if name in line), "其他") for line in sample_lines
    ]
    pains = [
        {"name": name, "summary": f"{name}相关抱怨。", "category": "体验粗糙"}
        for name in _TAXONOMY_NAMES
    ]
    return json.dumps({"pains": pains, "labels": labels}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _settings(**overrides: object) -> Settings:
    """构造不读取 ``.env`` 的配置对象。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _miner(tmp_path: Path, **overrides: object) -> tuple[PainMiner, _FakeVision]:
    """构造一台全程离线的 PainMiner，返回 ``(miner, 假 VLM)``。

    缓存默认落在 ``tmp_path`` 下的嵌套目录里 —— 父目录**不存在**，因此这同时
    覆盖了「库使用者没有调用过 ``Settings.ensure_dirs()``」的情形。
    """
    settings = _settings(
        **{
            "collector_backend": "fixture",
            "research_enabled": False,
            "db_path": tmp_path / "state" / "db.sqlite",
            **overrides,
        }
    )
    vlm = _FakeVision()
    return (
        PainMiner(settings=settings, llm=_FakeTextLLM(), vlm=vlm, embedder=_FakeEmbedder()),
        vlm,
    )


def _reinject(miner: PainMiner) -> _FakeVision:
    """重新注入离线替身，返回新的假 VLM。

    :meth:`PainMiner.close` 的既有语义是「释放它持有的全部对象，包括调用方注入的
    那些」，所以关闭之后再跑一次必须先重新注入 —— 这不是本次修复引入的语义，
    测试只是顺着它走。这里用私有属性而非构造参数，是因为要验的恰恰是
    **同一个实例**在 close 之后重新构造函数与缓存。
    """
    vlm = _FakeVision()
    miner._llm = _FakeTextLLM()
    miner._vlm = vlm
    miner._embedder = _FakeEmbedder()
    return vlm


def _evidence_count(result: MiningResult) -> int:
    """全部痛点簇的证据条数之和。"""
    return sum(len(cluster.evidences) for cluster in result.clusters)


# --------------------------------------------------------------------------- #
# 缓存接线
# --------------------------------------------------------------------------- #


class TestCacheIsWired:
    """缓存必须真的被注入分析器，并且真的省钱。"""

    def test_analyzer_receives_a_cache(self, tmp_path: Path):
        """最直接的接线断言：分析器手里必须有一个缓存。

        ``VlmAnalyzer.cache`` 默认是 ``None``（不缓存），少一个 ``cache=``
        参数不会报错、不会降级、也不会在产物上留下任何痕迹。
        """
        miner, _ = _miner(tmp_path)
        assert miner._get_vlm_analyzer().cache is not None

    def test_second_run_makes_no_vlm_calls(self, tmp_path: Path):
        """★ 省钱第 4 条：第二次跑同一品类，同样的图一张都不再调用。

        「第二次运行」用**两个 PainMiner 实例**模拟（真实场景就是另一次运行 /
        另一个进程）：它们只共享 ``settings.db_path`` 上的那个缓存文件。
        每次都必须是新的假 VLM，才能从调用次数上看出这一次到底花了多少。
        """
        first_miner, first_vlm = _miner(tmp_path)
        try:
            corpus = first_miner.collect("防晒霜", limit=_NOTES)
            assert corpus.total_images > 0, "样例语料没有图片，缓存测试会变成恒真断言"
            assert all(
                url.startswith("synthetic://") for note in corpus.notes for url in note.images
            ), "样例图片应当是合成图 —— 这条测试必须保持离线"
            first = first_miner.mine("防晒霜", corpus=corpus, deep=True)
        finally:
            first_miner.close()

        assert first_vlm.call_count > 0, "第一次运行没有产生任何 VLM 调用，缓存测试会变成恒真断言"
        assert not any("缓存不可用" in note for note in first.notes)

        second_miner, second_vlm = _miner(tmp_path)
        try:
            second = second_miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        finally:
            second_miner.close()

        assert second_vlm.call_count == 0, "第二次运行仍在发起 VLM 调用 —— 缓存没有被接上"
        assert second.cost.vlm_calls == 0
        assert second.cost.vlm_calls < first.cost.vlm_calls, "第二次运行的成本没有下降"
        # 缓存命中不能是"少算了"：图片证据照样要进入分析。
        assert second.cards
        assert _evidence_count(second) == _evidence_count(first)

    def test_cache_file_lands_on_settings_db_path(self, tmp_path: Path):
        """缓存文件必须落在 ``settings.db_path``，并真的写入了记录。"""
        db_path = tmp_path / "state" / "nested" / "db.sqlite"
        assert not db_path.parent.exists(), "父目录应当不存在，否则测不到自动建目录"

        miner, vlm = _miner(tmp_path, db_path=db_path)
        miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        miner.close()

        assert db_path.is_file(), f"缓存文件没有落在 {db_path}"
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM vlm_cache").fetchone()[0]
        # 每张成功分析的图写一行，行数即本次真实花掉的调用数。
        assert rows == vlm.call_count > 0

    def test_estimate_and_analyze_share_one_cache(self, tmp_path: Path):
        """CLI 的真实顺序是「先预估、后分析」，两者必须看到同一份缓存。"""
        miner, vlm = _miner(tmp_path)
        corpus = miner.collect("防晒霜", limit=_NOTES)

        before = miner.estimate_vlm_cost(corpus)
        assert before.cached_images == 0
        assert before.planned_calls > 0
        assert vlm.call_count == 0, "成本预估不得产生任何 VLM 调用"

        miner.mine("防晒霜", corpus=corpus, deep=True)

        after = miner.estimate_vlm_cost(miner.collect("防晒霜", limit=_NOTES))
        assert after.cached_images == before.planned_calls
        assert after.planned_calls == 0


class TestCacheLifecycle:
    """缓存的持有与释放。"""

    def test_mine_works_after_close(self, tmp_path: Path):
        """★ close() 之后必须能安全地再跑一次。

        实现要点：close() 关掉连接后要**清空引用**，让下一次运行重新构造缓存。
        否则第二次拿到的是一根已经关掉的 SQLite 连接，直接抛
        "Cannot operate on a closed database"。
        """
        miner, vlm = _miner(tmp_path)
        miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        assert vlm.call_count > 0

        miner.close()

        fresh_vlm = _reinject(miner)
        result = miner.mine("防晒霜", notes_count=_NOTES, deep=True)

        assert result.cards
        assert not any("缓存不可用" in note for note in result.notes), "重开缓存失败了"
        # 上一次运行的缓存内容仍在磁盘上：重开之后应当全部命中。
        assert fresh_vlm.call_count == 0, "关闭后重开的缓存没有命中"

    def test_close_is_idempotent(self, tmp_path: Path):
        """重复 close 不得抛异常，也不得在第二次之后留下悬空引用。"""
        miner, _ = _miner(tmp_path)
        miner.mine("防晒霜", notes_count=_NOTES, deep=True)

        miner.close()
        miner.close()

        assert miner._vlm_cache is None


class TestCacheDegradation:
    """缓存打不开时必须降级，而不是让整次分析崩掉。"""

    @pytest.fixture(params=["parent-is-a-file", "path-is-a-directory"])
    def broken_db_path(self, request: pytest.FixtureRequest, tmp_path: Path) -> Path:
        """两种让缓存打不开的真实情形。

        * ``parent-is-a-file`` —— ``db_path`` 的父路径被一个普通文件占住
          （``mkdir`` 抛 ``OSError``）。
        * ``path-is-a-directory`` —— ``db_path`` 本身是一个已存在的目录
          （SQLite 抛 ``sqlite3.Error``）。
        """
        if request.param == "parent-is-a-file":
            blocker = tmp_path / "not-a-dir"
            blocker.write_text("我不是目录", encoding="utf-8")
            return blocker / "db.sqlite"
        target = tmp_path / "i-am-a-directory"
        target.mkdir()
        return target

    def test_unusable_cache_degrades_to_no_cache(self, tmp_path: Path, broken_db_path: Path):
        """★ 缓存失败 → 退化为"不缓存" + 一条警告，而不是抛异常。"""
        miner, vlm = _miner(tmp_path, db_path=broken_db_path)

        result = miner.mine("防晒霜", notes_count=_NOTES, deep=True)

        assert result.cards, "缓存打不开不该拖垮分析"
        assert vlm.call_count > 0, "降级应当是'不缓存'，而不是'跳过图片分析'"
        warnings = [note for note in result.notes if "缓存不可用" in note]
        assert warnings, f"缓存不可用时没有在产物里留下警告：{result.notes}"
        assert str(broken_db_path) in warnings[0], "警告里要能看出是哪个路径出了问题"

    def test_estimate_vlm_cost_survives_unusable_cache(self, tmp_path: Path, broken_db_path: Path):
        """CLI 在 ``mine`` 之前就会调用预估 —— 这里也必须不抛异常。"""
        miner, vlm = _miner(tmp_path, db_path=broken_db_path)
        corpus = miner.collect("防晒霜", limit=4)

        estimate = miner.estimate_vlm_cost(corpus)

        assert estimate.planned_calls > 0
        assert vlm.call_count == 0

    def test_working_cache_produces_no_warning(self, tmp_path: Path):
        """反向守卫：正常路径不得留下"缓存不可用"的噪声警告。"""
        miner, _ = _miner(tmp_path)
        result = miner.mine("防晒霜", notes_count=6, deep=True)

        assert not [note for note in result.notes if "缓存" in note]

    def test_warning_repeats_on_every_run(self, tmp_path: Path, broken_db_path: Path):
        """★ 缓存不可用的警告必须在**每次**运行都出现，不能被"取走"。

        缓存只在构造分析器时尝试建立一次、失败后不重试 —— 若警告在第一次交付后
        就被清空，第二次运行会静默，而"这次仍然没有缓存、仍然会重复计费"这个
        事实并没有变。**成本异常静默化比重复提示更糟**：用户会以为缓存恢复了。
        """
        miner, _ = _miner(tmp_path, db_path=broken_db_path)

        first = miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        second = miner.mine("防晒霜", notes_count=_NOTES, deep=True)

        assert [note for note in first.notes if "缓存不可用" in note], "第一次运行缺警告"
        assert [note for note in second.notes if "缓存不可用" in note], (
            "第二次运行静默了 —— 警告被取走后没有再交付"
        )

    def test_close_does_not_accumulate_warnings(self, tmp_path: Path, broken_db_path: Path):
        """★ ``close()`` 后重新运行不得让警告**累积**。

        close() 会清掉缓存的引用，于是下一次 ``mine(deep=True)`` 会重新构造缓存、
        再追加一条**同样**的警告。不一起清空的话，第二次运行会带 2 条、第三次 3 条
        —— 而它们说的是同一件事，用户读到的是噪声而不是信息。
        """
        miner, _ = _miner(tmp_path, db_path=broken_db_path)

        miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        miner.close()
        _reinject(miner)
        second = miner.mine("防晒霜", notes_count=_NOTES, deep=True)
        miner.close()
        _reinject(miner)
        third = miner.mine("防晒霜", notes_count=_NOTES, deep=True)

        counts = [
            len([note for note in result.notes if "缓存不可用" in note])
            for result in (second, third)
        ]
        assert counts == [1, 1], f"警告在 close 后累积了：{counts}"
