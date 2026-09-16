"""降级路径的**名字**守卫（第二轮独立验证补）。

背景：P0 的修法是两处 —— ``label.py`` 的 ``_apply_label``（标注**成功**时按数据
判断要不要写名字）与 ``_degrade``（标注**失败**时按数据判断要不要保留已有名字）。
原来的回归测试只覆盖了 ``_degrade`` 那一半（用一个"完全不可用"的 LLM，让每个簇
都走降级分支），于是 ``_apply_label`` 那一半**可以整段改回旧实现而不触发任何测试
失败** —— 实测把 ``if not (keep_labels and cluster.label)`` 还原成
``if not keep_labels`` 后 891 个测试全绿，而"归纳失败 → 退回聚类 → 标注阶段 LLM
恢复"这条**生产中最可能的降级路径**会重新产出 48/48 个空名字的簇、49 张同名卡片。

那条路径与"LLM 彻底宕机"不是同一件事：归纳是一次调用（一次限流、一个坏 JSON 就会
失败），标注是逐个簇的另一次调用。前者失败而后者成功是完全现实的组合，也正是用户
最可能遇到的那一种 —— 报告看起来正常，只是所有卡片都没有方向。

因此本文件用"只在归纳阶段失败"的假 LLM 把那一半钉住。
"""

from __future__ import annotations

import json

from xhs_pain_miner import PainMiner
from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.base import LLMError, LLMResponse
from xhs_pain_miner.models import RunCost

_NAMES = ("搓泥", "假白", "闷痘", "难卸", "价格")
_NOTES = 60

_TAXONOMY_FAILURE = "mock: 归纳调用失败"


class _TaxonomyOnlyFailingLLM:
    """只在归纳阶段失败：标注阶段正常返回一个真名字。

    判定依据与项目其它替身一致：归纳阶段的提示词里含字面量 ``"pains"``。
    """

    name = "taxonomy-only-failing"

    def __init__(self) -> None:
        self.usage = RunCost()

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        prompt = messages[-1].content
        self.usage.llm_calls += 1
        if '"pains"' in prompt:
            raise LLMError(_TAXONOMY_FAILURE)
        name = next((n for n in ["其他", *_NAMES] if n in prompt), "其他")
        return LLMResponse(
            text=json.dumps(
                {
                    "label": name,
                    "summary": f"{name}相关抱怨。",
                    "category": "体验粗糙",
                    "sentiment": -0.8,
                    "stage": "growing",
                    "difficulty": 2,
                    "feasibility": "个人可做 / 1-2 周",
                },
                ensure_ascii=False,
            ),
            model="fake",
            input_tokens=1,
            output_tokens=1,
        )

    def close(self) -> None:
        pass


class _NGramEmbedder:
    """确定性假编码器（字符 2-gram 哈希）—— 不依赖模型权重，CI 里秒级。"""

    name = "fake-ngram"
    is_local = True

    def __init__(self, dimension: int = 192) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts, *, batch_size: int = 64):  # type: ignore[no-untyped-def]
        import hashlib

        vectors = []
        for text in texts:
            vector = [0.0] * self._dimension
            for index in range(max(len(text) - 1, 1)):
                gram = text[index : index + 2] or text[index:]
                digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vector[digest % self._dimension] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def close(self) -> None:
        pass


def _mine_with_taxonomy_failure(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    for key in ("LLM_API_KEY", "DB_PATH", "OUTPUT_DIR", "COLLECTOR_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(
        _env_file=None,
        collector_backend="fixture",
        research_enabled=False,
        db_path=str(tmp_path / "db" / "db.sqlite"),
        output_dir=str(tmp_path / "out"),
    )
    miner = PainMiner(
        settings=settings,
        llm=_TaxonomyOnlyFailingLLM(),
        vlm=_TaxonomyOnlyFailingLLM(),
        embedder=_NGramEmbedder(),
    )
    try:
        return miner.mine("防晒霜", notes_count=_NOTES)
    finally:
        miner.close()


def test_labeling_stage_fills_names_after_taxonomy_failure(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """★ 归纳失败退回聚类后，标注阶段必须把名字填上。

    这里断言的是**真名字**（非空、且不是 ``<未命名痛点 #N>`` 占位名）——
    只断言"非空"是抓不到旧实现的：旧实现留下的是空字符串，而占位名也是非空的。
    """
    result = _mine_with_taxonomy_failure(tmp_path, monkeypatch)

    assert any("已降级为聚类模式" in note for note in result.notes), "降级说明缺失"

    clusters = [c for c in result.clusters if not c.is_noise]
    assert clusters, "降级后应当仍有痛点簇"
    blank = [c.label for c in clusters if not c.label.strip()]
    assert not blank, f"这些簇仍然没有名字（P0 的症状）：{blank[:5]}"
    placeholder = [c.label for c in clusters if c.label.startswith("<")]
    assert not placeholder, f"标注阶段已恢复却仍在用占位名：{placeholder[:5]}"


def test_no_card_claims_to_be_an_unnamed_direction(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """★ 标注阶段恢复后，卡片不得退化成「待命名方向」。"""
    result = _mine_with_taxonomy_failure(tmp_path, monkeypatch)

    assert result.cards
    pending = [card.title for card in result.cards if "待命名方向" in card.title]
    assert not pending, f"仍有卡片没有方向：{pending[:5]}"

    unnamed_warning = [note for note in result.notes if "未能命名" in note]
    assert not unnamed_warning, (
        "标注阶段已经恢复了真名字，却仍然打印了「全部痛点都未能命名」的劝阻警告"
        f"（误报）：{unnamed_warning}"
    )
