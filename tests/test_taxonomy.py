"""痛点归集（taxonomy）测试 —— **M1 的默认生产路径**。

为什么这个文件必须存在
----------------------
M1 中途把痛点归集从 HDBSCAN 聚类换成了「LLM 归纳 + embedding 分类」，
`pipeline/taxonomy.py` 与 `cluster.group_by_taxonomy` 成了**默认路径**，
但换方案时没有同步补测试。独立验证做过变异测试：把 `size` 改成
``len(evidences)``（封顶 20）、强制走聚类路径、`keep_labels=False`、
标注阶段改写 `size` —— 这 4 个**直接改坏默认路径**的变异，全部 174 个测试
跑下来一个都没变红。

这个文件补上那些洞。测试全部离线：LLM 用假 provider，向量是手工构造的
确定性向量（这样"哪几条该分到哪个痛点"完全可控）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from xhs_pain_miner.llm.base import LLMError, LLMResponse
from xhs_pain_miner.models import RunCost, TextUnit
from xhs_pain_miner.pipeline.cluster import group_by_taxonomy
from xhs_pain_miner.pipeline.taxonomy import (
    DEFAULT_MATCH_THRESHOLD,
    MIN_PAINS,
    PainTaxonomy,
    TaxonomyEntry,
    assign_units,
    build_centroids,
    build_prompt,
    build_taxonomy,
    classify,
    parse_taxonomy_response,
    refine_centroids,
    sample_units,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class FakeLLM:
    """按预设内容返回的假 LLM。"""

    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.usage = RunCost()
        self.calls = 0

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.usage.llm_calls += 1
        return LLMResponse(text=self.reply, model="fake")

    def complete_vision(self, prompt, images, **kwargs):  # type: ignore[no-untyped-def]
        return LLMResponse(text="{}", model="fake")

    def close(self) -> None:
        pass


class FailingLLM(FakeLLM):
    """总是抛错的假 LLM。"""

    def __init__(self) -> None:
        super().__init__("")

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise LLMError("模拟服务不可用")


def make_units(count: int, *, prefix: str = "u", likes: int = 0, source: str = "comment"):
    """造一批文本单元。"""
    return [
        TextUnit(text=f"{prefix}{index}", source=source, likes=likes)  # type: ignore[arg-type]
        for index in range(count)
    ]


def reply_with(pains: list[str], labels: list[str]) -> str:
    """拼一个合法的归纳回复。"""
    return json.dumps(
        {
            "pains": [
                {"name": name, "summary": f"{name}的问题", "category": "体验"} for name in pains
            ],
            "labels": labels,
        },
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #


class TestSampleUnits:
    """样本挑选 —— 清单的覆盖度直接由它决定。"""

    def test_returns_all_when_size_exceeds_total(self):
        assert sample_units(make_units(5), size=10) == list(range(5))

    def test_returns_ordered_indices(self):
        picked = sample_units(make_units(50), size=10)
        assert picked == sorted(picked)
        assert len(picked) == len(set(picked)), "抽样不该重复取同一条"

    def test_is_reproducible(self):
        """同一份语料两次抽样必须一致 —— 否则同一份语料跑两次得到不同清单。"""
        units = make_units(100)
        assert sample_units(units, size=20, seed=0) == sample_units(units, size=20, seed=0)

    def test_zero_or_negative_size_returns_empty(self):
        assert sample_units(make_units(10), size=0) == []
        assert sample_units(make_units(10), size=-3) == []

    def test_stratifies_between_notes_and_comments(self):
        """笔记与评论都要有代表 —— 评论数量占优时会把笔记完全挤出去。"""
        notes = make_units(10, prefix="n", source="note")
        comments = make_units(100, prefix="c", source="comment")
        picked = sample_units(notes + comments, size=20)

        sources = {(notes + comments)[i].source for i in picked}
        assert sources == {"note", "comment"}, f"抽样未覆盖两类来源: {sources}"

    def test_prefers_high_likes(self):
        """高赞（= 更多人有同感）的发言应更可能入选。"""
        units = [TextUnit(text=f"u{i}", source="comment", likes=i * 100) for i in range(50)]
        picked = sample_units(units, size=10)
        assert min(units[i].likes for i in picked) > 0


# --------------------------------------------------------------------------- #
# 提示词与解析
# --------------------------------------------------------------------------- #


class TestBuildPrompt:
    def test_numbers_every_sample(self):
        """每条样本前面必须有编号：模型靠它对齐 labels，错位会让整批质心建错。"""
        units = make_units(3)
        prompt = build_prompt(units, [0, 1, 2], max_pains=20)

        for position in range(3):
            assert f"[{position}]" in prompt

    def test_declares_sample_count(self):
        prompt = build_prompt(make_units(7), list(range(7)), max_pains=20)
        assert "7" in prompt


class TestParseTaxonomyResponse:
    def test_parses_pains_and_labels(self):
        entries, labels = parse_taxonomy_response(
            reply_with(["搓泥", "假白"], ["搓泥", "假白", "其他"]), sample_count=3
        )
        assert [entry.name for entry in entries] == ["搓泥", "假白"]
        assert labels == ["搓泥", "假白", ""]

    def test_accepts_code_fenced_json(self):
        payload = reply_with(["搓泥", "假白"], ["搓泥"])
        entries, _ = parse_taxonomy_response(f"```json\n{payload}\n```", sample_count=1)
        assert len(entries) == 2

    def test_unknown_label_becomes_unmatched(self):
        """模型自造的、不在清单里的标签必须丢弃。

        留着一个查不到的标签，会让这条文本既不属于任何痛点、又占着一个
        `size` 名额 —— 而那个痛点的属性在清单里查不到。
        """
        _, labels = parse_taxonomy_response(
            reply_with(["搓泥", "假白"], ["搓泥", "凭空捏造"]), sample_count=2
        )
        assert labels == ["搓泥", ""]

    def test_short_labels_are_padded_not_guessed(self):
        """labels 少几条时补空，**不要猜** —— 猜错会让某个痛点的 size 凭空变大。"""
        _, labels = parse_taxonomy_response(reply_with(["搓泥", "假白"], ["搓泥"]), sample_count=4)
        assert labels == ["搓泥", "", "", ""]

    def test_long_labels_are_truncated(self):
        _, labels = parse_taxonomy_response(
            reply_with(["搓泥", "假白"], ["搓泥"] * 5), sample_count=2
        )
        assert len(labels) == 2

    def test_duplicate_pain_names_are_dropped(self):
        entries, _ = parse_taxonomy_response(reply_with(["搓泥", "搓泥"], ["搓泥"]), sample_count=1)
        assert [entry.name for entry in entries] == ["搓泥"]

    def test_respects_max_pains(self):
        names = [f"痛点{i}" for i in range(30)]
        entries, _ = parse_taxonomy_response(reply_with(names, []), sample_count=0, max_pains=5)
        assert len(entries) == 5

    @pytest.mark.parametrize(
        "payload",
        [
            "不是 JSON",
            "[]",
            '{"pains": []}',
            '{"pains": "字符串"}',
            '{"labels": ["a"]}',
            '{"pains": [{"name": ""}]}',
            '{"pains": [{"summary": "没有名字"}]}',
        ],
    )
    def test_rejects_malformed_payloads(self, payload: str):
        with pytest.raises(LLMError):
            parse_taxonomy_response(payload, sample_count=1)


# --------------------------------------------------------------------------- #
# 质心与分类
# --------------------------------------------------------------------------- #


def axis(dimension: int, position: int) -> list[float]:
    """构造一个单位基向量。"""
    vector = [0.0] * dimension
    vector[position] = 1.0
    return vector


class TestBuildCentroids:
    def test_centroid_is_normalized(self):
        """质心是平均出来的，必须重新归一化 —— 否则相似度会随样本数缩放。"""
        vectors = [axis(3, 0), axis(3, 0)]
        centroids = build_centroids(vectors, {0: "A", 1: "A"})
        norm = sum(value * value for value in centroids["A"]) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_averages_members(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        centroids = build_centroids(vectors, {0: "A", 1: "A"})
        expected = 0.5**0.5
        assert centroids["A"][0] == pytest.approx(expected)
        assert centroids["A"][1] == pytest.approx(expected)

    def test_respects_min_members(self):
        vectors = [axis(2, 0), axis(2, 1)]
        centroids = build_centroids(vectors, {0: "A", 1: "B"}, min_members=2)
        assert centroids == {}

    def test_skips_out_of_range_indices(self):
        assert build_centroids([axis(2, 0)], {5: "A"}) == {}


class TestClassify:
    def test_assigns_to_nearest_centroid(self):
        vectors = [axis(3, 0), axis(3, 1)]
        centroids = {"A": axis(3, 0), "B": axis(3, 1)}

        result = classify(vectors, centroids, threshold=0.0)

        assert result.labels == ["A", "B"]

    def test_below_threshold_is_unmatched(self):
        """低于阈值判为未命中 —— 硬塞给最近的痛点会让 size 虚高。"""
        # 与两个质心都正交，相似度为 0
        vectors = [axis(3, 2)]
        centroids = {"A": axis(3, 0), "B": axis(3, 1)}

        result = classify(vectors, centroids, threshold=0.5)

        assert result.labels == [""]
        assert result.similarities == [pytest.approx(0.0)]

    def test_empty_centroids_yields_all_unmatched(self):
        result = classify([axis(2, 0)], {}, threshold=0.0)
        assert result.labels == [""]

    def test_sizes_counts_members(self):
        vectors = [axis(2, 0), axis(2, 0), axis(2, 1)]
        centroids = {"A": axis(2, 0), "B": axis(2, 1)}

        assert classify(vectors, centroids, threshold=0.0).sizes() == {"A": 2, "B": 1}

    def test_unmatched_property(self):
        vectors = [axis(3, 2), axis(3, 0)]
        centroids = {"A": axis(3, 0)}
        result = classify(vectors, centroids, threshold=0.5)
        assert result.unmatched == 1


class TestRefineCentroids:
    def test_recomputes_from_all_members(self):
        vectors = [axis(2, 0), [0.6, 0.8]]
        refined = refine_centroids(vectors, ["A", "A"], min_members=2)

        assert "A" in refined
        norm = sum(value * value for value in refined["A"]) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_ignores_small_groups(self):
        assert refine_centroids([axis(2, 0)], ["A"], min_members=2) == {}

    def test_ignores_unmatched(self):
        assert refine_centroids([axis(2, 0), axis(2, 1)], ["", ""]) == {}


# --------------------------------------------------------------------------- #
# 归集（一次完整调用）
# --------------------------------------------------------------------------- #


class TestBuildTaxonomy:
    def test_returns_entries_and_labels(self):
        units = make_units(20)
        provider = FakeLLM(reply_with(["搓泥", "假白", "闷痘", "难卸", "价格"], ["搓泥"] * 20))

        taxonomy, labeled, warnings = build_taxonomy(units, provider=provider)

        assert len(taxonomy) == 5
        assert set(labeled.values()) == {"搓泥"}
        assert warnings == []

    def test_warns_when_most_samples_unmatched(self):
        units = make_units(20)
        provider = FakeLLM(reply_with(["搓泥", "假白", "闷痘", "难卸", "价格"], [""] * 20))

        _, labeled, warnings = build_taxonomy(units, provider=provider)

        assert labeled == {}
        assert any("无法归入" in warning for warning in warnings)

    def test_rejects_too_few_pains(self):
        """清单少于 MIN_PAINS 说明模型把不同问题合并了 —— 那无法指导产品决策。"""
        units = make_units(20)
        provider = FakeLLM(reply_with(["体验不好"], [""] * 20))

        with pytest.raises(LLMError, match="合并"):
            build_taxonomy(units, provider=provider)

    def test_raises_on_empty_units(self):
        with pytest.raises(LLMError):
            build_taxonomy([], provider=FakeLLM(reply_with([], [])))


class TestAssignUnits:
    """★ 默认生产路径的端到端行为。"""

    def build(self, *, reply: str | None = None):
        """三个痛点各 4 条，向量各自指向一个独立方向。"""
        names = ["搓泥", "假白", "闷痘", "难卸", "价格"]
        units = make_units(15)
        vectors = [axis(5, index // 3) for index in range(15)]
        default_reply = reply_with(names, [names[min(index // 3, 4)] for index in range(15)])
        return units, vectors, names, FakeLLM(reply or default_reply)

    def test_assigns_every_unit(self):
        units, vectors, _names, provider = self.build()

        _taxonomy, result, _warnings = assign_units(units, vectors, provider=provider)

        assert len(result.labels) == len(units)
        assert result.unmatched == 0

    def test_size_comes_from_classification_not_evidence_cap(self):
        """★ `size` 必须是分类命中的条数，不是证据条数（后者封顶 20）。

        独立验证曾把 size 改成 ``len(evidences)`` 而**全测试通过** —— 那会让
        大于 20 条的痛点全部显示成 20，而提及次数是本产品的核心指标。
        """
        names = ["搓泥", "假白", "闷痘", "难卸", "价格"]
        units = make_units(100)
        # 全部指向同一个方向 → 全部归到第一个痛点
        vectors = [axis(5, 0) for _ in range(100)]
        provider = FakeLLM(reply_with(names, ["搓泥"] * 100))

        taxonomy, result, _ = assign_units(units, vectors, provider=provider)
        clusters = group_by_taxonomy(units, result.labels, taxonomy=taxonomy)

        target = next(c for c in clusters if c.label == "搓泥")
        assert target.size == 100, "size 被证据上限截断了"
        assert len(target.evidences) <= 20, "证据应当仍受展示上限约束"

    def test_missing_centroid_falls_back_to_name_vectors(self):
        """★ 没有打标样本的痛点必须用痛点名向量兜底。

        否则分类会把**全部**文本塞给少数几个有质心的痛点，产出「搓泥，1142 条
        提及」这种看着正常却毫无意义的数字。
        """
        names = ["搓泥", "假白", "闷痘", "难卸", "价格"]
        units = make_units(15)
        vectors = [axis(5, index % 5) for index in range(15)]
        # 只给第一个痛点打标 → 其余四个没有样本质心
        provider = FakeLLM(reply_with(names, ["搓泥"] * 3 + [""] * 12))

        def encode(texts):
            return [axis(5, index) for index, _ in enumerate(texts)]

        taxonomy, result, warnings = assign_units(units, vectors, provider=provider, encode=encode)

        assert any("痛点名" in warning for warning in warnings)
        # 五个痛点都该有质心，因此不该出现"全部分给第一个"的情况
        assert len(taxonomy) == 5
        assert result.labels.count("搓泥") < len(units)

    def test_warns_when_centroids_are_few_and_no_encoder(self):
        names = ["搓泥", "假白", "闷痘", "难卸", "价格"]
        units = make_units(15)
        vectors = [axis(5, index % 5) for index in range(15)]
        provider = FakeLLM(reply_with(names, ["搓泥"] * 3 + [""] * 12))

        _, _, warnings = assign_units(units, vectors, provider=provider)

        assert any("质心" in warning for warning in warnings)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="长度不一致"):
            assign_units(make_units(3), [axis(5, 0)], provider=FakeLLM("{}"))

    def test_raises_when_no_labeled_samples(self):
        names = ["搓泥", "假白", "闷痘", "难卸", "价格"]
        units = make_units(15)
        vectors = [axis(5, index % 5) for index in range(15)]
        provider = FakeLLM(reply_with(names, [""] * 15))

        with pytest.raises(LLMError, match="质心"):
            assign_units(units, vectors, provider=provider)

    def test_threshold_can_reject_everything(self):
        units, vectors, _names, provider = self.build()

        _, result, _ = assign_units(units, vectors, provider=provider, threshold=1.1)

        assert result.unmatched == len(units)


class TestEncodeFallbackContract:
    """★ ``encode`` 兜底的契约：向量与痛点名必须**一一对应**。

    修复前的实现是 ``zip(missing, encode(missing))`` —— 返回的条数少于缺失痛点数
    时 ``zip`` 会**静默截断**，于是一部分痛点根本没有质心，而警告仍然声称它们
    "已改用痛点名本身的向量作为质心"。这是谎报：兜底没生效，用户却以为生效了。

    本类的契约（实现选择抛错，理由见 ``assign_units`` 里的注释）：条数不符时
    直接失败，既不猜也不部分采用。
    """

    names = ["搓泥", "假白", "闷痘", "难卸", "价格"]

    def setup_case(self, labeled: list[str] | None = None):
        """只给第一个痛点打标 → 其余四个没有样本质心，需要兜底。"""
        units = make_units(15)
        vectors = [axis(5, index % 5) for index in range(15)]
        labels = labeled if labeled is not None else ["搓泥"] * 3 + [""] * 12
        return units, vectors, FakeLLM(reply_with(self.names, labels))

    def test_short_encode_raises_instead_of_silently_truncating(self):
        """★ 少返一条必须报错 —— 修复前这里会静默截断并谎报"全部补上了"。"""
        units, vectors, provider = self.setup_case()

        def short_encode(texts: Sequence[str]) -> list[list[float]]:
            return [axis(5, index) for index, _ in enumerate(texts[:-1])]

        with pytest.raises(ValueError) as excinfo:
            assign_units(units, vectors, provider=provider, encode=short_encode)

        message = str(excinfo.value)
        assert "encode" in message
        assert "3" in message and "4" in message, "报错必须说清实际返回几个、需要几个"

    def test_extra_encode_raises_too(self):
        """多返同样是对应关系已破 —— zip 会静默丢掉多出来的向量。"""
        units, vectors, provider = self.setup_case()

        def long_encode(texts: Sequence[str]) -> list[list[float]]:
            return [axis(5, index % 5) for index in range(len(texts) + 1)]

        with pytest.raises(ValueError, match="encode"):
            assign_units(units, vectors, provider=provider, encode=long_encode)

    def test_empty_encode_result_raises(self):
        """返回空列表 —— 一个都没补上，更不能报告"已改用痛点名向量"。"""
        units, vectors, provider = self.setup_case()

        with pytest.raises(ValueError, match="encode"):
            assign_units(units, vectors, provider=provider, encode=lambda texts: [])

    def test_exact_length_backfills_every_missing_pain_and_says_so(self):
        """条数正确时兜底照常生效，且**每个**缺失的痛点都真的建了质心。

        断言刻意写成"每个痛点都分到了文本"，而不是"搓泥没有独占全部文本"——
        后者太弱：只补第一个痛点也会让"搓泥"不再独占（其余的文本会被挤给那个补上的
        质心），于是错误实现照样通过。而"只补一部分"正是这条兜底要防的事：
        没补上的痛点会被静默挤掉，它的 ``size`` 归零，产物看起来却完全正常。
        """
        units, vectors, provider = self.setup_case()
        calls: list[list[str]] = []

        def encode(texts: Sequence[str]) -> list[list[float]]:
            calls.append(list(texts))
            # 每个缺失的痛点给一个互不相交、且不与"搓泥"样本质心（axis 0）重合的方向，
            # 这样"有没有真的补上"可以从分类结果直接读出来。
            return [axis(5, 1 + index) for index, _ in enumerate(texts)]

        taxonomy, result, warnings = assign_units(units, vectors, provider=provider, encode=encode)

        assert calls == [["假白", "闷痘", "难卸", "价格"]], "只该给缺失的痛点补质心"
        backfill = next(w for w in warnings if "痛点名" in w)
        assert f"{len(self.names) - 1} 个痛点" in backfill

        assert len(taxonomy) == len(self.names)
        # ★ 每个痛点都必须出现在分类结果里 —— 少一个就说明它的质心没补上，
        #   而不是"文本恰好都没命中"
        missing_pains = set(self.names) - set(result.labels)
        assert not missing_pains, (
            f"这些痛点没分到任何文本，说明它们的质心没有被补上：{sorted(missing_pains)}"
        )

    def test_encode_is_not_called_when_nothing_is_missing(self):
        """每个痛点都有打标样本时不该白跑一次编码。"""
        units, vectors, provider = self.setup_case(labeled=[self.names[i // 3] for i in range(15)])
        calls: list[list[str]] = []

        def encode(texts: Sequence[str]) -> list[list[float]]:
            calls.append(list(texts))
            return []

        _taxonomy, _result, warnings = assign_units(
            units, vectors, provider=provider, encode=encode
        )

        assert calls == []
        assert not any("痛点名" in warning for warning in warnings)

    def test_no_encoder_still_falls_back_to_the_soft_warning(self):
        """不传 encode 的老路径不受影响 —— 仍是"质心不足"的软警告，不抛错。"""
        units, vectors, provider = self.setup_case()

        _taxonomy, _result, warnings = assign_units(units, vectors, provider=provider)

        assert any("质心" in warning for warning in warnings)


# --------------------------------------------------------------------------- #
# 分组
# --------------------------------------------------------------------------- #


class TestGroupByTaxonomy:
    def make_taxonomy(self) -> PainTaxonomy:
        return PainTaxonomy(
            entries=[
                TaxonomyEntry(name="搓泥", summary="上妆起白条", category="体验粗糙"),
                TaxonomyEntry(name="假白", summary="上脸泛白", category="结果不达预期"),
            ]
        )

    def test_groups_by_label(self):
        units = make_units(5)
        labels = ["搓泥", "搓泥", "假白", "", ""]

        clusters = group_by_taxonomy(units, labels, taxonomy=self.make_taxonomy())

        by_name = {cluster.label: cluster.size for cluster in clusters if not cluster.is_noise}
        assert by_name == {"搓泥": 2, "假白": 1}

    def test_size_is_member_count(self):
        """★ size 是成员条数 —— 这是产品的核心指标。"""
        units = make_units(30)
        clusters = group_by_taxonomy(units, ["搓泥"] * 30)

        target = next(c for c in clusters if c.label == "搓泥")
        assert target.size == 30
        assert len(target.evidences) <= 20, "证据数受展示上限约束，与 size 是两回事"

    def test_unmatched_becomes_noise_cluster(self):
        """未命中的单元不丢弃 —— 它们确实是真实发言，只是不属于已知痛点。"""
        units = make_units(5)
        clusters = group_by_taxonomy(units, ["搓泥", "", "", "", ""])

        noise = [cluster for cluster in clusters if cluster.is_noise]
        assert len(noise) == 1
        assert noise[0].size == 4
        assert clusters[-1].is_noise, "噪声簇必须排在最后"

    def test_taxonomy_fills_summary_and_category(self):
        units = make_units(3)
        clusters = group_by_taxonomy(units, ["搓泥"] * 3, taxonomy=self.make_taxonomy())

        target = next(c for c in clusters if c.label == "搓泥")
        assert target.summary == "上妆起白条"
        assert target.category == "体验粗糙"

    def test_pain_id_is_reproducible(self):
        """同一份语料两次运行必须得到同一个 id —— 历史趋势对比依赖它。"""
        units = make_units(3)
        first = group_by_taxonomy(units, ["搓泥"] * 3)
        second = group_by_taxonomy(units, ["搓泥"] * 3)
        assert first[0].id == second[0].id

    def test_different_names_get_different_ids(self):
        units = make_units(3)
        clusters = group_by_taxonomy(units, ["搓泥", "假白", "搓泥"])
        ids = {cluster.id for cluster in clusters if not cluster.is_noise}
        assert len(ids) == 2

    def test_sorted_by_size_descending(self):
        units = make_units(6)
        clusters = group_by_taxonomy(units, ["A", "A", "A", "B", "B", "C"])
        sizes = [cluster.size for cluster in clusters if not cluster.is_noise]
        assert sizes == sorted(sizes, reverse=True)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="长度不一致"):
            group_by_taxonomy(make_units(3), ["a"])

    def test_works_without_taxonomy(self):
        """没有清单时仍能分组（summary/category 留空），不该抛异常。"""
        clusters = group_by_taxonomy(make_units(3), ["搓泥"] * 3)
        assert clusters[0].label == "搓泥"
        assert clusters[0].summary == ""


class TestDefaults:
    def test_default_threshold_is_low(self):
        """默认阈值刻意偏低 —— 短文本余弦相似度整体压缩在 0.5 附近。"""
        assert DEFAULT_MATCH_THRESHOLD < 0.5

    def test_min_pains_is_at_least_two(self):
        assert MIN_PAINS >= 2
