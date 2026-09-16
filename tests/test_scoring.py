"""机会评分测试。

重点守卫三条公允性规则（见 ``scoring/opportunity.py`` 模块文档）：

1. **调研失败 ≠ 没有竞品** —— ``competitor_gap(research_failed=True)`` 必须是中性值。
   这是本文件最要紧的一条：它一旦失效，一次网络抖动就会凭空造出一个高分假机会，
   而用户会照着它去选题。
2. **缺数据取中性值，不取 0** —— 没有时间戳、没有难度标注时不能当成"确认很差"。
3. **权重归一化后再用** —— 用户把权重调成 ``{pain_strength: 5}`` 时总分不爆表。

另有一组**手算校验**：因子被构造成确定值后，总分必须等于手算结果。只断言
"分数在 0-100 之间"是恒真断言，守不住评分公式。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import cast

import pytest

from xhs_pain_miner.models import CompetitorFinding, Evidence, PainCluster
from xhs_pain_miner.scoring.opportunity import (
    FACTOR_NAMES,
    NEUTRAL,
    ScoreWeights,
    build_card,
    build_cards,
    competitor_gap,
    feasibility_score,
    growth_trend,
    mention_volume,
    pain_strength,
)

TODAY = date.today()
BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def ts(days: int) -> datetime:
    """相对 :data:`BASE` 的时间戳。"""
    return BASE + timedelta(days=days)


def evidence(
    *,
    likes: int = 0,
    created_at: datetime | None = None,
    text: str = "涂完假白到像糊了面粉",
) -> Evidence:
    return Evidence(text=text, source="comment", likes=likes, created_at=created_at)


def cluster(
    *,
    id_: str = "p1",
    label: str = "假白泛白",
    size: int = 10,
    sentiment: float = -0.8,
    evidences: list[Evidence] | None = None,
    stage: str = "stable",
    difficulty: int = 3,
    feasibility: str = "",
    summary: str = "涂完像戴了面具",
    category: str = "结果不达预期",
) -> PainCluster:
    return PainCluster(
        id=id_,
        label=label,
        summary=summary,
        size=size,
        sentiment=sentiment,
        evidences=evidences if evidences is not None else [evidence()],
        category=category,
        stage=stage,  # type: ignore[arg-type]
        difficulty=difficulty,
        feasibility=feasibility,
    )


def finding(
    *,
    name: str = "some-tool",
    url: str = "https://github.com/example/some-tool",
    stars: int | None = 120,
    last_active: date | None = TODAY,
) -> CompetitorFinding:
    return CompetitorFinding(
        source="github", name=name, url=url, stars=stars, last_active=last_active
    )


# --------------------------------------------------------------------------- #
# 权重
# --------------------------------------------------------------------------- #


class TestScoreWeights:
    """权重归一化与反序列化。"""

    def test_default_weights_are_already_normalized(self):
        assert ScoreWeights().normalized() == ScoreWeights()

    def test_normalized_sums_to_one(self):
        weights = ScoreWeights(
            pain_strength=3.0,
            mention_volume=1.0,
            growth_trend=0.0,
            competitor_gap=1.0,
            feasibility=0.0,
        ).normalized()
        total = sum(weights.to_dict().values())
        assert total == pytest.approx(1.0)
        assert weights.pain_strength == pytest.approx(0.6)
        assert weights.competitor_gap == pytest.approx(0.2)
        assert weights.growth_trend == 0.0

    def test_all_zero_falls_back_to_default_without_raising(self):
        """权重全清空是输入失误，给一个能用的结果，而不是抛异常。"""
        weights = ScoreWeights(0.0, 0.0, 0.0, 0.0, 0.0).normalized()
        assert weights == ScoreWeights()

    def test_all_negative_falls_back_to_default(self):
        weights = ScoreWeights(-1.0, -2.0, -3.0, -4.0, -5.0).normalized()
        assert weights == ScoreWeights()

    def test_negative_weight_is_clamped_to_zero(self):
        """单个负权重没有意义（"这个维度越好分越低"），按 0 处理而不是让总分为负。"""
        weights = ScoreWeights(
            pain_strength=-1.0,
            mention_volume=1.0,
            growth_trend=0.0,
            competitor_gap=0.0,
            feasibility=0.0,
        ).normalized()
        assert weights.pain_strength == 0.0
        assert weights.mention_volume == pytest.approx(1.0)

    def test_to_dict_follows_factor_order(self):
        assert tuple(ScoreWeights().to_dict()) == FACTOR_NAMES

    def test_from_dict_unknown_factor_raises(self):
        """拼错的键被静默忽略，用户会以为自己的调整生效了 —— 必须抛。"""
        with pytest.raises(ValueError) as excinfo:
            ScoreWeights.from_dict({"pain_strengh": 1.0})
        message = str(excinfo.value)
        assert "pain_strengh" in message
        assert "pain_strength" in message  # 提示可用因子名

    def test_from_dict_partial_keeps_defaults(self):
        weights = ScoreWeights.from_dict({"competitor_gap": 0.4})
        assert weights.competitor_gap == 0.4
        assert weights.pain_strength == ScoreWeights().pain_strength

    def test_from_dict_accepts_numeric_strings(self):
        assert ScoreWeights.from_dict({"pain_strength": "0.5"}).pain_strength == 0.5

    def test_from_dict_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            ScoreWeights.from_dict({"pain_strength": "很痛"})

    def test_from_dict_roundtrip(self):
        weights = ScoreWeights(0.1, 0.2, 0.3, 0.3, 0.1)
        assert ScoreWeights.from_dict(weights.to_dict()) == weights


# --------------------------------------------------------------------------- #
# 各因子
# --------------------------------------------------------------------------- #


class TestPainStrength:
    def test_no_evidence_is_neutral(self):
        """没有证据就是"不知道"，不能当作"不痛"（取 0）。"""
        assert pain_strength(cluster(evidences=[])) == NEUTRAL

    def test_most_negative_sentiment_is_strongest(self):
        strongest = pain_strength(cluster(sentiment=-1.0, evidences=[evidence()]))
        weakest = pain_strength(cluster(sentiment=1.0, evidences=[evidence()]))
        assert strongest > weakest
        assert weakest == 0.0

    def test_more_likes_means_stronger(self):
        """高赞的抱怨比无人问津的抱怨更重。"""
        popular = pain_strength(cluster(evidences=[evidence(likes=500)]))
        ignored = pain_strength(cluster(evidences=[evidence(likes=0)]))
        assert popular > ignored

    def test_zero_likes_still_counts(self):
        """零赞的抱怨仍然是抱怨 —— 强度不该被点赞数归零。"""
        assert pain_strength(
            cluster(sentiment=-1.0, evidences=[evidence(likes=0)])
        ) == pytest.approx(0.5)

    def test_neutral_sentiment_is_not_treated_as_painful(self):
        neutral = pain_strength(cluster(sentiment=0.0, evidences=[evidence(likes=1000)]))
        assert neutral <= NEUTRAL

    def test_stays_within_unit_range(self):
        for sentiment in (-5.0, -1.0, 0.0, 5.0):
            for likes in (-10, 0, 10**9):
                value = pain_strength(
                    cluster(sentiment=sentiment, evidences=[evidence(likes=likes)])
                )
                assert 0.0 <= value <= 1.0


class TestMentionVolume:
    def test_largest_cluster_is_one(self):
        assert mention_volume(cluster(size=42), max_size=42) == 1.0

    def test_zero_max_size_is_zero(self):
        assert mention_volume(cluster(size=10), max_size=0) == 0.0

    def test_zero_size_is_zero(self):
        assert mention_volume(cluster(size=0), max_size=10) == 0.0

    def test_log_normalization_keeps_long_tail_visible(self):
        """线性归一会让第 10 名挤在 0 附近；log 归一必须把它留在可区分区间。"""
        value = mention_volume(cluster(size=10), max_size=100)
        assert value == pytest.approx(0.5205, abs=0.01)
        assert value > 0.1  # 线性值

    def test_never_exceeds_one(self):
        assert mention_volume(cluster(size=500), max_size=100) == 1.0


class TestGrowthTrend:
    def test_no_evidence_is_neutral(self):
        assert growth_trend(cluster(evidences=[])) == NEUTRAL

    def test_missing_timestamps_is_neutral(self):
        """全部没有时间戳 → 推不出趋势，取中性而不是当作"平稳以外"的任何东西。"""
        evidences = [evidence() for _ in range(4)]
        assert growth_trend(cluster(evidences=evidences)) == NEUTRAL

    def test_more_than_half_missing_is_neutral(self):
        """★ 超过一半没有时间戳 → 半份数据推不出趋势。

        两条时间戳**跨越了足够长的时间**：若实现只判断"有没有时间戳"，这里会算出
        0.55 而不是中性值 —— 这个构造就是让那种实现变红。
        """
        evidences = [
            evidence(created_at=ts(0)),
            evidence(created_at=ts(100)),
            *[evidence() for _ in range(3)],
        ]
        assert growth_trend(cluster(evidences=evidences, stage="growing")) == NEUTRAL

    def test_exactly_half_missing_still_computes(self):
        """恰好一半有时间戳时仍有一半是硬数据，可以算（用 stage 偏移把它与中性值区分开）。"""
        evidences = [
            evidence(created_at=ts(0)),
            evidence(created_at=ts(100)),
            evidence(),
            evidence(),
        ]
        # 后续半段占比 0.5 + growing 修正 0.05 = 0.55，与 NEUTRAL 不同即说明真的算了
        assert growth_trend(cluster(evidences=evidences, stage="growing")) == pytest.approx(0.55)

    def test_all_timestamps_equal_is_neutral(self):
        """时间跨度为零时切不出前后两半。"""
        evidences = [evidence(created_at=ts(7)) for _ in range(3)]
        assert growth_trend(cluster(evidences=evidences)) == NEUTRAL

    def test_recent_heavy_is_rising(self):
        days = [0, 98, 99, 100]
        evidences = [evidence(created_at=ts(d)) for d in days]
        assert growth_trend(cluster(evidences=evidences)) == pytest.approx(0.75)

    def test_old_heavy_is_declining(self):
        days = [0, 1, 2, 100]
        evidences = [evidence(created_at=ts(d)) for d in days]
        assert growth_trend(cluster(evidences=evidences)) == pytest.approx(0.25)

    def test_balanced_is_flat(self):
        evidences = [evidence(created_at=ts(0)), evidence(created_at=ts(10))]
        assert growth_trend(cluster(evidences=evidences)) == pytest.approx(NEUTRAL)

    def test_stage_only_adjusts_within_bounds(self):
        evidences = [evidence(created_at=ts(0)), evidence(created_at=ts(10))]
        assert growth_trend(cluster(evidences=evidences, stage="new")) == pytest.approx(0.60)
        assert growth_trend(cluster(evidences=evidences, stage="declining")) == pytest.approx(0.40)

    def test_stage_cannot_overturn_timestamps(self):
        """模型对趋势的判断容易过度自信；修正项幅度受限，不能把上升说成下降。"""
        evidences = [evidence(created_at=ts(d)) for d in (0, 98, 99, 100)]
        value = growth_trend(cluster(evidences=evidences, stage="declining"))
        assert value == pytest.approx(0.65)
        assert value > NEUTRAL


class TestCompetitorGap:
    def test_research_failed_is_neutral(self):
        """★ 本模块最重要的一条：调研失败绝不返回高分。"""
        gap = competitor_gap([], research_failed=True)
        assert gap == NEUTRAL
        assert gap < 1.0  # 尤其不能是"没有竞品"的满分

    def test_research_failed_is_neutral_even_with_findings(self):
        """部分结果 + 中途失败，同样不能当作战果。"""
        assert competitor_gap([finding()], research_failed=True) == NEUTRAL

    def test_no_competitor_is_full_score(self):
        assert competitor_gap([]) == 1.0

    def test_all_stale_is_positive_but_not_full(self):
        stale = [
            finding(name="a", last_active=date(2019, 1, 1)),
            finding(name="b", last_active=date(2020, 5, 1)),
        ]
        assert competitor_gap(stale) == 0.75

    def test_active_competitor_lowers_the_gap(self):
        gap = competitor_gap([finding(stars=0)])
        assert gap < 0.75
        assert gap > 0.0

    def test_more_active_competitors_lower_the_gap(self):
        one = competitor_gap([finding(name="a", stars=0)])
        two = competitor_gap([finding(name="a", stars=0), finding(name="b", stars=0)])
        three = competitor_gap(
            [finding(name="a", stars=0), finding(name="b", stars=0), finding(name="c", stars=0)]
        )
        assert one > two > three

    def test_hotter_competitor_lowers_the_gap(self):
        cold = competitor_gap([finding(stars=0)])
        hot = competitor_gap([finding(stars=50_000)])
        assert hot < cold

    def test_unknown_stars_is_between_cold_and_hot(self):
        """stars 缺失是"不知道"，取中性影响，而不是当成"不热门"。"""
        unknown = competitor_gap([finding(stars=None)])
        cold = competitor_gap([finding(stars=0)])
        hot = competitor_gap([finding(stars=50_000)])
        assert hot < unknown < cold

    def test_unknown_last_active_counts_as_active(self):
        """最后活跃时间未知 ≠ 停更 —— 不能因为查不到时间就宣布市场空着。"""
        unknown = competitor_gap([finding(last_active=None)])
        assert unknown < 0.75

    def test_stale_months_parameter_is_honoured(self):
        recent = [finding(last_active=TODAY - timedelta(days=90))]
        assert competitor_gap(recent, stale_months=1) == 0.75
        assert competitor_gap(recent, stale_months=12) < 0.75

    def test_stale_and_active_mix_is_between(self):
        mixed = competitor_gap(
            [finding(name="old", last_active=date(2018, 1, 1)), finding(name="new", stars=0)]
        )
        assert 0.0 < mixed < 0.75

    def test_result_is_never_higher_than_empty_market(self):
        """任何有竞品的情形都必须低于"查证过没有竞品"。"""
        for findings in (
            [finding(stars=0)],
            [finding(stars=10**6)],
            [finding(last_active=date(2015, 1, 1))],
            [finding(stars=None, last_active=None)],
        ):
            assert competitor_gap(findings) < competitor_gap([])


class TestFeasibilityScore:
    def test_difficulty_one_is_full(self):
        assert feasibility_score(cluster(difficulty=1)) == 1.0

    def test_difficulty_five_is_zero(self):
        assert feasibility_score(cluster(difficulty=5)) == 0.0

    def test_difficulty_three_is_half(self):
        assert feasibility_score(cluster(difficulty=3)) == pytest.approx(0.5)

    def test_out_of_range_is_clamped(self):
        assert feasibility_score(cluster(difficulty=0)) == 1.0
        assert feasibility_score(cluster(difficulty=99)) == 0.0

    def test_missing_difficulty_is_neutral(self):
        """字段缺失是"没评估过"，不是"难度为零/满分"。"""
        bare = cast("PainCluster", object())  # 既没有 difficulty 也没有 evidences 的替身
        assert feasibility_score(bare) == NEUTRAL


class _NoTimestampEvidence:
    """没有任何时间字段的证据替身（模拟采集后端还没接上时间）。"""

    def __init__(self) -> None:
        self.text = "没有时间信息"
        self.source = "comment"
        self.likes = 3

    def __getattr__(self, name: str) -> None:  # 任何未知属性都当作缺失
        raise AttributeError(name)


class TestDebtTolerantReaders:
    """字段缺失时必须降级为中性值，而不是崩溃。"""

    def test_missing_timestamp_attribute_is_neutral(self):
        """采集后端还没接上时间时，趋势必须是"不知道"，而不是当作"很早"。"""
        fake = PainCluster(id="p", size=2)
        fake.evidences = [_NoTimestampEvidence()] * 2  # type: ignore[list-item]
        assert growth_trend(fake) == NEUTRAL

    def test_missing_timestamp_attribute_does_not_break_other_factors(self):
        fake = PainCluster(id="p", size=2)
        fake.evidences = [_NoTimestampEvidence()] * 2  # type: ignore[list-item]
        assert 0.0 <= pain_strength(fake) <= 1.0
        assert mention_volume(fake, max_size=2) == 1.0


# --------------------------------------------------------------------------- #
# 卡片
# --------------------------------------------------------------------------- #


class TestBuildCard:
    def test_id_is_derived_from_cluster_id(self):
        card = build_card(
            cluster(id_="cluster-7"),
            findings=[],
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=10,
        )
        assert card.id == "card-cluster-7"

    def test_id_is_reproducible(self):
        """同一份语料两次运行必须得到同一个 ID，否则历史趋势对比无从谈起。"""
        args = dict(findings=[], weights=ScoreWeights(), keyword="防晒霜", max_size=10)
        first = build_card(cluster(), **args)
        second = build_card(cluster(), **args)
        assert first.id == second.id
        assert first.score == second.score

    def test_title_is_a_direction_not_the_pain_label(self):
        """标题是"要做什么"，label 是"用户卡在哪"，两者不能是同一句话。"""
        card = build_card(
            cluster(label="假白泛白"),
            findings=[],
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=10,
        )
        assert card.title != card.pain.label
        assert card.pain.label in card.title

    def test_direction_pattern_is_used_when_pain_pattern_matches(self):
        card = build_card(
            cluster(label="导出太麻烦"),
            findings=[],
            weights=ScoreWeights(),
            keyword="笔记工具",
            max_size=10,
        )
        assert "更省事" in card.title
        assert card.title != "导出太麻烦"

    def test_degraded_cluster_never_uses_raw_text_as_title(self):
        """降级簇没有痛点名 —— 绝不能截一段证据原文当标题（不变式 5）。"""
        raw = "我用的那支上脸假白到像糊了面粉，同事问我是不是过敏了"
        card = build_card(
            cluster(label="<未命名痛点 #3>", evidences=[evidence(text=raw)]),
            findings=[],
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=10,
        )
        assert raw not in card.title
        assert "<未命名" not in card.title
        assert card.title

    def test_breakdown_covers_every_factor_in_order(self):
        card = build_card(
            cluster(), findings=[], weights=ScoreWeights(), keyword="防晒霜", max_size=10
        )
        assert tuple(card.score_breakdown) == FACTOR_NAMES
        for value in card.score_breakdown.values():
            assert 0.0 <= value <= 1.0

    def test_score_matches_hand_computation(self):
        """★ 手算校验：每个因子都构造成确定值，总分必须等于手算结果。"""
        card = build_card(
            cluster(size=10, sentiment=-1.0, evidences=[evidence(likes=0)], difficulty=1),
            findings=[],
            weights=ScoreWeights(),  # 默认权重
            keyword="防晒霜",
            max_size=10,
        )
        expected = 100.0 * (
            0.25 * 0.5  # 痛点强度：情感 -1 但零点赞 → 0.5 × 1.0
            + 0.20 * 1.0  # 提及量：本身就是最大簇
            + 0.20 * 0.5  # 增长趋势：无时间戳 → 中性
            + 0.25 * 1.0  # 竞品空白度：查证过没有竞品
            + 0.10 * 1.0  # 实现难度：difficulty=1
        )
        assert card.score == pytest.approx(expected, abs=0.05)
        assert card.score == pytest.approx(77.5, abs=0.05)
        assert card.score_breakdown["pain_strength"] == pytest.approx(0.5)
        assert card.score_breakdown["growth_trend"] == NEUTRAL

    def test_score_matches_hand_computation_with_custom_weights(self):
        """自造权重下同样可手算：3:1 归一后为 0.75 / 0.25。"""
        card = build_card(
            cluster(size=10, sentiment=-1.0, evidences=[evidence(likes=0)], difficulty=1),
            findings=[],
            weights=ScoreWeights(
                pain_strength=3.0,
                mention_volume=0.0,
                growth_trend=0.0,
                competitor_gap=1.0,
                feasibility=0.0,
            ),
            keyword="防晒霜",
            max_size=10,
        )
        expected = 100.0 * (0.75 * 0.5 + 0.25 * 1.0)
        assert card.score == pytest.approx(expected, abs=0.05)
        assert card.score == pytest.approx(62.5, abs=0.05)

    def test_normalized_weights_never_push_score_over_100(self):
        """用户把权重调成 {pain_strength: 5} 时总分不该爆表。"""
        worst = cluster(size=10, sentiment=-1.0, evidences=[evidence(likes=10**6)], difficulty=1)
        card = build_card(
            worst,
            findings=[],  # 空白度 1.0
            weights=ScoreWeights(pain_strength=5.0),
            keyword="防晒霜",
            max_size=10,
        )
        assert card.score <= 100.0
        assert card.score >= 0.0

    def test_research_failure_lowers_the_score(self):
        """调研失败按中性计 —— 必须比"查证过没有竞品"低，否则就是凭空造机会。"""
        args = dict(weights=ScoreWeights(), keyword="防晒霜", max_size=10)
        verified_empty = build_card(cluster(), findings=[], research_failed=False, **args)
        failed = build_card(cluster(), findings=[], research_failed=True, **args)
        assert failed.score_breakdown["competitor_gap"] == NEUTRAL
        assert verified_empty.score_breakdown["competitor_gap"] == 1.0
        assert failed.score < verified_empty.score

    def test_feasibility_text_comes_from_cluster(self):
        card = build_card(
            cluster(feasibility="个人可做 / 1-2 周"),
            findings=[],
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=10,
        )
        assert card.feasibility == "个人可做 / 1-2 周"

    def test_feasibility_text_derived_from_difficulty_when_absent(self):
        card = build_card(
            cluster(difficulty=4, feasibility=""),
            findings=[],
            weights=ScoreWeights(),
            keyword="防晒霜",
            max_size=10,
        )
        assert card.feasibility == "需要团队"


class TestBuildCards:
    def test_cards_are_sorted_by_score_desc(self):
        clusters = [
            cluster(id_="small", size=2, sentiment=-0.2, evidences=[evidence()]),
            cluster(id_="big", size=50, sentiment=-1.0, evidences=[evidence(likes=100)]),
        ]
        cards, _ = build_cards(clusters, findings_by_cluster={}, keyword="防晒霜")
        assert [card.pain.id for card in cards] == ["big", "small"]
        assert cards[0].score > cards[1].score

    def test_min_size_filters_small_clusters(self):
        clusters = [cluster(id_="a", size=10), cluster(id_="b", size=2)]
        cards, _ = build_cards(clusters, findings_by_cluster={}, min_size=5)
        assert [card.pain.id for card in cards] == ["a"]

    def test_mention_volume_is_normalized_by_the_largest_cluster(self):
        """归一化基准是最大簇的 size —— 除头名外的提及量都按 log 比例落位。"""
        clusters = [
            cluster(id_="huge", size=100),
            cluster(id_="mid", size=50),
            cluster(id_="tiny", size=5),
        ]
        cards, _ = build_cards(clusters, findings_by_cluster={}, keyword="防晒霜")
        by_id = {card.pain.id: card for card in cards}
        assert by_id["huge"].score_breakdown["mention_volume"] == 1.0
        assert by_id["mid"].score_breakdown["mention_volume"] == pytest.approx(
            mention_volume(clusters[1], max_size=100)
        )
        assert (
            by_id["tiny"].score_breakdown["mention_volume"]
            < by_id["mid"].score_breakdown["mention_volume"]
            < 1.0
        )

    def test_findings_are_routed_by_cluster_id(self):
        clusters = [cluster(id_="a"), cluster(id_="b")]
        cards, _ = build_cards(
            clusters,
            findings_by_cluster={"a": [finding(name="only-for-a")]},
            keyword="防晒霜",
        )
        by_id = {card.pain.id: card for card in cards}
        assert [c.name for c in by_id["a"].competitors] == ["only-for-a"]
        assert by_id["b"].competitors == []

    def test_default_weights_produce_no_weight_warning(self):
        cards, warnings = build_cards([cluster()], findings_by_cluster={})
        assert cards
        assert not any("权重" in warning for warning in warnings)

    def test_rescaled_weights_warn_the_user(self):
        """用户改了一个"看起来生效了"的权重，必须被告知实际生效的值。"""
        _, warnings = build_cards(
            [cluster()],
            findings_by_cluster={},
            weights=ScoreWeights(pain_strength=5.0),
        )
        assert any("归一化" in warning for warning in warnings)

    def test_custom_weights_change_the_score(self):
        clusters = [cluster(id_="a", size=10), cluster(id_="b", size=5)]
        default_cards, _ = build_cards(clusters, findings_by_cluster={})
        custom_cards, _ = build_cards(
            clusters,
            findings_by_cluster={},
            weights=ScoreWeights(
                pain_strength=0.0,
                mention_volume=0.0,
                growth_trend=0.0,
                competitor_gap=1.0,
                feasibility=0.0,
            ),
        )
        default_by_id = {card.pain.id: card for card in default_cards}
        custom_by_id = {card.pain.id: card for card in custom_cards}
        assert custom_by_id["b"].score != default_by_id["b"].score

    def test_all_zero_weights_fall_back_and_warn(self):
        cards, warnings = build_cards(
            [cluster()],
            findings_by_cluster={},
            weights=ScoreWeights(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        assert any("默认权重" in warning for warning in warnings)
        expected, _ = build_cards([cluster()], findings_by_cluster={})
        assert cards[0].score == expected[0].score

    def test_research_failure_is_surfaced_as_a_warning(self):
        """失败必须出现在警告里 —— 静默降级会让用户以为看到的是完整结果。"""
        _, warnings = build_cards(
            [cluster(id_="a")],
            findings_by_cluster={},
            failed_clusters=["a"],
        )
        assert any("竞品调研失败" in warning for warning in warnings)

    def test_research_failure_warning_is_not_emitted_without_failures(self):
        _, warnings = build_cards([cluster(id_="a")], findings_by_cluster={})
        assert not any("竞品调研失败" in warning for warning in warnings)

    def test_failed_cluster_gets_neutral_gap(self):
        cards, _ = build_cards(
            [cluster(id_="a")],
            findings_by_cluster={},
            failed_clusters=["a"],
        )
        assert cards[0].score_breakdown["competitor_gap"] == NEUTRAL

    def test_unknown_failed_cluster_id_is_ignored(self):
        cards, warnings = build_cards(
            [cluster(id_="a")],
            findings_by_cluster={},
            failed_clusters=["不存在的簇"],
        )
        assert len(cards) == 1
        assert not any("竞品调研失败" in warning for warning in warnings)

    def test_empty_input_returns_empty(self):
        assert build_cards([], findings_by_cluster={}) == ([], [])

    def test_run_is_reproducible(self):
        clusters = [cluster(id_="a"), cluster(id_="b", size=3)]
        first, _ = build_cards(clusters, findings_by_cluster={}, keyword="防晒霜")
        second, _ = build_cards(clusters, findings_by_cluster={}, keyword="防晒霜")
        assert [(card.id, card.score, card.title) for card in first] == [
            (card.id, card.score, card.title) for card in second
        ]


class TestTitleUniqueness:
    """★ 卡片标题必须两两不同 —— 卡片是本产品的核心交付物，用户看的就是标题。

    方向模板只有六个，匹配规则却很宽松（"难"一个字就能吃掉所有含"难"的痛点名），
    所以多个痛点命中同一个模板是常态。一旦共用，标题就会撞名，报告里出现几张
    一模一样的卡片，用户无法区分 —— 修复前实测 4 个不同痛点全部得到
    「防晒霜 · 零门槛的工具」。

    契约：模板被**多个**痛点共用时，这些痛点**全部**退回默认模板
    ``解决「{label}」的工具``（自带痛点名，天然唯一）；只有独占模板的才用它。
    """

    KEYWORD = "防晒霜"

    def titles_by_label(self, clusters: list[PainCluster]) -> dict[str, str]:
        cards, _ = build_cards(clusters, findings_by_cluster={}, keyword=self.KEYWORD)
        assert len(cards) == len(clusters), "所有簇都该生成卡片"
        return {card.pain.label: card.title for card in cards}

    def build(self, labels: list[str]) -> list[PainCluster]:
        return [
            cluster(id_=f"p{index}", label=label, size=len(labels) * 2 - index)
            for index, label in enumerate(labels)
        ]

    def test_shared_template_is_dropped_for_every_holder(self):
        """共用一个模板的三个痛点全部退回默认模板 —— 不是只有后来者退。"""
        labels = ["难卸妆", "包装难用", "上手门槛高"]
        titles = self.titles_by_label(self.build(labels))

        assert len(set(titles.values())) == 3, f"标题撞名: {titles}"
        assert set(titles.values()) == {
            f"{self.KEYWORD} · 解决「{label}」的工具" for label in labels
        }

    def test_lone_holder_keeps_its_template(self):
        """独占模板的痛点照旧用模板措辞 —— 唯一化不能把所有标题都降级成默认。"""
        titles = self.titles_by_label(self.build(["难卸妆", "包装难用", "价格虚高"]))

        assert titles["价格虚高"] == f"{self.KEYWORD} · 更低成本的工具"
        assert titles["难卸妆"] == f"{self.KEYWORD} · 解决「难卸妆」的工具"
        assert titles["包装难用"] == f"{self.KEYWORD} · 解决「包装难用」的工具"

    def test_the_four_pains_from_the_bug_report_all_get_distinct_titles(self):
        """★ 回归守卫：修复前这四个痛点全部得到「防晒霜 · 零门槛的工具」。"""
        titles = self.titles_by_label(
            self.build(["难卸妆", "包装难用", "不会选色号", "上手门槛高"])
        )

        assert len(set(titles.values())) == 4, f"标题撞名: {titles}"

    def test_default_template_holders_need_no_fallback(self):
        """默认模板自带 label，两个痛点共用也不会撞名 —— 不该被无谓地改写。"""
        titles = self.titles_by_label(self.build(["假白泛白", "搓泥"]))

        assert titles == {
            "假白泛白": f"{self.KEYWORD} · 解决「假白泛白」的工具",
            "搓泥": f"{self.KEYWORD} · 解决「搓泥」的工具",
        }

    def test_single_holder_is_unchanged(self):
        """不回归：只有一个痛点命中模板时行为与修复前完全一致。"""
        assert self.titles_by_label(self.build(["导出太麻烦"])) == {
            "导出太麻烦": f"{self.KEYWORD} · 更省事的工具"
        }
        assert self.titles_by_label(self.build(["闷痘闭口"])) == {
            "闷痘闭口": f"{self.KEYWORD} · 解决「闷痘闭口」的工具"
        }

    def test_degraded_cluster_still_gets_pending_direction(self):
        """降级簇既不共用模板、也不回退 —— 回退会把占位名拼进标题（不变式 5）。

        它的标题里根本没有那个模板，因此也**不占**模板名额：旁边的痛点照旧用
        「零门槛」，不被无谓地拖回默认模板。
        """
        titles = self.titles_by_label(self.build(["难卸妆", "<未命名痛点 #3>"]))

        pending = titles["<未命名痛点 #3>"]
        assert pending == f"{self.KEYWORD} · 待命名方向"
        assert "未命名" not in pending and "<" not in pending
        assert titles["难卸妆"] == f"{self.KEYWORD} · 零门槛的工具"

    def test_filtered_clusters_do_not_count_as_holders(self):
        """被 min_size 滤掉的簇根本不出现在报告里，不该把标题"拖"回默认模板。"""
        clusters = [
            cluster(id_="big", label="难卸妆", size=10),
            cluster(id_="tiny", label="包装难用", size=1),
        ]
        cards, _ = build_cards(clusters, findings_by_cluster={}, keyword=self.KEYWORD, min_size=5)

        assert [card.pain.label for card in cards] == ["难卸妆"]
        assert cards[0].title == f"{self.KEYWORD} · 零门槛的工具"

    def test_fallback_does_not_depend_on_cluster_order(self):
        """★ 为什么不是"先到先得"：那样顺序一变就换人拿模板。

        簇的顺序来自 ``size`` 降序，而语料稍有变化 size 就会变 —— 同一份需求
        两次分析得到两份措辞不同的报告。全退与顺序无关。
        """
        clusters = self.build(["难卸妆", "包装难用", "价格虚高"])

        forward = self.titles_by_label(clusters)
        backward = self.titles_by_label(list(reversed(clusters)))

        assert forward == backward

    def test_no_keyword_still_yields_unique_titles(self):
        """没有品类关键词时标题没有前缀，唯一性不能因此失效。"""
        labels = ["难卸妆", "包装难用", "不会选色号"]
        cards, _ = build_cards(self.build(labels), findings_by_cluster={})

        assert len({card.title for card in cards}) == 3
