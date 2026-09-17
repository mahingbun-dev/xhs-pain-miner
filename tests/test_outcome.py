"""竞品调研结论契约的测试。

**这是 M2 最关键的一组测试**：M1 把三种完全不同的处境（查证过确实没有 /
检索不到 / 没查成）压成了同一个值（空列表），而评分层把空列表解读成
「查证过没有竞品」—— 机会分里最强的正面信号。用户会照着一个假信号去做
一个实际已经很拥挤的方向。

所以这里的断言必须钉住**判定依据**，而不只是"没崩"。
"""

from __future__ import annotations

import pytest

from xhs_pain_miner.models import CompetitorFinding
from xhs_pain_miner.research.outcome import (
    QueryTrace,
    ResearchOutcome,
    build_outcome,
    classify_status,
    warning_for,
)


def finding(name: str = "some-tool", url: str = "https://example.test/x") -> CompetitorFinding:
    return CompetitorFinding(source="appstore", name=name, url=url)


def ok_trace(query: str = "美妆 成分查询", *, hits: int = 3, kept: int = 1) -> QueryTrace:
    """一次**成功且有命中**的检索。"""
    return QueryTrace(query=query, channel="appstore", hits=hits, kept=kept)


def empty_trace(query: str = "防晒搓泥") -> QueryTrace:
    """一次成功但**平台没返回任何东西**的检索。"""
    return QueryTrace(query=query, channel="appstore", hits=0, kept=0)


def failed_trace(query: str = "搓泥", error: str = "429 限流") -> QueryTrace:
    return QueryTrace(query=query, channel="appstore", error=error)


# --------------------------------------------------------------------------- #
# classify_status —— 本模块存在的全部理由
# --------------------------------------------------------------------------- #


class TestClassifyStatus:
    def test_findings_mean_ok(self):
        assert classify_status([ok_trace()], [finding()]) == "ok"

    def test_hits_but_nothing_kept_is_verified_empty(self):
        """★ 平台**搜到过东西**、只是不相关 → 这才叫「查证过确实没有竞品」。

        这是唯一允许给出 1.0 空白度的情形。
        """
        trace = QueryTrace(query="notes export", channel="github", hits=3571, kept=0)
        assert classify_status([trace], []) == "no_competitor"

    def test_zero_hits_is_unsearchable_not_empty(self):
        """★★ M2 的核心修复：平台**一条都没返回**时，绝不能说「没有竞品」。

        实测（2026-09-16）：`防晒搓泥` 在 App Store 与 GitHub 上都是 0 命中，
        因为痛点名描述的是"问题"，而竞品是"解法" —— 两者词汇没有交集。
        把它判成「查证过确实没有竞品」会让用户去做一个实际已有
        百度网盘（92 万评分）这类强势竞品的红海方向。

        变异提示：把 ``classify_status`` 的 ``any(hits > 0)`` 改成无条件
        ``return "no_competitor"``，这条测试必须变红。
        """
        assert classify_status([empty_trace()], []) == "unsearchable"

    def test_all_queries_failed_is_failed(self):
        assert classify_status([failed_trace()], []) == "failed"

    def test_no_queries_at_all_is_unsearchable(self):
        """压根没有查询词（比如降级簇）—— 没查过，不是"查了没成"。"""
        assert classify_status([], []) == "unsearchable"

    def test_partial_failure_blocks_the_claim(self):
        """★ 有查询**没查成**时不能断言「没有竞品」—— 那次里可能正躺着竞品。

        缺了这条会产出**自相矛盾**的结论：失败轨迹当时的文案写着"该簇的竞品空白度
        必须按中性值处理"，而结论给出的正是 1.0、报告印「✅ 查证过，没有相关
        竞品」。跨渠道有 ``ResearchOutcome.merged`` 挡着（一个渠道没查成就不断言），
        同渠道内原本没有 —— 同一条不变式的缺口。（轨迹文案后来改成只陈述"结果不
        完整"，处方移到 :func:`warning_for`；这条守卫本身不变。）

        这条路径**可达**：``_search_channels`` 在渠道内首次失败即放弃（避免加深
        限流），所以"第一条词查到、第二条被限流"正是限流随运行累积时的常见形态。
        实测后果：卡片空白度 1.0，而保守取值应为 0.5，机会分虚高 12.5 分。

        变异提示：去掉 ``classify_status`` 里 ``len(succeeded) < len(queries)``
        那条判断，这条测试必须变红。
        """
        hits = QueryTrace(query="notes export", channel="github", hits=12, kept=0)
        assert classify_status([failed_trace(), empty_trace()], []) == "unsearchable"
        assert classify_status([failed_trace(), hits], []) == "unsearchable", (
            "有一条查询没查成，就不能说「查证过确实没有」"
        )
        # 全部查询都成功时，才允许下这个断言
        assert classify_status([hits, empty_trace()], []) == "no_competitor"

    def test_findings_win_over_everything(self):
        """留下了相关竞品就是 ok，无论其它查询是失败还是空手而归。"""
        assert classify_status([failed_trace(), empty_trace()], [finding()]) == "ok"


# --------------------------------------------------------------------------- #
# QueryTrace / ResearchOutcome 的派生语义
# --------------------------------------------------------------------------- #


class TestTraceSemantics:
    def test_succeeded_is_about_error_only(self):
        """``succeeded`` 只看有没有报错 —— 命中 0 条也是一次成功的查询。

        把"0 命中"当成失败会让结论退化成 ``failed``，用户就看不到
        "这个词在该平台检索不到"这条更有价值的信息。
        """
        assert empty_trace().succeeded is True
        assert failed_trace().succeeded is False


class TestOutcomeSemantics:
    def test_verified_empty_only_for_no_competitor(self):
        assert ResearchOutcome(status="no_competitor").verified_empty is True
        for status in ("ok", "unsearchable", "failed"):
            assert ResearchOutcome(status=status).verified_empty is False  # type: ignore[arg-type]

    def test_inconclusive_covers_both_unsearchable_and_failed(self):
        """对评分而言两者完全一样（中性值）—— 分开只是为了文案说清原因。"""
        assert ResearchOutcome(status="unsearchable").inconclusive is True
        assert ResearchOutcome(status="failed").inconclusive is True
        assert ResearchOutcome(status="ok").inconclusive is False
        assert ResearchOutcome(status="no_competitor").inconclusive is False

    def test_research_failed_covers_the_zero_hit_case(self):
        """★★ M2 最容易被集成层撤销的一处。

        评分侧最自然的写法是 ``status == "failed"``，而那会把 ``unsearchable``
        （0 命中 —— M2 修掉的假空白）漏在外面，于是它被翻回 1.0：卡片虚高
        12.5 分，报告重新印出「✅ 未发现竞品 —— 查证过」。``classify_status``
        里做对的事，就这样被一个看似合理的映射原样撤销。

        变异提示：把 ``research_failed`` 改成 ``status == "failed"``，
        这条测试必须变红。
        """
        # 只有这两种状态允许让空白度偏离中性值
        assert ResearchOutcome(status="ok").research_failed is False
        assert ResearchOutcome(status="no_competitor").research_failed is False
        # 0 命中必须按"没查成"处理 —— 这是 M2 的全部意义所在
        assert ResearchOutcome(status="unsearchable").research_failed is True
        assert ResearchOutcome(status="failed").research_failed is True

    def test_research_failed_defaults_to_true(self):
        """忘记赋值时不许白送一个 1.0。"""
        assert ResearchOutcome().research_failed is True


class TestMerged:
    def test_findings_from_either_channel_win(self):
        left = ResearchOutcome(findings=(finding("a"),), status="ok", queries=(ok_trace(),))
        right = ResearchOutcome(status="unsearchable", queries=(empty_trace(),))
        merged = left.merged(right)
        assert merged.status == "ok"
        assert [f.name for f in merged.findings] == ["a"]

    def test_both_verified_empty_stays_verified_empty(self):
        left = ResearchOutcome(status="no_competitor", queries=(ok_trace(),))
        right = ResearchOutcome(status="no_competitor", queries=(ok_trace(),))
        assert left.merged(right).status == "no_competitor"

    def test_one_inconclusive_channel_blocks_the_claim(self):
        """★ 保守优先：一个渠道查证没有、另一个渠道没查成 → **不能**断言没有。

        没查成的那个渠道里可能正躺着一堆竞品。把"没查成"当成"没有"是本项目
        最危险的一类错误（见模块文档），多渠道路由不该把它重新引进来。
        """
        verified = ResearchOutcome(status="no_competitor", queries=(ok_trace(),))
        unsearchable = ResearchOutcome(status="unsearchable", queries=(empty_trace(),))
        assert verified.merged(unsearchable).status == "unsearchable"
        assert unsearchable.merged(verified).status == "unsearchable"

    def test_merging_failed_gives_up_the_claim(self):
        verified = ResearchOutcome(status="no_competitor", queries=(ok_trace(),))
        failed = ResearchOutcome(status="failed", queries=(failed_trace(),))
        assert verified.merged(failed).status == "unsearchable"

    def test_queries_are_concatenated(self):
        left = ResearchOutcome(queries=(ok_trace("a"),))
        right = ResearchOutcome(queries=(ok_trace("b"),))
        assert [t.query for t in left.merged(right).queries] == ["a", "b"]

    def test_duplicate_findings_are_deduped_by_url(self):
        """同一竞品可能跨渠道被搜到两次，最终清单里只应出现一次。"""
        shared = "https://example.test/same"
        left = ResearchOutcome(findings=(finding("a", shared),), status="ok")
        right = ResearchOutcome(findings=(finding("b", shared),), status="ok")
        merged = left.merged(right)
        assert len(merged.findings) == 1

    def test_warnings_are_kept_from_both_sides(self):
        left = ResearchOutcome(status="failed", warning="左边没查成")
        right = ResearchOutcome(status="failed", warning="右边也失败")
        merged = left.merged(right)
        assert "左边没查成" in (merged.warning or "")
        assert "右边也失败" in (merged.warning or "")

    def test_judgement_failure_propagates(self):
        """★ 任一渠道的判定没做成，整个结论的竞品清单就都是"未经判定"的。

        少了这条传播，A 渠道正常判定、B 渠道判定失败时，合并结论会声称 ``ok``
        而把"没验过"这件事丢掉 —— 卡片上就与一次正常判定**长得一模一样**。
        """
        clean = ResearchOutcome(status="ok", findings=(finding("a"),))
        failed = ResearchOutcome(status="ok", findings=(finding("b"),), judgement_failed=True)

        assert clean.merged(clean).judgement_failed is False
        assert clean.merged(failed).judgement_failed is True
        assert failed.merged(clean).judgement_failed is True


class TestJudgementFailedFlag:
    def test_build_outcome_carries_it(self):
        outcome = build_outcome([empty_trace()], [finding()], subject="防晒", judgement_failed=True)
        assert outcome.judgement_failed is True

    def test_build_outcome_defaults_to_false(self):
        """默认必须是"判定正常"吗？—— 是：这个标记说的是"发生了什么"，
        不是"保守起见说什么"。判定没失败却说失败，只会让提示变成噪音，
        最终人人忽略它。"""
        assert build_outcome([empty_trace()], [], subject="防晒").judgement_failed is False


# --------------------------------------------------------------------------- #
# 警告文案：三种"不是 ok 也不是 no_competitor"必须说成不同的话
# --------------------------------------------------------------------------- #


class TestWarningFor:
    def test_ok_has_no_warning(self):
        assert warning_for("ok", [ok_trace()], subject="防晒") is None

    def test_ok_with_a_failed_query_warns_the_list_may_be_incomplete(self):
        """★ 找到了竞品、但**不是每条检索词都查成** → 必须提示列表不完整。

        这条路径**可达**：``_search_channels`` 在同一渠道内首次失败即 ``break``（避免
        加深限流），前面几条检索词已经拿到的 findings 会保留。于是 ``classify_status``
        因 findings 非空判成 ``ok``、``research_failed`` 为 ``False`` ——
        ``_unresolved_warning`` 抓不到它，少一个入口就会静默。

        缺了这条断言，产物会自相矛盾：失败轨迹原本写着"该簇的竞品空白度必须按中性值
        处理"，而评分并没有退回中性（已拿到真实竞品，见
        ``scoring.opportunity.build_cards``）。处方已移到结论层 —— 说的必须是**实际
        发生的事**：列表不完整，不是"分数按中性值算"。
        """
        text = warning_for("ok", [ok_trace(), failed_trace("笔记 导出 工具")], subject="小红书导出")
        assert text is not None
        assert "可能不完整" in text
        assert "笔记 导出 工具" in text

    def test_ok_stays_silent_when_every_query_succeeded(self):
        """没有失败就没什么可说的 —— 提示一旦变成常态，用户就会开始忽略它。"""
        assert warning_for("ok", [ok_trace(), ok_trace("美妆")], subject="防晒") is None

    def test_ok_warning_names_at_most_three_queries(self):
        """失败词多时只报前三条，避免一整屏检索词把提示淹掉。"""
        text = warning_for(
            "ok",
            [ok_trace()] + [failed_trace(name) for name in ("a", "b", "c", "d")],
            subject="防晒",
        )
        assert text is not None
        assert "「d」" not in text
        assert all(f"「{name}」" in text for name in ("a", "b", "c"))

    def test_unsearchable_says_retrieval_failed_and_hints_at_the_fix(self):
        """检索不到时，文案要说清"检索不到 ≠ 不存在"，并给出可操作的下一步。

        只丢一句"无法判断"会让用户以为这个渠道对他没用 —— 实际上换个更贴近
        "用户会去找什么工具"的说法往往就能搜到。
        """
        text = warning_for("unsearchable", [empty_trace()], subject="防晒搓泥")
        assert text is not None
        assert "没有返回任何结果" in text
        assert "检索不到 ≠ 不存在" in text
        assert "中性值" in text

    def test_failed_says_it_failed_not_absent(self):
        text = warning_for("failed", [failed_trace()], subject="搓泥")
        assert text is not None
        assert "失败" in text
        assert "429 限流" in text
        assert "不代表该方向没有竞品" in text

    def test_no_queries_explains_not_searched(self):
        text = warning_for("unsearchable", [], subject="搓泥")
        assert text is not None
        assert "没有可用的检索词" in text

    def test_verified_empty_quotes_the_queries_that_were_run(self):
        """给出结论时必须**附上实际搜过什么** —— 否则这个结论无法被质疑。"""
        trace = QueryTrace(query="美妆 成分查询", channel="appstore", hits=9, kept=0)
        text = warning_for("no_competitor", [trace], subject="防晒搓泥")
        assert text is not None
        assert "美妆 成分查询" in text
        assert "可逐条复核" in text


class TestBuildOutcome:
    def test_bundles_status_and_warning(self):
        outcome = build_outcome([empty_trace()], [], subject="防晒搓泥")
        assert outcome.status == "unsearchable"
        assert outcome.warning is not None
        assert outcome.inconclusive is True

    def test_verified_empty_has_both_flags_right(self):
        trace = QueryTrace(query="notes export", channel="github", hits=42, kept=0)
        outcome = build_outcome([trace], [], subject="小红书导出")
        assert outcome.verified_empty is True
        assert outcome.inconclusive is False

    def test_findings_shortcut_to_ok(self):
        outcome = build_outcome([empty_trace()], [finding()], subject="任意")
        assert outcome.status == "ok"
        assert outcome.warning is None


# --------------------------------------------------------------------------- #
# 不变式：契约自身不能出现"M1 那种混淆"
# --------------------------------------------------------------------------- #


class TestContractInvariants:
    @pytest.mark.parametrize("hits", [0, 1, 5, 3571])
    def test_empty_findings_never_means_verified_empty_without_hits(self, hits: int) -> None:
        """★ 把 M1 的判据显式写成不变式：**没有命中就不许说"确实没有"**。

        只要平台一条都没返回，结论就只能是 ``unsearchable``；
        反之只要平台返回过东西，就只能是 ``no_competitor``。
        这两条合起来就是 M2 修复的全部内容。
        """
        trace = QueryTrace(query="q", channel="github", hits=hits, kept=0)
        status = classify_status([trace], [])
        if hits == 0:
            assert status == "unsearchable", "0 命中被当成了「查证过没有竞品」"
        else:
            assert status == "no_competitor", "平台返回过东西，却说「无法判断」"

    def test_default_status_is_the_conservative_one(self):
        """默认值必须是"无法判断"而不是"没有竞品"—— 忘记赋值时不许白送一个 1.0。"""
        assert ResearchOutcome().status == "unsearchable"
        assert ResearchOutcome().verified_empty is False
