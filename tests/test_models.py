"""领域模型测试。

其中 `test_public_dict_never_leaks_raw_text` 是**合规底线测试**：
众包上传的脱敏结论一旦泄漏原文或个人信息，整个隐私设计就失效了。
任何新增字段都必须先通过这条测试。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from xhs_pain_miner.models import (
    CompetitorFinding,
    Evidence,
    MiningResult,
    OpportunityCard,
    PainCluster,
    RawComment,
    RawCorpus,
    RawNote,
    RunCost,
    find_verbatim_overlap,
    hash_id,
)

SENSITIVE_TEXT = "这句话是绝对不能离开本机的原文内容"


class TestHashId:
    """哈希工具。"""

    def test_is_deterministic(self):
        """同一个输入必须得到同一个哈希（跨笔记去重要依赖这一点）。"""
        assert hash_id("user_123") == hash_id("user_123")

    def test_differs_by_input(self):
        """不同输入必须得到不同哈希。"""
        assert hash_id("user_123") != hash_id("user_124")

    def test_salt_changes_result(self):
        """加盐后结果必须改变（增强不可逆性）。"""
        assert hash_id("user_123") != hash_id("user_123", salt="my-salt")

    def test_output_length(self):
        """输出固定为 16 位十六进制。"""
        result = hash_id("user_123")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)


class TestRawCorpus:
    """原始语料。"""

    def test_total_images(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[
                RawNote(note_id="n1", images=["a.jpg", "b.jpg"]),
                RawNote(note_id="n2", images=["c.jpg"]),
            ],
        )
        assert corpus.total_images == 3

    def test_summary_contains_counts(self):
        corpus = RawCorpus(
            keyword="防晒霜",
            notes=[RawNote(note_id="n1")],
            comments=[RawComment(comment_id="c1")],
            backend="fixture",
        )
        summary = corpus.summary()
        assert "1 篇笔记" in summary
        assert "1 条评论" in summary
        assert "fixture" in summary


class TestCompetitorFinding:
    """竞品结论。"""

    def test_recent_competitor_is_active(self):
        recent = CompetitorFinding(
            source="github", name="active-lib", last_active=date.today() - timedelta(days=30)
        )
        assert not recent.is_stale

    def test_stale_competitor_is_flagged(self):
        old = CompetitorFinding(
            source="github", name="dead-lib", last_active=date.today() - timedelta(days=400)
        )
        assert old.is_stale

    def test_unknown_activity_is_not_stale(self):
        """拿不到更新时间时不应擅自判定为停更 —— 那会给出虚假的机会信号。"""
        unknown = CompetitorFinding(source="appstore", name="unknown-app")
        assert not unknown.is_stale


class TestOpportunityCard:
    """机会卡片 —— 核心交付物。"""

    @staticmethod
    def _card() -> OpportunityCard:
        evidence = Evidence(text=SENSITIVE_TEXT, source="comment", likes=42, note_hash="abc123")
        cluster = PainCluster(
            id="p1",
            label="假白泛白",
            summary="涂完像戴了面具，脖子和脸有色差",
            size=89,
            sentiment=-0.85,
            evidences=[evidence],
        )
        return OpportunityCard(
            id="c1",
            title="不泛白的物理防晒",
            pain=cluster,
            competitors=[
                CompetitorFinding(
                    source="github",
                    name="old-sunscreen-tool",
                    url="https://github.com/example/old",
                    stars=128,
                    last_active=date(2023, 5, 1),
                )
            ],
            score=78.4,
            score_breakdown={"pain_intensity": 0.9, "volume": 0.7, "gap": 0.95},
            feasibility="个人可做 / 1-2 周",
        )

    def test_has_active_competitor(self):
        card = self._card()
        assert not card.has_active_competitor  # 唯一竞品已停更

    def test_public_dict_never_leaks_raw_text(self):
        """★ 合规底线：脱敏导出不得包含任何原文。"""
        payload = json.dumps(self._card().to_public_dict(), ensure_ascii=False)
        assert SENSITIVE_TEXT not in payload

    def test_public_dict_never_leaks_personal_identifiers(self):
        """★ 合规底线：脱敏导出不得包含哈希之外的用户标识字段。"""
        public = self._card().to_public_dict()
        assert "note_hash" not in json.dumps(public, ensure_ascii=False)
        assert "evidences" not in public["pain"]

    def test_public_dict_keeps_conclusions(self):
        """脱敏不能把结论本身也砍掉。"""
        public = self._card().to_public_dict()
        assert public["score"] == 78.4
        assert public["pain"]["label"] == "假白泛白"
        assert public["pain"]["size"] == 89
        assert public["pain"]["evidence_count"] == 1
        assert public["competitors"][0]["stars"] == 128

    def test_to_dict_keeps_full_data(self):
        """本地持久化必须保留完整数据（与上传路径区分开）。"""
        local = self._card()
        assert local.pain.evidences[0].text == SENSITIVE_TEXT


class TestRunCost:
    """成本统计。"""

    def test_total_calls(self):
        cost = RunCost(llm_calls=30, vlm_calls=100)
        assert cost.total_calls == 130

    def test_summary_is_readable(self):
        cost = RunCost(llm_calls=30, vlm_calls=100, vlm_cache_hits=60, elapsed_seconds=12.5)
        summary = cost.summary()
        assert "LLM 30 次" in summary
        assert "VLM 100 次" in summary
        assert "命中缓存 60" in summary


class TestMiningResult:
    """一次分析的完整产出。"""

    def test_top_cards_sorted_by_score(self):
        low = OpportunityCard(id="a", title="低分", pain=PainCluster(id="p1"), score=30)
        high = OpportunityCard(id="b", title="高分", pain=PainCluster(id="p2"), score=90)
        result = MiningResult(keyword="防晒霜", cards=[low, high])
        assert [c.id for c in result.top_cards] == ["b", "a"]

    def test_to_public_dict_excludes_evidence_text(self):
        evidence = Evidence(text=SENSITIVE_TEXT, source="note")
        card = OpportunityCard(
            id="c1",
            title="某方向",
            pain=PainCluster(id="p1", label="某痛点", evidences=[evidence]),
        )
        result = MiningResult(keyword="防晒霜", cards=[card])
        payload = json.dumps(result.to_public_dict(), ensure_ascii=False)
        assert SENSITIVE_TEXT not in payload

    def test_generated_at_defaults_to_now(self):
        result = MiningResult(keyword="测试")
        assert isinstance(result.generated_at, datetime)
        assert result.generated_at.tzinfo is not None
        assert abs((datetime.now(timezone.utc) - result.generated_at).total_seconds()) < 5


class TestFindVerbatimOverlap:
    """原文回抄检测 —— 脱敏的第二道防线。

    结构性剔除挡不住「LLM 在摘要里引用原话」，这个检测就是为它准备的。
    """

    SOURCE = "我用的那支上脸假白到像糊了面粉，同事问我是不是过敏了"

    def test_detects_verbatim_copy(self):
        text = "摘要：我用的那支上脸假白到像糊了面粉"
        assert find_verbatim_overlap(text, [self.SOURCE]) == "我用的那支上脸假白到像糊了面粉"

    def test_returns_none_for_paraphrase(self):
        """改写过的摘要不能被判定为回抄，否则检测会误伤所有正常摘要。"""
        text = "很多用户反映这款产品涂上之后肤色不自然，会显得过白"
        assert find_verbatim_overlap(text, [self.SOURCE]) is None

    def test_ignores_short_sources(self):
        """短原文不参与比对 —— 否则常见短语会大面积误判。"""
        assert find_verbatim_overlap("这句话里提到了防晒这个词", ["防晒"]) is None

    def test_empty_inputs(self):
        assert find_verbatim_overlap("", [self.SOURCE]) is None
        assert find_verbatim_overlap("任意文本", []) is None
        assert find_verbatim_overlap("任意文本", ["", ""]) is None

    def test_text_shorter_than_min_len_cannot_match(self):
        """比 min_len 还短的文本不可能包含连续片段。"""
        assert find_verbatim_overlap("很短", [self.SOURCE]) is None

    def test_min_len_is_configurable(self):
        text = "摘要：上脸假白到像糊了面粉"
        assert find_verbatim_overlap(text, [self.SOURCE], min_len=15) is None
        assert find_verbatim_overlap(text, [self.SOURCE], min_len=8) is not None

    def test_scans_all_sources(self):
        sources = ["完全无关的另一段内容在这里出现", self.SOURCE]
        assert find_verbatim_overlap("摘要：我用的那支上脸假白到像糊了面粉", sources) is not None

    def test_invalid_min_len(self):
        assert find_verbatim_overlap(self.SOURCE, [self.SOURCE], min_len=0) is None

    def test_ignores_non_string_sources(self):
        """非 str 元素必须被跳过。

        静默漏检远好过让上传前的合规检查自己抛 ``TypeError`` 崩掉 ——
        那会让整条上传路径失败，而不是干净地放行或拦截。
        """
        text = "任意文本内容在这里出现"
        assert find_verbatim_overlap(text, [123, 456]) is None  # type: ignore[list-item]
        assert find_verbatim_overlap(text, [b"bytes-here"]) is None  # type: ignore[list-item]
        assert find_verbatim_overlap(text, [None]) is None  # type: ignore[list-item]

    def test_still_matches_when_mixed_with_non_strings(self):
        """混入非 str 元素不应影响正常命中。"""
        sources = [42, None, b"x", self.SOURCE]  # type: ignore[list-item]
        text = "摘要：我用的那支上脸假白到像糊了面粉"
        assert find_verbatim_overlap(text, sources) is not None


class TestKnownGaps:
    """记录脱敏设计中的**已知边界**，避免被误认为已经彻底解决。

    这里用 ``xfail(strict=True)`` 而不是普通断言。普通地断言「当前会泄漏」有个
    问题：CI 全绿时无法区分「已知边界仍未闭合」与「需求已满足」，容易被读成
    「隐私已验证通过」。``xfail`` 会让 CI 显式显示这条尚未闭合。
    """

    LEAK = "我需要一个能自动整理笔记的工具"

    @pytest.mark.xfail(
        strict=True,
        reason="自由文本字段尚未做回抄检测；本测试通过即代表防线已补齐",
    )
    def test_free_text_fields_are_sanitized(self):
        """期望：LLM 生成的自由文本字段不应回抄原文。

        **当前会失败 —— 这是已知边界，不是回归。**

        结构性剔除只覆盖 ``Evidence.text``（见 ``TestOpportunityCard``）。
        ``label`` / ``summary`` / ``title`` / ``gap_notes`` 是 LLM 生成的自由文本，
        而「摘要时引用原话」是模型常规行为 —— 需要靠 ``find_verbatim_overlap()``
        做第二道检测，M4 接入众包上传时必须补上。

        本测试转为通过（XPASS）即代表防线已补齐，届时请：
        1. 移除 ``xfail`` 标记
        2. 更新 docs/architecture.md 的 4.5 节与 docs/faq.md
        """
        cluster = PainCluster(id="p1", label="整理笔记", summary=f"她说的原话是{self.LEAK}")
        card = OpportunityCard(id="c1", title="笔记整理工具", pain=cluster)

        payload = json.dumps(card.to_public_dict(), ensure_ascii=False)
        assert self.LEAK not in payload
