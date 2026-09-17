"""渲染层测试 —— 安全、降级、两种产物的分工。

本文件的重点不是"能不能渲染出 HTML"，而是三件容易出事的事：

1. **HTML 转义**：卡片字段来自采集内容（不可信输入）。断言必须落在"输出里没有
   未转义的 ``<script``"上，而不是"没抛异常"—— 后者是恒真断言，守不住任何东西。
2. **调研结论的四种状态必须说成四句不同的话**（查到 / 查证过没有 / 检索不到 /
   没查成）。判据是卡片上的 ``research_status`` —— M1 靠 ``competitor_gap == 0.5``
   反推，那条推理有精确碰撞（见
   ``TestHtmlResearchState::test_two_cold_competitors_do_not_look_like_a_failure``）。
3. **Markdown 不含证据原文**：这是与 HTML 产物刻意的设计差异，用一条守卫测试钉住。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

from xhs_pain_miner.models import (
    CompetitorFinding,
    Evidence,
    MiningResult,
    OpportunityCard,
    PainCluster,
    QueryTrace,
    ResearchStatus,
    RunCost,
)
from xhs_pain_miner.render.html import DEFAULT_TITLE, render_html, write_html
from xhs_pain_miner.render.markdown import render_markdown

XSS_SCRIPT = "<script>alert(1)</script>"
XSS_IMG = '<img src=x onerror="alert(1)">'
XSS_BREAKOUT = '"><script>alert(2)</script>'
RAW_TEXT = "我用的那支上脸假白到像糊了面粉，同事问我是不是过敏了，这句话只属于本机"
OLD_DATE = date(2023, 5, 1)


def make_cluster(
    *,
    id_: str = "p1",
    label: str = "假白泛白",
    summary: str = "涂完像戴了面具，脖子和脸有色差",
    size: int = 89,
    sentiment: float = -0.85,
    evidences: list[Evidence] | None = None,
    category: str = "结果不达预期",
    stage: str = "growing",
    difficulty: int = 2,
    feasibility: str = "个人可做 / 1-2 周",
    is_noise: bool = False,
) -> PainCluster:
    return PainCluster(
        id=id_,
        label=label,
        summary=summary,
        size=size,
        sentiment=sentiment,
        evidences=evidences
        if evidences is not None
        else [Evidence(text=RAW_TEXT, source="comment", likes=42)],
        category=category,
        stage=stage,  # type: ignore[arg-type]
        difficulty=difficulty,
        feasibility=feasibility,
        is_noise=is_noise,
    )


def make_card(
    cluster: PainCluster | None = None,
    *,
    id_: str = "card-p1",
    title: str = "防晒霜 · 更省事的工具",
    competitors: list[CompetitorFinding] | None = None,
    score: float = 78.4,
    score_breakdown: dict[str, float] | None = None,
    feasibility: str = "个人可做 / 1-2 周",
    research_status: ResearchStatus = "no_competitor",
    research_queries: tuple[QueryTrace, ...] = (),
    research_judgement_failed: bool = False,
) -> OpportunityCard:
    """造一张卡片。

    ``research_status`` 的默认值 ``no_competitor``（"查证过，没有竞品"）与另外两个
    默认值（``competitors=[]``、``competitor_gap=1.0``）是配套的 —— 这三个字段说的
    必须是同一件事，否则造出来的卡片本身就是不自洽的。
    """
    return OpportunityCard(
        id=id_,
        title=title,
        pain=cluster if cluster is not None else make_cluster(),
        competitors=competitors if competitors is not None else [],
        score=score,
        score_breakdown=score_breakdown
        if score_breakdown is not None
        else {
            "pain_strength": 0.92,
            "mention_volume": 0.71,
            "growth_trend": 0.55,
            "competitor_gap": 1.0,
            "feasibility": 0.75,
        },
        feasibility=feasibility,
        research_status=research_status,
        research_queries=research_queries,
        research_judgement_failed=research_judgement_failed,
    )


def make_result(
    *,
    keyword: str = "防晒霜",
    cards: list[OpportunityCard] | None = None,
    clusters: list[PainCluster] | None = None,
    notes: list[str] | None = None,
) -> MiningResult:
    card_list = cards if cards is not None else [make_card()]
    return MiningResult(
        keyword=keyword,
        cards=card_list,
        clusters=clusters if clusters is not None else [card.pain for card in card_list],
        total_notes=200,
        total_comments=2400,
        generated_at=datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc),
        cost=RunCost(llm_calls=35, vlm_calls=12, elapsed_seconds=93.5),
        notes=notes if notes is not None else [],
    )


def _assert_payload_is_inert(document: str) -> None:
    """断言注入载荷没有变成标记。"""
    assert "<script" not in document
    assert "<img" not in document
    assert "<iframe" not in document


# --------------------------------------------------------------------------- #
# HTML：转义（本模块最重要的一组）
# --------------------------------------------------------------------------- #


class TestHtmlEscaping:
    """★ 安全测试：每个字段都要单独撞一次。

    逐字段构造的意义在于**失败时能直接定位到漏转义的字段**，而不是拿到一句
    "报告里有未转义的 script"。
    """

    def test_script_in_report_title_is_escaped(self):
        document = render_html(make_result(keyword=XSS_SCRIPT))
        _assert_payload_is_inert(document)
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document

    def test_script_in_explicit_title_is_escaped(self):
        _assert_payload_is_inert(render_html(make_result(), title=XSS_SCRIPT))

    def test_script_in_card_title_is_escaped(self):
        _assert_payload_is_inert(render_html(make_result(cards=[make_card(title=XSS_SCRIPT)])))

    def test_script_in_pain_label_is_escaped(self):
        _assert_payload_is_inert(
            render_html(make_result(cards=[make_card(make_cluster(label=XSS_SCRIPT))]))
        )

    def test_img_onerror_in_pain_label_is_escaped(self):
        document = render_html(make_result(cards=[make_card(make_cluster(label=XSS_IMG))]))
        _assert_payload_is_inert(document)
        assert "&lt;img src=x" in document
        assert 'onerror="alert(1)"' not in document  # 属性逃逸必须失败
        assert "&quot;" in document  # 引号确实被转义成了实体

    def test_script_in_summary_is_escaped(self):
        _assert_payload_is_inert(
            render_html(make_result(cards=[make_card(make_cluster(summary=XSS_SCRIPT))]))
        )

    def test_script_in_category_is_escaped(self):
        _assert_payload_is_inert(
            render_html(make_result(cards=[make_card(make_cluster(category=XSS_SCRIPT))]))
        )

    def test_script_in_cluster_id_is_escaped(self):
        """簇 id 会进 ``id`` 属性与卡片 id —— 属性上下文同样要转义。"""
        _assert_payload_is_inert(
            render_html(make_result(cards=[make_card(make_cluster(id_=XSS_BREAKOUT))]))
        )

    def test_script_in_evidence_text_is_escaped(self):
        """证据原文是最典型的不可信输入（采集内容）。"""
        evidence = Evidence(text=XSS_SCRIPT, source="comment", likes=3)
        document = render_html(make_result(cards=[make_card(make_cluster(evidences=[evidence]))]))
        _assert_payload_is_inert(document)
        assert "&lt;script&gt;" in document

    def test_script_in_competitor_name_is_escaped(self):
        finding = CompetitorFinding(source="github", name=XSS_SCRIPT, url="https://e.test/x")
        _assert_payload_is_inert(render_html(make_result(cards=[make_card(competitors=[finding])])))

    def test_script_in_competitor_gap_notes_is_escaped(self):
        finding = CompetitorFinding(
            source="github", name="tool", url="https://e.test/x", gap_notes=XSS_SCRIPT
        )
        _assert_payload_is_inert(render_html(make_result(cards=[make_card(competitors=[finding])])))

    def test_script_in_competitor_description_is_escaped(self):
        """平台描述是**第三方自由文本**（GitHub 仓库描述 / App Store 商店文案），
        由平台用户自己填 —— 它此前只进判定提示词，进渲染层就是新开的一处攻击面。
        """
        finding = CompetitorFinding(
            source="appstore", name="tool", url="https://e.test/x", description=XSS_SCRIPT
        )
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        _assert_payload_is_inert(document)
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document

    def test_img_onerror_in_competitor_description_is_escaped(self):
        """属性逃逸必须失败：描述里的引号与尖括号都要变成实体。"""
        finding = CompetitorFinding(
            source="appstore", name="tool", url="https://e.test/x", description=XSS_IMG
        )
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        _assert_payload_is_inert(document)
        assert 'onerror="alert(1)"' not in document
        assert "&quot;" in document

    def test_script_in_competitor_url_cannot_break_out_of_href(self):
        finding = CompetitorFinding(
            source="github", name="tool", url=f"https://evil.test/{XSS_BREAKOUT}"
        )
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        _assert_payload_is_inert(document)
        assert "&quot;&gt;" in document  # 引号与尖括号都被转义

    def test_script_in_notes_is_escaped(self):
        _assert_payload_is_inert(render_html(make_result(notes=[XSS_SCRIPT])))

    def test_script_in_keyword_and_notes_together(self):
        _assert_payload_is_inert(render_html(make_result(keyword=XSS_IMG, notes=[XSS_BREAKOUT])))

    def test_plain_text_is_not_over_escaped_into_garbage(self):
        """转义不能把正常中文与标点弄坏。"""
        document = render_html(make_result())
        assert "假白泛白" in document
        assert "涂完像戴了面具" in document


class TestHtmlUrlHardening:
    """转义拦不住 ``javascript:`` —— 那是合法 URL，只能靠协议白名单。"""

    def test_javascript_url_is_not_turned_into_a_link(self):
        finding = CompetitorFinding(source="github", name="sneaky", url="javascript:alert(1)")
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        assert "javascript:" not in document
        assert "sneaky" in document  # 名字仍然展示，只是不可点

    def test_url_with_control_characters_is_not_linked(self):
        finding = CompetitorFinding(source="github", name="sneaky", url="java\nscript:alert(1)")
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        assert "javascript:" not in document

    def test_https_url_is_linked(self):
        finding = CompetitorFinding(
            source="github", name="real", url="https://github.com/example/real"
        )
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        assert 'href="https://github.com/example/real"' in document


class TestHtmlSelfContained:
    """自包含：内联样式、无外部资源、无 JavaScript。"""

    def test_document_shell(self):
        document = render_html(make_result())
        assert document.startswith("<!DOCTYPE html>")
        assert '<html lang="zh-CN">' in document
        assert '<meta charset="utf-8">' in document
        assert "<style>" in document
        assert document.rstrip().endswith("</html>")

    def test_no_external_resources(self):
        document = render_html(make_result())
        for marker in ("<link", "@import", "url(", "<script", "<iframe", "<img"):
            assert marker not in document, marker

    def test_default_title_uses_keyword(self):
        assert "防晒霜 机会卡片" in render_html(make_result())

    def test_empty_keyword_falls_back_to_default_title(self):
        assert DEFAULT_TITLE in render_html(make_result(keyword=""))


class TestHtmlResearchState:
    """★ 四种调研结论必须说成四句不同的话。

    M1 靠 ``competitor_gap == 0.5`` 反推"调研失败"，M2 改成读
    ``card.research_status``。这里除了四种状态各自的措辞，还钉住了那条反推法的
    精确碰撞（见 :meth:`test_two_cold_competitors_do_not_look_like_a_failure`）。
    """

    def test_failed_research_is_not_reported_as_no_competitor(self):
        card = make_card(
            competitors=[],
            research_status="failed",
            score_breakdown={
                "pain_strength": 0.9,
                "mention_volume": 0.7,
                "growth_trend": 0.5,
                "competitor_gap": 0.5,  # 中性值
                "feasibility": 0.75,
            },
        )
        document = render_html(make_result(cards=[card]))
        assert "调研失败" in document
        assert "没有相关竞品" not in document
        assert "不代表该方向没有竞品" in document

    def test_unsearchable_is_not_reported_as_no_competitor(self):
        """★ "检索不到"与"查证过确实没有"是两件事，措辞与下一步动作都不同。

        括号里那句"也可能是这次没有可用的检索词"是刻意加的：``unsearchable``
        同时覆盖"检索词没返回东西"与"压根没有可用的检索词"（检索词生成失败、
        调研被关闭），只写前者会在后一种情形下变成一句失实的话。
        """
        card = make_card(
            competitors=[],
            research_status="unsearchable",
            score_breakdown={"competitor_gap": 0.5},
        )
        document = render_html(make_result(cards=[card]))
        assert "检索不到" in document
        assert "没有相关竞品" not in document
        assert "也可能是这次没有可用的检索词" in document

    def test_unsearchable_and_failed_say_different_things(self):
        def verdict(text: str) -> str:
            return text.split('class="verdict"')[1].split("</p>")[0]

        unsearchable = render_html(
            make_result(cards=[make_card(competitors=[], research_status="unsearchable")])
        )
        failed = render_html(
            make_result(cards=[make_card(competitors=[], research_status="failed")])
        )
        assert verdict(unsearchable) != verdict(failed)

    def test_verified_empty_market_says_no_competitor(self):
        card = make_card(
            competitors=[], research_status="no_competitor", score_breakdown={"competitor_gap": 1.0}
        )
        document = render_html(make_result(cards=[card]))
        assert "没有相关竞品" in document
        assert "调研失败" not in document
        assert "检索不到" not in document

    def test_two_cold_competitors_do_not_look_like_a_failure(self):
        """★ 反推法的精确碰撞：2 个零 star 的活跃竞品，空白度恰好也是 0.5。

        ``0.60 × (1 - 0.5 × 0) = 0.5`` —— 与中性值逐位相同。M1 的渲染层据此多印
        一句"（本次调研未完成，结果可能不完整）"，而这张卡片上明明列着 2 个竞品。
        改成读结论类别后，有竞品 ⇒ ``ok``，与空白度是多少无关。
        """
        cold = [
            CompetitorFinding(
                source="github",
                name=f"cold-{i}",
                url=f"https://e.test/cold-{i}",
                stars=0,
                last_active=date.today(),
            )
            for i in range(2)
        ]
        card = make_card(
            competitors=cold,
            research_status="ok",
            score_breakdown={"competitor_gap": 0.5},
        )
        document = render_html(make_result(cards=[card]))
        assert "活跃维护" in document
        assert "调研未完成" not in document
        assert "结果可能不完整" not in document

    def test_all_stale_competitors_are_reported_honestly(self):
        stale = CompetitorFinding(
            source="github", name="old", url="https://e.test/old", stars=128, last_active=OLD_DATE
        )
        document = render_html(
            make_result(cards=[make_card(competitors=[stale], research_status="ok")])
        )
        assert "均已停更" in document
        assert "2023-05-01" in document

    def test_active_competitors_are_reported(self):
        active = CompetitorFinding(
            source="github",
            name="live",
            url="https://e.test/live",
            stars=9000,
            last_active=date.today(),
        )
        document = render_html(
            make_result(cards=[make_card(competitors=[active], research_status="ok")])
        )
        assert "活跃维护" in document
        assert "9000" in document


class TestHtmlMissingFields:
    """缺字段时少显示，而不是崩溃 —— 真实语料里这些字段大量为空。"""

    def test_minimal_card_renders(self):
        cluster = make_cluster(
            label="", summary="", evidences=[], category="", feasibility="", size=0, sentiment=0.0
        )
        card = make_card(cluster, title="", feasibility="", score=0.0, score_breakdown={})
        document = render_html(make_result(cards=[card]))
        assert "<!DOCTYPE html>" in document
        assert "未评估" in document  # 可行度缺失时如实说"未评估"

    def test_result_without_cards_renders(self):
        document = render_html(make_result(cards=[], clusters=[]))
        assert "<!DOCTYPE html>" in document
        assert "机会卡片" in document

    def test_evidence_without_timestamp_renders(self):
        evidence = Evidence(text="没有时间信息", source="note", likes=0, created_at=None)
        document = render_html(make_result(cards=[make_card(make_cluster(evidences=[evidence]))]))
        assert "没有时间信息" in document

    def test_evidence_with_timestamp_shows_date(self):
        evidence = Evidence(
            text="有时间的证据",
            source="comment",
            likes=7,
            created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        document = render_html(make_result(cards=[make_card(make_cluster(evidences=[evidence]))]))
        assert "2026-08-01" in document

    def test_competitor_without_stars_or_activity_renders(self):
        finding = CompetitorFinding(
            source="github", name="mystery", url="", stars=None, last_active=None
        )
        document = render_html(make_result(cards=[make_card(competitors=[finding])]))
        assert "stars 未知" in document
        assert "最后活跃：未知" in document

    def test_evidence_over_limit_is_truncated_with_a_note(self):
        evidences = [Evidence(text=f"证据 {i}", source="comment", likes=i) for i in range(9)]
        document = render_html(
            make_result(cards=[make_card(make_cluster(evidences=evidences))]), max_evidence=3
        )
        assert "证据 0" in document
        assert "证据 3" not in document
        assert "另有 6 条证据" in document

    def test_max_evidence_zero_hides_all_evidence(self):
        document = render_html(make_result(), max_evidence=0)
        assert RAW_TEXT not in document
        # 没内嵌 ≠ 没有证据：条数仍然要报出来
        assert "该簇共 1 条" in document

    def test_noise_cards_are_excluded_by_default(self):
        noise = make_card(
            make_cluster(id_="noise", is_noise=True, label="长尾怪癖"), id_="card-noise"
        )
        normal = make_card()
        document = render_html(make_result(cards=[noise, normal]))
        assert "长尾怪癖" not in document
        assert "假白泛白" in document

    def test_noise_cards_can_be_included(self):
        noise = make_card(
            make_cluster(id_="noise", is_noise=True, label="长尾怪癖"), id_="card-noise"
        )
        document = render_html(make_result(cards=[noise]), include_noise=True)
        assert "长尾怪癖" in document
        assert "长尾低频痛点" in document

    def test_weights_are_only_shown_when_they_can_be_recovered(self):
        """卡片不保存权重：默认口径可以还原，自定义口径必须如实说明而不是编一个。"""
        consistent = make_card(
            score=77.5,
            score_breakdown={
                "pain_strength": 0.5,
                "mention_volume": 1.0,
                "growth_trend": 0.5,
                "competitor_gap": 1.0,
                "feasibility": 1.0,
            },
        )
        document = render_html(make_result(cards=[consistent]))
        assert "权重 25%" in document

        inconsistent = make_card(score=42.0, score_breakdown={"pain_strength": 0.5})
        document = render_html(make_result(cards=[inconsistent]))
        assert "自定义权重" in document


class TestWriteHtml:
    """文件写入。"""

    def test_writes_utf8_and_returns_absolute_path(self, tmp_path: Path):
        target = tmp_path / "报告" / "cards.html"
        written = write_html(make_result(keyword="防晒霜"), target)
        assert written.is_absolute()
        assert written.exists()
        content = written.read_text(encoding="utf-8")
        assert "防晒霜" in content
        assert content == render_html(make_result(keyword="防晒霜"))

    def test_bytes_are_valid_utf8(self, tmp_path: Path):
        target = write_html(make_result(), tmp_path / "cards.html")
        target.read_bytes().decode("utf-8")  # 解不开就会抛

    def test_parent_directory_is_created(self, tmp_path: Path):
        target = tmp_path / "a" / "b" / "c.html"
        write_html(make_result(), target)
        assert target.parent.is_dir()

    def test_overwrites_existing_file(self, tmp_path: Path):
        target = tmp_path / "cards.html"
        write_html(make_result(keyword="第一次"), target)
        write_html(make_result(keyword="第二次"), target)
        content = target.read_text(encoding="utf-8")
        assert "第二次" in content
        assert "第一次" not in content


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


class TestMarkdownNoRawText:
    """★ 守卫测试：Markdown 产物一个字原文都不能有。"""

    def test_evidence_text_never_appears(self):
        result = make_result()
        markdown = render_markdown(result)
        assert RAW_TEXT not in markdown

    def test_evidence_text_never_appears_even_with_noise_included(self):
        noise = make_card(
            make_cluster(id_="n", is_noise=True, evidences=[Evidence(text=RAW_TEXT, source="note")])
        )
        markdown = render_markdown(make_result(cards=[noise]), include_noise=True)
        assert RAW_TEXT not in markdown

    def test_conclusion_fields_do_appear(self):
        """不嵌原文 ≠ 砍掉结论：痛点名、摘要、因子、竞品都得在。"""
        markdown = render_markdown(make_result())
        assert "假白泛白" in markdown
        assert "涂完像戴了面具" in markdown
        assert "痛点强度" in markdown
        assert "78/100" in markdown

    def test_evidence_count_is_reported_without_the_text(self):
        markdown = render_markdown(make_result())
        assert "**证据**：1 条" in markdown


class TestMarkdownEscaping:
    """Markdown 不是纯文本：``|`` 与 ``#`` 会破坏结构，``<tag>`` 会被渲染。"""

    def test_pipe_and_hash_are_escaped(self):
        markdown = render_markdown(
            make_result(cards=[make_card(make_cluster(label="假白|泛白#1"))])
        )
        assert "假白\\|泛白\\#1" in markdown

    def test_newlines_are_flattened(self):
        markdown = render_markdown(
            make_result(cards=[make_card(make_cluster(summary="第一行\n第二行"))])
        )
        assert "第一行 第二行" in markdown
        assert "\n第二行" not in markdown

    def test_inline_html_is_escaped(self):
        markdown = render_markdown(make_result(cards=[make_card(make_cluster(label=XSS_SCRIPT))]))
        assert "<script" not in markdown
        assert "&lt;script&gt;" in markdown

    def test_competitor_description_pipe_and_hash_are_escaped(self):
        """竞品描述是自由文本，里面一个 ``|`` 或 ``#`` 就够撑断/改写结构。"""
        finding = CompetitorFinding(
            source="appstore", name="tool", url="https://e.test/x", description="假白|泛白#1"
        )
        markdown = render_markdown(make_result(cards=[make_card(competitors=[finding])]))
        assert "假白\\|泛白\\#1" in markdown

    def test_competitor_description_inline_html_is_escaped(self):
        """``<tag>`` 是 Markdown 的行内 HTML，GitHub / Notion 都会渲染它。"""
        finding = CompetitorFinding(
            source="appstore", name="tool", url="https://e.test/x", description=XSS_SCRIPT
        )
        markdown = render_markdown(make_result(cards=[make_card(competitors=[finding])]))
        assert "<script" not in markdown
        assert "&lt;script&gt;" in markdown

    def test_competitor_description_newline_is_flattened(self):
        """裸换行会从引用块里逃出去 —— 商店文案恰恰大量使用 ``\\n\\n`` 排版。"""
        finding = CompetitorFinding(
            source="appstore", name="tool", url="https://e.test/x", description="第一行\n第二行"
        )
        markdown = render_markdown(make_result(cards=[make_card(competitors=[finding])]))
        assert "第一行 第二行" in markdown
        assert "\n第二行" not in markdown

    def test_javascript_url_is_not_linked(self):
        finding = CompetitorFinding(source="github", name="sneaky", url="javascript:alert(1)")
        markdown = render_markdown(make_result(cards=[make_card(competitors=[finding])]))
        assert "javascript:" not in markdown
        assert "sneaky" in markdown

    def test_https_competitor_url_is_linked(self):
        finding = CompetitorFinding(
            source="github", name="real", url="https://github.com/example/real", stars=12
        )
        markdown = render_markdown(make_result(cards=[make_card(competitors=[finding])]))
        assert "[real](https://github.com/example/real)" in markdown
        assert "12★" in markdown


class TestMarkdownLayout:
    def test_starts_with_keyword_heading(self):
        assert render_markdown(make_result()).startswith("# 防晒霜 机会卡片")

    def test_empty_keyword_heading(self):
        assert render_markdown(make_result(keyword="")).startswith("# 机会卡片")

    def test_cards_are_ordered_by_score(self):
        low = make_card(make_cluster(id_="low", label="低分痛点"), id_="card-low", score=30.0)
        high = make_card(make_cluster(id_="high", label="高分痛点"), id_="card-high", score=90.0)
        markdown = render_markdown(make_result(cards=[low, high]))
        assert markdown.index("高分痛点") < markdown.index("低分痛点")
        assert markdown.index("1. 90/100") < markdown.index("2. 30/100")

    def test_max_cards_limits_output(self):
        cards = [
            make_card(make_cluster(id_=f"p{i}", label=f"痛点{i}"), id_=f"card-{i}", score=90.0 - i)
            for i in range(5)
        ]
        markdown = render_markdown(make_result(cards=cards), max_cards=2)
        assert "痛点0" in markdown
        assert "痛点1" in markdown
        assert "痛点2" not in markdown
        assert "已按 max_cards 截断" in markdown

    def test_noise_excluded_by_default(self):
        noise = make_card(make_cluster(id_="n", label="长尾怪癖", is_noise=True), id_="card-n")
        markdown = render_markdown(make_result(cards=[make_card(), noise]))
        assert "长尾怪癖" not in markdown

    def test_research_failure_is_not_reported_as_empty_market(self):
        card = make_card(
            competitors=[], research_status="failed", score_breakdown={"competitor_gap": 0.5}
        )
        markdown = render_markdown(make_result(cards=[card]))
        assert "调研失败" in markdown
        assert "没有相关竞品" not in markdown

    def test_unsearchable_is_not_reported_as_empty_market(self):
        """★ 分享出去的那一份更不能把"检索不到"写成"未发现竞品"。"""
        card = make_card(
            competitors=[], research_status="unsearchable", score_breakdown={"competitor_gap": 0.5}
        )
        markdown = render_markdown(make_result(cards=[card]))
        assert "检索不到" in markdown
        assert "没有相关竞品" not in markdown

    def test_failed_and_unsearchable_read_differently(self):
        failed = render_markdown(
            make_result(cards=[make_card(competitors=[], research_status="failed")])
        )
        unsearchable = render_markdown(
            make_result(cards=[make_card(competitors=[], research_status="unsearchable")])
        )
        assert failed != unsearchable

    def test_empty_result_renders(self):
        markdown = render_markdown(make_result(cards=[], clusters=[]))
        assert markdown.startswith("# 防晒霜 机会卡片")
        assert "没有可展示的卡片" in markdown

    def test_minimal_card_renders(self):
        cluster = make_cluster(label="", summary="", evidences=[], category="", feasibility="")
        markdown = render_markdown(make_result(cards=[make_card(cluster, feasibility="")]))
        assert "（未命名）" in markdown

    def test_notes_are_surfaced(self):
        markdown = render_markdown(make_result(notes=["VLM 分析缺失"]))
        assert "运行提示" in markdown
        assert "VLM 分析缺失" in markdown


# --------------------------------------------------------------------------- #
# 检索轨迹 —— 「结论可逐条复核」的落地
# --------------------------------------------------------------------------- #


class TestResearchTraces:
    """报告里必须看得到"这个结论是怎么得出来的"。

    M2 之前，结论文案写着"以上结论附带完整检索轨迹，可逐条复核"，而产物里
    根本没有轨迹 —— 一句写进交付物的空头承诺。轨迹是"结论可被质疑"的唯一入口，
    而"能被质疑"正是本产品对"免费的 LLM 摘要"的正面防守。
    """

    TRACES = (
        QueryTrace(query="美妆 成分查询", channel="appstore", hits=9, kept=0),
        QueryTrace(query="cosmetic ingredient lookup", channel="github", hits=12, kept=1),
    )

    def test_markdown_lists_every_query_with_hits_and_kept(self):
        card = make_card(research_queries=self.TRACES)
        markdown = render_markdown(make_result(cards=[card]))

        assert "检索轨迹" in markdown
        for trace in self.TRACES:
            assert trace.query in markdown, f"轨迹里少了「{trace.query}」"
        assert "命中 9 条 · 保留 0 条" in markdown
        assert "命中 12 条 · 保留 1 条" in markdown

    def test_html_lists_every_query_with_hits_and_kept(self):
        card = make_card(research_queries=self.TRACES)
        document = render_html(make_result(cards=[card]))

        assert "检索轨迹" in document
        for trace in self.TRACES:
            assert trace.query in document
        assert "命中 9 条 · 保留 0 条" in document

    def test_failed_query_is_shown_not_hidden(self):
        """★ 失败的查询必须留下 —— 它正是"这次没查成"的证据。

        把它藏起来，结论就说得比实际更确定了。
        """
        card = make_card(
            research_status="failed",
            research_queries=(QueryTrace(query="护肤", channel="appstore", error="HTTP 429"),),
        )
        markdown = render_markdown(make_result(cards=[card]))
        document = render_html(make_result(cards=[card]))

        assert "未查成" in markdown and "429" in markdown
        assert "未查成" in document and "429" in document

    def test_no_queries_means_no_trace_section(self):
        """手工构造的卡片没有轨迹时，不能凭空印一个小节。"""
        markdown = render_markdown(make_result(cards=[make_card()]))
        assert "检索轨迹" not in markdown

    def test_the_promise_is_only_made_when_there_is_a_trace(self):
        """★ "检索轨迹见下方"是一句**承诺**，没有轨迹时不许说。

        这条守卫的是"报告不说空话" —— M2 修的很大一类问题就是声明与事实不符
        （结论文案声称有轨迹、而产物里根本没有）。
        """
        with_trace = make_card(research_queries=self.TRACES)
        without = make_card()

        assert "见下方" in render_html(make_result(cards=[with_trace]))
        assert "见下方" not in render_html(make_result(cards=[without]))


class TestUnjudgedCompetitorsAreLabelled:
    """相关性判定失败时，卡片必须说清"这些竞品没验过"。

    判定失败时全部候选被**保留**（保守取舍，见 ``research/relevance.py``），于是
    findings 非空 ⇒ 状态是 ``ok``。缺了这个标记，一次 LLM 抖动在卡片上与一次
    正常判定**长得一模一样** —— 用户会把"候选全量保留"当成"这个方向真的有这些
    竞品"，而那正是 M2 验收门要抓的误报。
    """

    UNRELATED = [
        CompetitorFinding(source="github", name="someone/books", url="https://example.test/books")
    ]

    def test_markdown_warns_the_competitors_are_unjudged(self):
        card = make_card(
            competitors=self.UNRELATED,
            research_status="ok",
            research_judgement_failed=True,
        )
        assert "未经相关性判定" in render_markdown(make_result(cards=[card]))

    def test_html_warns_the_competitors_are_unjudged(self):
        card = make_card(
            competitors=self.UNRELATED,
            research_status="ok",
            research_judgement_failed=True,
        )
        assert "未经相关性判定" in render_html(make_result(cards=[card]))

    def test_normal_judgement_says_nothing_extra(self):
        """反向守卫：判定正常时不加这句话 —— 否则提示会变成人人忽略的噪音。"""
        card = make_card(competitors=self.UNRELATED, research_status="ok")
        assert "未经相关性判定" not in render_markdown(make_result(cards=[card]))
        assert "未经相关性判定" not in render_html(make_result(cards=[card]))


class TestCompetitorDescriptions:
    """竞品描述必须出现在**本地产物**里。

    它是"这条为什么算竞品"的唯一依据，也是 M2 验收门「人工抽检准确率」的输入 ——
    此前它只进判定提示词与出网载荷，本地产物里反而看不到，用户只能逐个点开链接
    自行判断（而抽检要看的正是这个）。
    """

    DESCRIPTION = "拍照查询化妆品成分，覆盖十万种市售产品"

    def _card(self) -> OpportunityCard:
        return make_card(
            competitors=[
                CompetitorFinding(
                    source="appstore",
                    name="美丽修行",
                    url="https://apps.apple.com/cn/app/x",
                    description=self.DESCRIPTION,
                )
            ],
            research_status="ok",
        )

    def test_html_shows_the_description(self):
        assert self.DESCRIPTION in render_html(make_result(cards=[self._card()]))

    def test_markdown_shows_the_description(self):
        assert self.DESCRIPTION in render_markdown(make_result(cards=[self._card()]))

    def test_long_description_is_truncated(self):
        """超长描述必须截断 —— App Store 的副标题能到上千字，几张卡片就淹没报告。

        截断长度与判定用的保持一致（160 字）：**判定看多少字，人就该看到多少字**，
        否则用户复核时会发现"报告里的描述不足以判出这个结论"。
        """
        long_text = "描" * 900
        card = make_card(
            competitors=[
                CompetitorFinding(
                    source="appstore", name="x", url="https://e.test/x", description=long_text
                )
            ],
            research_status="ok",
        )
        document = render_html(make_result(cards=[card]))
        assert long_text not in document
        assert "描" * 159 + "…" in document

    def test_markdown_long_description_is_truncated(self):
        """HTML 截断了 Markdown 也必须截断 —— 同一个字段在两种产物上要给同样的口径，
        否则"贴进 issue 的那一份"和"发给别人的那一份"会不一样长。
        """
        long_text = "描" * 900
        card = make_card(
            competitors=[
                CompetitorFinding(
                    source="appstore", name="x", url="https://e.test/x", description=long_text
                )
            ],
            research_status="ok",
        )
        document = render_markdown(make_result(cards=[card]))
        assert long_text not in document
        assert "描" * 159 + "…" in document

    def test_truncation_length_matches_the_judgement(self):
        """两个渲染器的截断长度都必须**等于判定用的常量**。

        这条钉的是口径本身而不是字面量 160：判定看多少字、人就该看到多少字。改了
        判定那边却忘了改渲染，用户复核时会发现"报告里的描述不足以判出这个结论"——
        而那是这个字段存在的全部意义。
        """
        from xhs_pain_miner.render import html as html_render
        from xhs_pain_miner.render import markdown as markdown_render
        from xhs_pain_miner.research.relevance import _MAX_DESCRIPTION_CHARS

        assert html_render._COMPETITOR_DESC_CHARS == _MAX_DESCRIPTION_CHARS
        assert markdown_render._COMPETITOR_DESC_CHARS == _MAX_DESCRIPTION_CHARS


class TestCompetitorDescriptionMissing:
    """缺描述时**少显示一行**，不出现任何占位文案、也不留空段落。

    占位（"（无描述）"）比不显示更糟：它看起来像一条结论，读者会以为"平台给了、
    但内容是空的"，而实际情况是"这个渠道根本没提供这个字段"—— 两件事的含义不同，
    报告不该把它们说成同一件。空段落同理：一个空的 ``comp-gap`` / 空的引用块行，
    在版面上就是"这里本来该有条结论"。缺什么少什么，是渲染层对外的承诺。

    "缺"有八种写法，**每一条都要撞**：空串、``None``、只有空白的串、只有换行的串、
    非字符串（列表 / 字典），以及**只由不可见字符组成**的串（单个零宽空格、以及
    一串混在一起的零宽字符 / ZWJ / BOM）。上游 ``_describe`` 会把前几族都压成空串，
    但那是对接方的行为、不是渲染层可以依赖的保证 —— 只在空串上测，等于没测
    ``None``（崩溃点）、纯空白（空段落）与不可见字符（看不见的空段落）。
    """

    PLACEHOLDERS = ("（无描述）", "无描述", "暂无描述", "（平台未提供）", "未提供")
    MISSING: tuple[object, ...] = (
        "",
        None,
        "   ",
        "\t\n",
        ["a"],
        {"k": "v"},
        "\u200b",  # 零宽空格 —— str.strip() 拦不住的那一族
        "\u200b\u200d\ufeff",  # 一串不可见字符
    )

    def _card(self, description: object = "") -> OpportunityCard:
        finding = CompetitorFinding(
            source="appstore", name="美丽修行", url="https://apps.apple.com/cn/app/x"
        )
        finding.description = cast("str", description)
        return make_card(competitors=[finding], research_status="ok")

    def _assert_every_family_is_still_covered(self) -> None:
        """守卫这一组测试的**输入集合本身**。

        下面几条测试都是"对 ``MISSING`` 逐项撞"，所以 ``MISSING`` 一旦被削短或换掉，
        它们不会红 —— 它们只是**静默变弱**，什么都不再证明。光钉数量不够：
        八项全换成 ``1..8`` 长度照样达标，而 ``None`` / 纯空白 / 非字符串 / 不可见
        字符四族会一起失去覆盖。所以钉的是"这四族都还在"。
        """
        assert any(m is None for m in self.MISSING), "MISSING 里没有 None —— 崩溃点失去覆盖"
        assert sum(1 for m in self.MISSING if isinstance(m, str) and not m.strip()) >= 2, (
            "MISSING 里的空白串不足两条 —— 空段落失去覆盖"
        )
        assert any(not isinstance(m, str) and m is not None for m in self.MISSING), (
            "MISSING 里没有非字符串 —— repr 泄漏失去覆盖"
        )
        assert any(isinstance(m, str) and m.strip() for m in self.MISSING), (
            "MISSING 里没有看不见但非空白的串 —— 不可见字符族失去覆盖"
        )

    def test_html_has_no_placeholder(self):
        self._assert_every_family_is_still_covered()
        for missing in self.MISSING:
            document = render_html(make_result(cards=[self._card(description=missing)]))
            for placeholder in self.PLACEHOLDERS:
                assert placeholder not in document, (
                    f"description={missing!r} 时 HTML 里出现了占位文案：{placeholder}"
                )

    def test_markdown_has_no_placeholder(self):
        self._assert_every_family_is_still_covered()
        for missing in self.MISSING:
            markdown = render_markdown(make_result(cards=[self._card(description=missing)]))
            for placeholder in self.PLACEHOLDERS:
                assert placeholder not in markdown, (
                    f"description={missing!r} 时 Markdown 里出现了占位文案：{placeholder}"
                )

    def test_html_shows_one_line_fewer(self):
        """少的是**那一行本身**，不是留一个空段落占位。"""
        self._assert_every_family_is_still_covered()
        with_desc = render_html(make_result(cards=[self._card(description="有描述")]))
        for missing in self.MISSING:
            without = render_html(make_result(cards=[self._card(description=missing)]))
            assert with_desc.count('class="comp-gap"') == without.count('class="comp-gap"') + 1, (
                f"description={missing!r} 时少的不止一行（或多了个空段落）"
            )
            assert '<p class="comp-gap"></p>' not in without

    def test_markdown_shows_one_line_fewer(self):
        self._assert_every_family_is_still_covered()
        with_desc = render_markdown(make_result(cards=[self._card(description="有描述")]))
        for missing in self.MISSING:
            without = render_markdown(make_result(cards=[self._card(description=missing)]))
            assert with_desc.count("\n  > ") == without.count("\n  > ") + 1, (
                f"description={missing!r} 时少的不止一行（或多了个空引用行）"
            )

    def test_every_missing_spelling_renders_like_the_empty_one(self):
        """空串 / ``None`` / 纯空白 / 纯换行 / 非字符串，五种都必须产出**逐字节相同**的产物。

        只断言"不崩"太松：崩溃之外，"渲染出一个空段落""印了占位""把 stars 或链接
        弄丢了""顺序变了"都能从它下面溜过去。逐字节相等一次排除全部 —— 而且它把
        HTML 与 Markdown 钉成同一行为，不会再出现"一个崩一个不崩"。
        """
        # 循环体跑的是 ``MISSING[1:]``：``MISSING`` 若被削到只剩一项，下面两条断言
        # 一次都不执行、这条测试会**静默全绿**。先钉住输入集合本身。
        self._assert_every_family_is_still_covered()

        expected_html = render_html(make_result(cards=[self._card()]))
        expected_markdown = render_markdown(make_result(cards=[self._card()]))
        for missing in self.MISSING[1:]:
            result = make_result(cards=[self._card(description=missing)])
            assert render_html(result) == expected_html, f"description={missing!r} 的 HTML 产物不同"
            assert render_markdown(result) == expected_markdown, (
                f"description={missing!r} 的 Markdown 产物不同"
            )

    def test_none_description_does_not_crash_html(self):
        """``None`` 也必须不崩。

        ``description`` 声明成 ``str``，但那是**调用方的类型约定，不是运行时保证**：
        渲染层对外的承诺是"缺什么少显示什么，而不是抛异常让用户拿不到报告"（见
        ``render_html`` 的 Note）。这条把 HTML 与 Markdown 的行为钉成一致 —— 修之前
        同一个字段在 HTML 上抛 ``TypeError``、在 Markdown 上安然渲染。
        """
        document = render_html(make_result(cards=[self._card(description=None)]))
        assert "美丽修行" in document

    def test_none_description_does_not_crash_markdown(self):
        markdown = render_markdown(make_result(cards=[self._card(description=None)]))
        assert "美丽修行" in markdown
