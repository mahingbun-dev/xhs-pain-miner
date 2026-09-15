"""Markdown 渲染 —— 便于贴进 issue、Notion、README 或直接喂给 AI 助手。

相比 HTML 版本，这里**不内嵌证据原文**：Markdown 的主要用途是分享与二次加工，
而原文属于本地内容（见 :meth:`~xhs_pain_miner.models.OpportunityCard.to_public_dict`
的合规边界）。需要带证据的自包含报告时用 :mod:`~xhs_pain_miner.render.html`。

这不是"少写一段代码"，而是一条**刻意的不变量**：

    Markdown 产物只允许出现结论字段（标题 / 痛点名 / 摘要 / 因子得分 / 竞品公开信息），
    ``card.pain.evidences`` 里的 ``text`` 一个字都不能进来。

由 ``tests/test_render.py::TestMarkdownNoRawText`` 守卫 —— 有人"顺手"把证据加进
Markdown 时会立刻变红。

转义（容易被低估的一节）
------------------------
"Markdown 看起来是纯文本"是错的。它有两个能被执行/被破坏的入口：

* ``|`` 会把表格结构撑断，``#`` 会把正文变成标题 —— 标签或摘要里带一个就够了。
* ``<tag>`` 是 Markdown 的**行内 HTML**，GitHub / Notion 都会渲染它。卡片字段
  来自采集内容与 LLM 输出，属于不可信输入，因此这里一并做 HTML 转义。

换行也会被折叠成空格：表格单元格里的一个裸换行同样能把整张表撑坏。
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import date, datetime

from xhs_pain_miner.models import CompetitorFinding, MiningResult, OpportunityCard
from xhs_pain_miner.scoring.opportunity import FACTOR_LABELS, FACTOR_NAMES, NEUTRAL

MAX_CARDS = 20
"""默认最多渲染多少张卡片。"""

_STAGE_LABELS = {"new": "新兴", "growing": "上升", "stable": "平稳", "declining": "下降"}

_ALLOWED_URL_PREFIXES = ("http://", "https://")
"""链接协议白名单 —— ``javascript:`` 是合法 URL，转义拦不住它。"""


def _esc(value: object) -> str:
    """转义任意值，供插入 Markdown 正文。

    顺序很重要：先折叠换行、再 HTML 转义、最后转义 ``|`` 与 ``#``。反过来的话
    ``&`` 会被二次转义成 ``&amp;`` 的变体，用户在报告里看到的就是 ``&amp;``。
    """
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = html.escape(text, quote=False)
    return text.replace("|", "\\|").replace("#", "\\#")


def _link(url: str, label: str) -> str:
    """竞品链接；非 http(s) 的 URL 只显示名字。"""
    target = (url or "").strip()
    safe = target.lower().startswith(_ALLOWED_URL_PREFIXES) and not any(
        ch.isspace() or ord(ch) < 32 for ch in target
    )
    if safe:
        # URL 里的括号会提前闭合 Markdown 链接语法，转义掉
        return f"[{_esc(label)}]({_esc(target).replace('(', '%28').replace(')', '%29')})"
    return _esc(label)


def _date_text(value: date | datetime | None) -> str:
    """日期文本；缺失时返回空串。"""
    return "" if value is None else value.strftime("%Y-%m-%d")


def _ago_text(value: date | datetime | None) -> str:
    """相对时间；缺失时返回空串。只用 ``date`` 相减，避免时区混用抛异常。"""
    if value is None:
        return ""
    day = value.date() if isinstance(value, datetime) else value
    days = (date.today() - day).days
    if days < 0:
        return "时间在未来"
    if days == 0:
        return "今天"
    if days < 30:
        return f"{days} 天前"
    if days < 365:
        return f"{days // 30} 个月前"
    return f"{days // 365} 年前"


def _gap_is_neutral(card: OpportunityCard) -> bool:
    """``competitor_gap`` 是否恰好等于中性值（即调研失败）。

    推理依据见 :mod:`~xhs_pain_miner.render.html` 的同名函数：只有调研失败时
    ``competitor_gap`` 才返回 0.5，因此"没有竞品且空白度为 0.5"就是"没查成"。
    在 Markdown 里这条同样重要 —— 分享出去的文本更不能把"没查成"说成"没有竞品"。
    """
    gap = card.score_breakdown.get("competitor_gap")
    return gap is not None and abs(float(gap) - NEUTRAL) < 1e-9


def _competitor_lines(card: OpportunityCard) -> list[str]:
    """竞品小节。"""
    findings: Sequence[CompetitorFinding] = card.competitors
    if not card.competitors and _gap_is_neutral(card):
        return [
            "⚠️ 竞品调研未完成（空白度按中性值 0.5 计）—— 这不代表该方向没有竞品，只是这次没查成。"
        ]
    if not findings:
        return ["未发现竞品 —— 查证过，目前没有可查到的成熟实现。"]

    lines: list[str] = []
    for finding in findings:
        parts = []
        if finding.stars is not None:
            parts.append(f"{finding.stars}★")
        else:
            parts.append("stars 未知")
        if finding.last_active is not None:
            when = _date_text(finding.last_active)
            ago = _ago_text(finding.last_active)
            parts.append(f"最后活跃 {when}" + (f"（{ago}）" if ago else ""))
            parts.append("**已停更**" if finding.is_stale else "仍在维护")
        else:
            parts.append("最后活跃时间未知")
        gap = f"：{_esc(finding.gap_notes)}" if finding.gap_notes else ""
        lines.append(f"- {_link(finding.url, finding.name)} — {' · '.join(parts)}{gap}")
    return lines


def _card_block(card: OpportunityCard, index: int) -> list[str]:
    """单张卡片的 Markdown。"""
    cluster = card.pain
    title = f"### {index}. {card.score:.0f}/100 · {_esc(card.title)}"
    pain_bits = [f"提及 {_esc(cluster.size)} 次"]
    pain_bits.append(f"情感 {cluster.sentiment:.2f}")
    pain_bits.append(f"趋势 {_esc(_STAGE_LABELS.get(cluster.stage, cluster.stage))}")
    if cluster.category:
        pain_bits.append(f"类别 {_esc(cluster.category)}")
    if cluster.is_noise:
        pain_bits.append("长尾低频痛点")

    lines = [title, ""]
    lines.append(f"- **痛点**：{_esc(cluster.label or '（未命名）')}（{' · '.join(pain_bits)}）")
    if cluster.evidences:
        # 只报条数，不报原文 —— 原文属于本地内容，详见模块 docstring
        lines.append(
            f"- **证据**：{_esc(len(cluster.evidences))} 条（原文见本地 HTML 产物或数据库）"
        )
    if card.feasibility:
        lines.append(f"- **可行度**：{_esc(card.feasibility)}")
    if cluster.summary:
        lines.append(f"- **摘要**：{_esc(cluster.summary)}")

    factors = [
        f"{_esc(FACTOR_LABELS.get(name, name))} {card.score_breakdown[name]:.2f}"
        for name in FACTOR_NAMES
        if name in card.score_breakdown
    ]
    if factors:
        lines.append(f"- **因子得分**：{' · '.join(factors)}")
    lines.append("")
    lines.append("**竞品调研**")
    lines.append("")
    lines.extend(_competitor_lines(card))
    lines.append("")
    return lines


def render_markdown(
    result: MiningResult,
    *,
    max_cards: int = MAX_CARDS,
    include_noise: bool = False,
) -> str:
    """把分析结果渲染成 Markdown。

    Args:
        result: 分析结果。
        max_cards: 最多渲染多少张卡片（按分数降序）。
        include_noise: 是否包含噪声簇。

    Returns:
        Markdown 文本。首行是 ````# <关键词> 机会卡片````，随后是概览与卡片列表。

    Note:
        Markdown 里的表格与标题同样需要转义 —— 标签或摘要里若含有 ``|`` 或
        ``#``，会把表格结构撑坏。这与 HTML 转义是同一类问题，别因为"Markdown
        看起来是纯文本"就跳过。
    """
    keyword = result.keyword.strip()
    heading = f"# {_esc(keyword)} 机会卡片" if keyword else "# 机会卡片"

    cards = [card for card in result.top_cards if include_noise or not card.pain.is_noise]
    shown = cards[: max(max_cards, 0)]

    lines = [
        heading,
        "",
        "> 本文件**不含证据原文**，可直接贴进 issue / Notion / 喂给 AI 助手做二次加工。"
        "需要带证据链的自包含报告请用 HTML 产物。",
        "",
        f"- **样本**：{_esc(result.total_notes)} 篇笔记 / {_esc(result.total_comments)} 条评论",
        f"- **生成时间**：{_esc(result.generated_at.strftime('%Y-%m-%d %H:%M'))}"
        + (" UTC" if result.generated_at.tzinfo is not None else ""),
        f"- **机会卡片**：{_esc(len(shown))} 张"
        + (f"（共 {_esc(len(cards))} 张，已按 max_cards 截断）" if len(cards) > len(shown) else ""),
        f"- **成本**：{_esc(result.cost.summary())}",
        "",
    ]

    notes = [note.strip() for note in result.notes if note and note.strip()]
    if notes:
        lines.append("## 运行提示（降级与口径）")
        lines.append("")
        lines.extend(f"- {_esc(note)}" for note in notes)
        lines.append("")

    if not shown:
        lines.extend(["## 机会卡片", "", "没有可展示的卡片（所有簇都被过滤或为空）。", ""])
    else:
        lines.append("## 机会卡片")
        lines.append("")
        for index, card in enumerate(shown, start=1):
            lines.extend(_card_block(card, index))
            lines.append("---")
            lines.append("")

    lines.extend(
        [
            "机会分 = 100 × Σ(权重 × 因子得分)；因子得分均已归一化到 0-1，"
            "缺数据的因子取中性值 0.5（0 的含义是「确认这个维度很差」）。",
            "",
            "由 XHS Pain Miner 生成。",
            "",
        ]
    )
    return "\n".join(lines)
