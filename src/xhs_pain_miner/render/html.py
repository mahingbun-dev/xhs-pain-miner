"""单文件 HTML 渲染 —— 本产品的核心交付物。

产物是**一个自包含的 HTML 文件**：内联 CSS、无外部资源、无 JavaScript 依赖，
双击即可打开、可以直接发给别人。不用 Next.js 或任何前端框架，因为用户的真实
使用场景是"跑完看一眼"，不是"部署一个站点"。

安全（本模块最重要的一条）
--------------------------
卡片里的每一个字段都可能来自平台采集内容，也就是**完全不可信的输入**。
``pain.label`` 虽然是 LLM 生成的，但 LLM 的输入同样来自采集内容 ——
prompt injection 可以让它吐出任意字符串。因此：

    **任何插入 HTML 的值都必须经过 ``html.escape()``，没有例外。**

一次漏转义就是一次存储型 XSS：用户在浏览器里打开报告，脚本就能读取他本机
的其他内容。与 M0 里 rich markup 注入是同一类问题（见 ``cli._safe``），
只是危害更大 —— 终端里最多渲染出个超链接，浏览器里能执行任意代码。

本模块的转义口径（三件事，缺一不可）：

1. **文本节点与属性值**一律走 :func:`_esc`（``html.escape(quote=True)``）。
2. **数字**在转义前先格式化（``f"{value:.0f}"``），保证插进 ``style`` 的只有
   数字字符 —— 卡片里确实有一个按分数计算宽度的进度条。
3. **链接**额外做协议白名单（:func:`_link`）：``javascript:`` 是一个**合法**的
   URL，转义拦不住它。竞品信息来自第三方 API，不能假设它一定安分。

呈现上的两个取舍
----------------
* 竞品调研的**四种结论类别**必须显示成四句不同的话（见 :func:`_competitor_verdict`）：
  "查到竞品" / "查证过，确实没有" / "检索不到，无法判断" / "调研失败"。判据是卡片上
  的 :attr:`~xhs_pain_miner.models.OpportunityCard.research_status`。
* 权重不进报告表格，除非能从卡片还原出默认权重（见 :func:`_weights_view`）。
  卡片只保存因子得分与总分，**不保存权重**；编一个"权重"填进表格比不显示更糟。
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from xhs_pain_miner.models import Evidence, MiningResult, OpportunityCard
from xhs_pain_miner.research.outcome import STATUS_LABELS
from xhs_pain_miner.scoring.opportunity import FACTOR_LABELS, FACTOR_NAMES, NEUTRAL, ScoreWeights

DEFAULT_TITLE = "机会卡片"

MAX_EVIDENCE_PER_CARD = 5
"""每张卡片最多内嵌多少条证据。

全量内嵌会让 200 篇笔记的报告膨胀到几 MB —— 用户要的是结论，不是语料库。
超出部分在卡片上标注"另有 N 条"，完整证据在本地 SQLite 里可查。
"""

_HTML_LANG = "zh-CN"

_ALLOWED_URL_PREFIXES = ("http://", "https://")
"""链接协议白名单。

``javascript:`` 是合法 URL，``html.escape`` 不会也不能拦它 —— 只有协议白名单
能拦住。非 http(s) 的竞品链接降级成纯文本，用户仍然看得到名字。
"""

_SOURCE_LABELS = {"note": "笔记", "comment": "评论"}

_STAGE_LABELS = {
    "new": "新兴",
    "growing": "上升",
    "stable": "平稳",
    "declining": "下降",
}

_BAND_LABELS = {"high": "高分机会", "mid": "值得关注", "low": "建议观望"}

_SCORE_HIGH = 70.0
_SCORE_MID = 45.0
"""机会分的分档阈值（展示用，不参与计算）。"""


def _esc(value: object) -> str:
    """转义任意值，供插入文本节点或属性值。

    本模块**唯一**允许把动态值变成 HTML 的入口。任何绕过它的插值都是一次
    潜在的存储型 XSS。
    """
    return html.escape(str(value), quote=True)


def _num(value: float, digits: int = 0) -> str:
    """把数字格式化成只含数字与小数点的字符串（转义后供 style/文本使用）。"""
    return _esc(f"{value:.{digits}f}")


def _link(url: str, label: str) -> str:
    """把竞品 URL 渲染成链接，非 http(s) 的一律降级为纯文本。"""
    target = (url or "").strip()
    safe_scheme = target.lower().startswith(_ALLOWED_URL_PREFIXES)
    # 控制字符与空白：浏览器会先剔除再解析，"java\\nscript:" 这类绕过正是靠它生效
    clean = bool(target) and not any(ch.isspace() or ord(ch) < 32 for ch in target)
    if safe_scheme and clean:
        return (
            f'<a href="{_esc(target)}" target="_blank" rel="noopener noreferrer nofollow">'
            f"{_esc(label)}</a>"
        )
    return f"<span>{_esc(label)}</span>"


def _date_text(value: date | datetime | None) -> str:
    """格式化日期；缺失时返回空串（缺什么少显示什么，不显示占位符）。"""
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d")


def _ago_text(value: date | datetime | None) -> str:
    """相对时间描述。

    只用 ``date`` 相减：``datetime`` 可能带时区也可能不带，直接和 ``date.today()``
    做减法会在混用时抛 ``TypeError`` —— 而渲染层不允许因为一个时间字段崩掉。
    """
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


def _score_band(score: float) -> str:
    """机会分的展示分档。"""
    if score >= _SCORE_HIGH:
        return "high"
    if score >= _SCORE_MID:
        return "mid"
    return "low"


def _weights_view(card: OpportunityCard) -> dict[str, float] | None:
    """尝试从卡片还原本次运行使用的因子权重。

    卡片只保存 ``score`` 与 ``score_breakdown``，不保存权重。默认权重是唯一能
    被还原的情形：用它重算的总分与卡片上的分数一致（容差覆盖 round(1) 的舍入）。
    还原不出来就返回 ``None``，由渲染层改用不带权重的表格 —— 编一个权重填进
    溯源表格，比不显示更糟。
    """
    default = ScoreWeights().normalized().to_dict()
    expected = 100.0 * sum(
        default[name] * card.score_breakdown.get(name, 0.0) for name in FACTOR_NAMES
    )
    return default if abs(expected - card.score) <= 0.15 else None


# --------------------------------------------------------------------------- #
# 样式（静态字符串，不含任何插值）
# --------------------------------------------------------------------------- #

_CSS = """
:root{
  --bg:#f5f6f8;--panel:#fff;--ink:#15181e;--muted:#69707c;--faint:#98a0ac;
  --line:#e6e8ec;--line-strong:#d2d6dd;--brand:#c9385a;--brand-soft:#fdeef1;
  --high:#0f7a57;--high-soft:#e8f6f0;--mid:#a9701a;--mid-soft:#fdf3e2;
  --low:#69707c;--low-soft:#f0f1f4;--radius:14px;
  --shadow:0 1px 2px rgba(18,20,26,.05),0 10px 28px rgba(18,20,26,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#101216;--panel:#181b21;--ink:#e8eaee;--muted:#9aa2ae;--faint:#7b8391;
    --line:#262a33;--line-strong:#333846;--brand:#ef6b88;--brand-soft:#2a1a20;
    --high:#46c295;--high-soft:#152620;--mid:#dfae57;--mid-soft:#272013;
    --low:#9aa2ae;--low-soft:#20242c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.35);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.62;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
  "Hiragino Sans GB","Microsoft YaHei",sans-serif}
a{color:var(--brand);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 56px}
.hero{color:#fff;padding:36px 0 30px;margin-bottom:24px;
  background:linear-gradient(135deg,#1b1e26,#2b3040)}
.hero .wrap{padding-bottom:0}
.eyebrow{margin:0 0 8px;font-size:11.5px;letter-spacing:.18em;text-transform:uppercase;
  color:#98a2b6}
.hero h1{margin:0 0 10px;font-size:29px;font-weight:650;line-height:1.28;
  letter-spacing:-.01em}
.hero-meta{margin:0;font-size:13px;color:#b6bdcb}
.stats{display:grid;gap:12px;margin-bottom:22px;
  grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:13px 16px;box-shadow:var(--shadow)}
.stat .k{font-size:11.5px;color:var(--muted);letter-spacing:.05em}
.stat .v{margin-top:3px;font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.stat .s{margin-top:2px;font-size:11.5px;color:var(--faint)}
.notes{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--mid);
  border-radius:10px;padding:13px 18px;margin-bottom:22px}
.notes h2{margin:0 0 6px;font-size:13.5px}
.notes ul{margin:0;padding-left:18px;font-size:13px;color:var(--muted)}
.notes li+li{margin-top:4px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:20px 24px 18px;margin-bottom:18px}
.card-top{display:flex;gap:18px;align-items:flex-start}
.score{flex:0 0 auto;width:84px;height:84px;border-radius:50%;display:flex;
  flex-direction:column;align-items:center;justify-content:center;
  border:3px solid var(--line-strong);background:var(--low-soft)}
.score .score-num{font-size:25px;font-weight:700;line-height:1;
  font-variant-numeric:tabular-nums}
.score .score-cap{font-size:10.5px;letter-spacing:.06em;color:var(--muted);margin-top:2px}
.score.high{border-color:var(--high);color:var(--high);background:var(--high-soft)}
.score.mid{border-color:var(--mid);color:var(--mid);background:var(--mid-soft)}
.score.low{border-color:var(--line-strong);color:var(--low);background:var(--low-soft)}
.card-head{flex:1 1 auto;min-width:0}
.card-head h2{margin:1px 0 6px;font-size:19.5px;line-height:1.36;font-weight:650;
  word-break:break-word}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;background:var(--bg);
  border:1px solid var(--line);font-size:11.5px;color:var(--muted)}
.chip.band-high{color:var(--high);border-color:var(--high);background:var(--high-soft);
  font-weight:600}
.chip.band-mid{color:var(--mid);border-color:var(--mid);background:var(--mid-soft);
  font-weight:600}
.pain-line{margin:0;font-size:13.5px;color:var(--muted);word-break:break-word}
.summary{margin:14px 0 0;font-size:14px;color:var(--ink);background:var(--bg);
  border-left:2px solid var(--line-strong);border-radius:0 8px 8px 0;
  padding:9px 13px;word-break:break-word}
.factors{margin-top:16px;border-top:1px dashed var(--line);padding-top:13px}
.factor{display:grid;gap:10px;align-items:center;font-size:12.5px;margin-bottom:7px;
  grid-template-columns:104px 1fr 52px 104px}
.factor .f-name{color:var(--muted)}
.factor .f-bar{height:7px;background:var(--line);border-radius:99px;overflow:hidden}
.factor .f-bar>i{display:block;height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--brand),#ef8a63)}
.factor .f-val{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.factor .f-w{text-align:right;color:var(--faint);font-variant-numeric:tabular-nums}
.cols{display:grid;gap:22px;margin-top:18px;grid-template-columns:1.1fr .9fr}
.col-title{margin:0 0 9px;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted)}
.ev{list-style:none;margin:0;padding:0}
.ev li{background:var(--bg);border:1px solid var(--line);border-radius:10px;
  padding:10px 12px;margin-bottom:8px}
.ev .ev-text{margin:0;font-size:13.5px;line-height:1.6;word-break:break-word}
.ev .ev-meta{margin:7px 0 0;display:flex;flex-wrap:wrap;gap:9px;align-items:center;
  font-size:11.5px;color:var(--faint)}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--line);
  color:var(--muted);font-size:11px}
.more{margin:6px 0 0;font-size:11.5px;color:var(--faint)}
.comps{list-style:none;margin:0;padding:0}
.comps li{border:1px solid var(--line);border-radius:10px;padding:10px 12px;
  margin-bottom:8px}
.comp-name{font-size:13.5px;font-weight:600;word-break:break-all}
.comp-name .comp-src{font-weight:400;color:var(--faint);font-size:11.5px;margin-left:6px}
.comp-meta{margin:5px 0 0;display:flex;flex-wrap:wrap;gap:9px;font-size:11.5px;
  color:var(--faint)}
.comp-gap{margin:6px 0 0;font-size:12.5px;color:var(--muted);word-break:break-word}
.stale{color:var(--high);font-weight:600}
.verdict{margin:10px 0 0;font-size:12.5px;line-height:1.55;color:var(--muted)}
.traces{margin-top:8px;font-size:12px;color:var(--muted)}
.traces summary{cursor:pointer;color:var(--brand);font-weight:600}
.traces ul{margin:7px 0 0;padding-left:18px}
.traces li{margin:3px 0;word-break:break-word}
.traces code{background:var(--bg);border-radius:4px;padding:1px 5px;font-size:11.5px}
.traces li span{margin-left:6px}
.traces .failed{color:var(--high)}
.empty{margin:0;font-size:12.5px;color:var(--muted);background:var(--bg);
  border:1px dashed var(--line-strong);border-radius:10px;padding:11px 13px}
.caliber{margin-top:16px;border-top:1px dashed var(--line);padding-top:10px;
  font-size:12.5px;color:var(--muted)}
.caliber summary{cursor:pointer;color:var(--brand);font-weight:600}
.caliber p{margin:8px 0}
.caliber table{width:100%;border-collapse:collapse;margin-top:6px;
  font-variant-numeric:tabular-nums}
.caliber th,.caliber td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
.caliber th{color:var(--faint);font-weight:500}
.caliber td.n,.caliber th.n{text-align:right}
.foot{margin-top:20px;border-top:1px solid var(--line);padding-top:14px;
  font-size:11.5px;line-height:1.7;color:var(--faint)}
.foot p{margin:0 0 6px}
@media (max-width:820px){.cols{grid-template-columns:1fr;gap:18px}}
@media (max-width:640px){
  .hero h1{font-size:23px}
  .card{padding:16px 16px 14px}
  .card-top{gap:13px}
  .score{width:68px;height:68px}
  .score .score-num{font-size:21px}
  .factor{grid-template-columns:86px 1fr 46px}
  .factor .f-w{display:none}
}
@media print{
  body{background:#fff}
  .card,.stat{break-inside:avoid;box-shadow:none}
  .caliber{display:none}
}
""".strip()


# --------------------------------------------------------------------------- #
# 片段渲染
# --------------------------------------------------------------------------- #


def _select_cards(result: MiningResult, *, include_noise: bool) -> list[OpportunityCard]:
    """挑出要展示的卡片，按机会分降序。

    噪声簇（HDBSCAN 的 ``-1``）默认不进主报告：它们量大且零散，会淹没高分机会。
    但它们**不该被删除**，``include_noise=True`` 时应能取到。
    """
    cards = [card for card in result.top_cards if include_noise or not card.pain.is_noise]
    return sorted(cards, key=lambda card: (-card.score, -card.pain.size, card.id))


def _render_stats(result: MiningResult, cards: Sequence[OpportunityCard]) -> str:
    """顶部概览数字。"""
    scores = sorted((card.score for card in cards), reverse=True)
    top = f"{scores[0]:.0f}" if scores else "—"
    median = f"{scores[len(scores) // 2]:.0f}" if scores else "—"
    noise = sum(1 for cluster in result.clusters if cluster.is_noise)
    active = sum(1 for card in cards if card.has_active_competitor)
    tiles = [
        ("机会卡片", str(len(cards)), "不含噪声簇" if noise else "按机会分降序"),
        ("最高 / 中位机会分", f"{top} / {median}", "满分 100，因子可展开溯源"),
        ("痛点簇", str(len(result.clusters)), f"其中噪声簇 {noise} 个"),
        ("活跃竞品", str(active), "有活跃竞品的卡片数"),
        ("样本", str(result.total_notes), f"篇笔记 / {result.total_comments} 条评论"),
    ]
    cells = "".join(
        f'<div class="stat"><div class="k">{_esc(name)}</div>'
        f'<div class="v">{_esc(value)}</div><div class="s">{_esc(sub)}</div></div>'
        for name, value, sub in tiles
    )
    return f'<section class="stats">{cells}</section>'


def _render_notes(notes: Sequence[str]) -> str:
    """降级警告 —— 静默降级会让用户以为看到的是完整结果。"""
    items = [note for note in notes if note and note.strip()]
    if not items:
        return ""
    lis = "".join(f"<li>{_esc(note)}</li>" for note in items)
    return f'<section class="notes"><h2>运行提示（降级与口径）</h2><ul>{lis}</ul></section>'


def _render_evidence(card: OpportunityCard, *, max_evidence: int) -> str:
    """证据链 —— 每条结论都能点回原文，这是本产品的核心防守点。"""
    evidences: Sequence[Evidence] = card.pain.evidences
    shown = list(evidences[: max(max_evidence, 0)])
    if not shown:
        # 没内嵌 ≠ 没有证据：查得到条数就要如实报出来，否则用户会以为这个痛点
        # 一条证据都没有（那是对结论可信度的否定）
        body = (
            '<p class="empty">该簇没有证据记录（可能是图片派生簇或证据已裁剪）</p>'
            if not evidences
            else f'<p class="empty">本次未内嵌证据（上限 {_esc(max_evidence)} 条）—— '
            f"该簇共 {_esc(len(evidences))} 条，完整记录在本地数据库中可查</p>"
        )
        return f'<section><h3 class="col-title">证据链</h3>{body}</section>'

    items: list[str] = []
    for evidence in shown:
        source = _SOURCE_LABELS.get(evidence.source, str(evidence.source))
        meta = [f'<span class="badge">{_esc(source)}</span>']
        if evidence.likes:
            meta.append(f"<span>♥ {_esc(evidence.likes)}</span>")
        stamp: Any = getattr(evidence, "created_at", None)
        if isinstance(stamp, (datetime, date)):
            text = _date_text(stamp)
            ago = _ago_text(stamp)
            meta.append(f"<span>{_esc(text)}{f'（{_esc(ago)}）' if ago else ''}</span>")
        items.append(
            '<li><p class="ev-text">{text}</p><p class="ev-meta">{meta}</p></li>'.format(
                text=_esc(evidence.text), meta="".join(meta)
            )
        )

    rest = len(evidences) - len(shown)
    more = (
        f'<p class="more">另有 {_esc(rest)} 条证据未内嵌（上限 {_esc(max_evidence)} 条），'
        "完整记录在本地数据库中可查</p>"
        if rest > 0
        else ""
    )
    return (
        '<section><h3 class="col-title">证据链</h3>'
        f'<ul class="ev">{"".join(items)}</ul>{more}</section>'
    )


def _competitor_verdict(card: OpportunityCard) -> str:
    """竞品调研的一句话结论。

    **四种状态必须说成四句不同的话**，判据只能是卡片上的
    :attr:`~xhs_pain_miner.models.OpportunityCard.research_status`：

    * ``ok`` —— 查到竞品（卡片上会列出它们）
    * ``no_competitor`` —— 平台搜得到内容，只是没有与这个痛点相关的实现
    * ``unsearchable`` —— 检索不到，无法判断
    * ``failed`` —— 这次没查成

    后两种对**用户该做什么**的指示完全不同：``unsearchable`` 可以换个更贴近
    "用户会去找什么工具"的说法再搜一次，``failed`` 只能等额度或网络恢复。
    把它们说成同一句话，用户就无从决定下一步。

    M1 靠 ``competitor_gap == 0.5`` 反推"没查成"，那条推理有一个精确碰撞：
    :data:`~xhs_pain_miner.scoring.opportunity.ACTIVE_COMPETITOR_COUNT_SCORE` 的下调
    系数在"2 个活跃竞品、stars 全为 0"时恰好得到 ``0.60 × (1 - 0.5 × 0) = 0.5``
    —— 与中性值逐位相同（扫描 stars 0…200000 × 竞品数 1…3，只有这一组命中）。
    于是报告会在"有 2 个竞品、但都没什么热度"的卡片上多印一句"（本次调研未完成，
    结果可能不完整）" —— 一句没有依据的话。读结论类别之后这个碰撞自然消失：
    有竞品 ⇒ ``ok``，与空白度是多少无关。
    """
    status = card.research_status
    findings = card.competitors
    # 先按"有没有竞品"分岔，再在**没有竞品**的那一支里按结论类别说不同的四句话。
    # 顺序不能反：手工构造的卡片可能出现"写着 no_competitor 却列出 7 个竞品"这种
    # 不自洽的组合，那时展示真实存在的竞品，比照着状态字段断言"没有竞品"诚实。
    if findings:
        active = [finding for finding in findings if not finding.is_stale]
        if not active:
            return (
                f'<p class="verdict">✅ {_esc(len(findings))} 个竞品均已停更 —— '
                "有人验证过需求，但市场现在是空的。进场前请确认它为什么停下。</p>"
            )
        hottest = max((finding.stars or 0) for finding in active)
        # 有竞品却仍要提示"结果可能不完整"的情形有两种，**都必须说**：
        #   * ``research_failed`` —— 还有渠道没查成，清单可能不全；
        #   * ``research_judgement_failed`` —— 这些竞品**根本没验过**相关性
        #     （判定失败时候选被全部保留，保守取舍）。后者此前到不了卡片：
        #     有 findings ⇒ 状态必是 ``ok`` ⇒ ``research_failed`` 恒为 False，
        #     于是"未经判定"只留在运行提示里，而卡片和一次正常判定长得一模一样。
        partial = (
            "（本次调研未完成，结果可能不完整）"
            if card.research_failed or card.research_judgement_failed
            else ""
        )
        if card.research_judgement_failed:
            partial = "（**这些竞品未经相关性判定**，是候选全量保留的结果，请点开自行判断）"
        return (
            f'<p class="verdict">🔧 {_esc(len(active))} 个竞品仍在活跃维护'
            f"（共查到 {_esc(len(findings))} 个，最热 {_esc(hottest)}★）—— "
            f"已有玩家，需要找差异化切口{_esc(partial)}</p>"
        )

    if status in ("unsearchable", "failed"):
        label = _esc(STATUS_LABELS[status])
        detail = (
            # "检索不到"只是这个状态最常见的成因（M2 修的正是它），不是全部：
            # 检索词生成失败、调研被关闭时也会落到这里，那时说"检索不到"就是一句
            # 失实的话（我们根本没有发出去过任何检索词）。所以两种成因都写出来。
            "这不代表该方向没有竞品：可能是这些检索词在平台上没有返回任何东西"
            "（换个更贴近「用户会去找什么工具」的说法再搜，往往就能搜到），"
            "也可能是这次没有可用的检索词。"
            if status == "unsearchable"
            else "这不代表该方向没有竞品，只是这次没查成。"
        )
        return (
            f'<p class="verdict">⚠️ {label} —— 「竞品空白度」按中性值 '
            f"{_num(NEUTRAL, 1)} 计，{detail}</p>"
        )
    if status == "no_competitor":
        # 有轨迹才敢说"见下方" —— 手工构造的卡片可能没有，那时这句话就是空头承诺
        hint = "（检索轨迹见下方，可逐条复核）" if card.research_queries else ""
        return (
            f'<p class="verdict">✅ {_esc(STATUS_LABELS[status])} —— '
            f"平台能搜到内容，但没有与这个痛点相关的实现{hint}。</p>"
        )
    # ``ok`` 却没有竞品 —— 只有手工构造的卡片会这样（classify_status 产不出它）。
    # 此时说"查证过确实没有"是没有依据的断言，如实说没有记录即可。
    return '<p class="verdict">本次没有可展示的竞品记录。</p>'


def _render_traces(card: OpportunityCard) -> str:
    """检索轨迹 —— 「结论可逐条复核」这个卖点的落地。

    它回答的是"这个结论是怎么得出来的"：实际搜了什么词、发给了哪个平台、平台回了
    几条、最后留了几条。**没有它，一个「查证过，没有相关竞品」无法被质疑** ——
    而"能被质疑"正是本产品对"免费的 LLM 摘要"的正面防守。

    默认**收起**（``<details>``）：多数用户不会逐条复核，但它必须**存在且可点开**，
    否则结论文案里那句"可逐条复核"就是一句空话。

    检索词是 LLM 生成的自由文本。它出现在这里（本地产物）是刻意的，而**不出现在
    ``to_public_dict()`` 里**也是刻意的 —— 见 :attr:`OpportunityCard.research_queries`。
    """
    if not card.research_queries:
        return ""

    items: list[str] = []
    for trace in card.research_queries:
        if trace.succeeded:
            detail = f"命中 {trace.hits} 条 · 保留 {trace.kept} 条"
            cls = ""
        else:
            # 失败的查询必须留下 —— 它正是"这次没查成"的证据，藏起来就等于
            # 把结论说得比实际更确定
            detail = f"未查成：{trace.error}"
            cls = ' class="failed"'
        items.append(
            f"<li{cls}><code>{_esc(trace.channel)}</code> "
            f"「{_esc(trace.query)}」<span>{_esc(detail)}</span></li>"
        )
    return (
        '<details class="traces"><summary>检索轨迹（可逐条复核）</summary>'
        f"<ul>{''.join(items)}</ul></details>"
    )


def _render_competitors(card: OpportunityCard) -> str:
    """竞品清单 —— 带链接与"最后活跃"时间，停更的如实标注。"""
    rows: list[str] = []
    for finding in card.competitors:
        name = _link(finding.url, finding.name)
        meta: list[str] = [
            f"<span>{_esc(finding.stars)}★</span>"
            if finding.stars is not None
            else "<span>stars 未知</span>"
        ]
        if finding.last_active is not None:
            ago = _ago_text(finding.last_active)
            label = f"最后活跃 {_date_text(finding.last_active)}"
            if ago:
                label += f"（{ago}）"
            meta.append(f'<span class="{"stale" if finding.is_stale else ""}">{_esc(label)}</span>')
        else:
            meta.append("<span>最后活跃：未知</span>")
        gap = f'<p class="comp-gap">{_esc(finding.gap_notes)}</p>' if finding.gap_notes else ""
        rows.append(
            f'<li><div class="comp-name">{name}'
            f'<span class="comp-src">{_esc(finding.source)}</span></div>'
            f'<p class="comp-meta">{"".join(meta)}</p>{gap}</li>'
        )

    # 没有调研记录时只留结论那句话 —— 空态框和结论说的是同一件事，重复只会稀释重点
    body = f'<ul class="comps">{"".join(rows)}</ul>' if rows else ""
    return (
        f'<section><h3 class="col-title">竞品调研</h3>{body}'
        f"{_competitor_verdict(card)}{_render_traces(card)}</section>"
    )


def _render_factors(card: OpportunityCard, *, weights: dict[str, float] | None) -> str:
    """因子条 —— 可解释机会分的载体：每个因子都能看到自己的得分。"""
    rows: list[str] = []
    for name in FACTOR_NAMES:
        value = card.score_breakdown.get(name)
        if value is None:
            continue
        ratio = max(0.0, min(1.0, float(value))) * 100.0
        weight_cell = (
            f'<span class="f-w">权重 {_num(weights.get(name, 0.0) * 100.0)}%</span>'
            if weights is not None
            else '<span class="f-w"></span>'
        )
        rows.append(
            '<div class="factor">'
            f'<span class="f-name">{_esc(FACTOR_LABELS.get(name, name))}</span>'
            f'<span class="f-bar"><i style="width:{_num(ratio)}%"></i></span>'
            f'<span class="f-val">{_num(float(value), 2)}</span>'
            f"{weight_cell}</div>"
        )
    return f'<div class="factors">{"".join(rows)}</div>'


def _render_caliber(card: OpportunityCard, *, weights: dict[str, float] | None) -> str:
    """评分口径（可展开）—— 让用户能质疑这个数字，而不只是相信它。"""
    if weights is not None:
        rows = "".join(
            "<tr>"
            f"<td>{_esc(FACTOR_LABELS.get(name, name))}</td>"
            f'<td class="n">{_num(weights[name] * 100.0)}%</td>'
            f'<td class="n">{_num(card.score_breakdown.get(name, 0.0), 2)}</td>'
            f'<td class="n">'
            f"{_num(card.score_breakdown.get(name, 0.0) * weights[name] * 100.0, 1)}"
            "</td>"
            "</tr>"
            for name in FACTOR_NAMES
        )
        table = (
            '<table><tr><th>因子</th><th class="n">权重</th>'
            '<th class="n">得分</th><th class="n">贡献</th></tr>'
            f"{rows}</table>"
        )
        note = "权重为默认口径（与本次运行的分数一致）。"
    else:
        rows = "".join(
            "<tr>"
            f"<td>{_esc(FACTOR_LABELS.get(name, name))}</td>"
            f'<td class="n">{_num(card.score_breakdown.get(name, 0.0), 2)}</td>'
            "</tr>"
            for name in FACTOR_NAMES
        )
        table = f'<table><tr><th>因子</th><th class="n">得分</th></tr>{rows}</table>'
        note = (
            "本次运行使用了自定义权重：卡片只保存因子得分与总分，不保存权重，"
            "因此这里不列权重 —— 需要权重明细请用默认权重重跑。"
        )
    return (
        '<details class="caliber"><summary>评分口径</summary>'
        "<p>机会分 = 100 × Σ(权重 × 因子得分)，因子得分均已归一化到 0-1。"
        "缺数据的因子取中性值 0.5 —— 0 的含义是「确认这个维度很差」，与「不知道」是两回事。</p>"
        f"{table}<p>{_esc(note)}</p></details>"
    )


def _render_card(card: OpportunityCard, *, max_evidence: int) -> str:
    """单张机会卡片。"""
    band = _score_band(card.score)
    weights = _weights_view(card)
    cluster = card.pain

    chips = [f'<span class="chip band-{_esc(band)}">{_esc(_BAND_LABELS[band])}</span>']
    chips.append(f'<span class="chip">提及 {_esc(cluster.size)} 次</span>')
    if cluster.category:
        chips.append(f'<span class="chip">{_esc(cluster.category)}</span>')
    chips.append(
        f'<span class="chip">趋势 {_esc(_STAGE_LABELS.get(cluster.stage, cluster.stage))}</span>'
    )
    chips.append(f'<span class="chip">可行度 {_esc(card.feasibility or "未评估")}</span>')
    chips.append(f'<span class="chip">情感 {_num(cluster.sentiment, 2)}</span>')
    if cluster.is_noise:
        chips.append('<span class="chip">长尾低频痛点</span>')

    pain_line = f"痛点：{_esc(cluster.label or '（未命名）')}"
    summary = f'<p class="summary">{_esc(cluster.summary)}</p>' if cluster.summary else ""
    anchor = _esc(card.id)

    return (
        f'<article class="card" id="{anchor}">'
        '<div class="card-top">'
        f'<div class="score {_esc(band)}"><span class="score-num">{_num(card.score)}</span>'
        '<span class="score-cap">机会分</span></div>'
        '<div class="card-head">'
        f"<h2>{_esc(card.title)}</h2>"
        f'<p class="pain-line">{pain_line}</p>'
        f'<div class="chips">{"".join(chips)}</div>'
        "</div></div>"
        f"{summary}"
        f"{_render_factors(card, weights=weights)}"
        '<div class="cols">'
        f"{_render_evidence(card, max_evidence=max_evidence)}"
        f"{_render_competitors(card)}"
        "</div>"
        f"{_render_caliber(card, weights=weights)}"
        "</article>"
    )


def _default_title(keyword: str) -> str:
    """默认报告标题。"""
    keyword = (keyword or "").strip()
    return f"{keyword} {DEFAULT_TITLE}" if keyword else DEFAULT_TITLE


def _render_hero(title: str, result: MiningResult) -> str:
    """页头。"""
    generated = result.generated_at
    stamp = generated.strftime("%Y-%m-%d %H:%M")
    suffix = " UTC" if generated.tzinfo is not None else ""
    parts = []
    if result.keyword.strip():
        parts.append(f"关键词：{_esc(result.keyword.strip())}")
    parts.append(f"样本：{_esc(result.total_notes)} 篇笔记 / {_esc(result.total_comments)} 条评论")
    parts.append(f"生成于 {_esc(stamp)}{_esc(suffix)}")
    meta = " · ".join(parts)
    return (
        '<header class="hero"><div class="wrap">'
        '<p class="eyebrow">XHS Pain Miner · 机会卡片</p>'
        f"<h1>{_esc(title)}</h1>"
        f'<p class="hero-meta">{meta}</p>'
        "</div></header>"
    )


def _render_footer(result: MiningResult) -> str:
    """页脚：成本、口径承诺、以及"这份文件含原文"的必要提醒。"""
    return (
        '<footer class="foot">'
        f"<p>本次运行成本：{_esc(result.cost.summary())}</p>"
        "<p>机会分可复算：因子得分、权重与公式都在卡片内，改权重后重跑即可对比 —— "
        "这不是一个需要你「相信」的黑箱分数。</p>"
        "<p>本文件完全自包含（内联样式、无外部资源、无 JavaScript），可离线打开。"
        "它内嵌了证据原文，转发前请自行确认合规性；需要可安全分享的版本请使用 Markdown 产物。</p>"
        "</footer>"
    )


def render_html(
    result: MiningResult,
    *,
    title: str | None = None,
    max_evidence: int = MAX_EVIDENCE_PER_CARD,
    include_noise: bool = False,
) -> str:
    """把分析结果渲染成自包含的 HTML 字符串。

    Args:
        result: 分析结果。
        title: 报告标题，默认用「<关键词> 机会卡片」。
        max_evidence: 每张卡片内嵌的证据条数上限。
        include_noise: 是否包含噪声簇（长尾低频痛点）。默认不包含 —— 噪声簇
            量大且零散，放在主报告里会淹没高分机会；但它们不该被删除，
            需要时应能通过本参数取到。

    Returns:
        完整的 HTML 文档字符串（含 ``<!DOCTYPE html>``）。

    Note:
        渲染**不得**因为字段缺失而崩溃 —— 真实语料里 ``publish_time`` 可能为
        ``None``、竞品可能没有 stars、降级的簇没有 summary。缺什么就少显示什么，
        而不是抛异常让用户拿不到报告。
    """
    report_title = (title or "").strip() or _default_title(result.keyword)
    cards = _select_cards(result, include_noise=include_noise)
    body = "\n".join(
        part
        for part in (
            _render_hero(report_title, result),
            '<main class="wrap">',
            _render_stats(result, cards),
            _render_notes(result.notes),
            *(_render_card(card, max_evidence=max_evidence) for card in cards),
            _render_footer(result),
            "</main>",
        )
        if part
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            f'<html lang="{_HTML_LANG}">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_esc(report_title)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def write_html(
    result: MiningResult,
    path: str | Path,
    *,
    title: str | None = None,
    max_evidence: int = MAX_EVIDENCE_PER_CARD,
    include_noise: bool = False,
) -> Path:
    """渲染并写入文件。

    Args:
        result: 分析结果。
        path: 输出路径。父目录不存在时会创建。
        title: 报告标题。
        max_evidence: 每张卡片内嵌的证据条数上限。
        include_noise: 是否包含噪声簇。

    Returns:
        写入的绝对路径。

    Raises:
        OSError: 无法创建目录或写入文件。

    Note:
        写入必须显式指定 ``encoding="utf-8"`` —— 用默认编码在 Windows 上
        会让中文变成乱码，而这是一个中文语料的报告。
    """
    target = Path(path).expanduser()
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    content = render_html(
        result,
        title=title,
        max_evidence=max_evidence,
        include_noise=include_noise,
    )
    target.write_text(content, encoding="utf-8")
    return target.resolve()
