"""命令行入口。

三个子命令：

* ``mine``    —— 采集并分析，产出机会卡片（分析流水线 M1 实现）。
* ``collect`` —— 只采集并打印统计，用于验证采集层与配置。
* ``doctor``  —— 诊断环境配置。

退出码：``0`` 成功 / ``1`` 运行期错误（含采集失败、后端不可用）/
``2`` 配置缺失、功能未实现或参数错误。

.. warning::
   **所有来自用户输入或采集数据的文本，插入 rich 输出前必须经过** :func:`_safe`。
   rich 默认把方括号当样式标签解析：轻则 `-k "[/]"` 直接抛 ``MarkupError`` 崩溃，
   重则采集到的内容可注入 ``[link=...]`` 渲染出可点击链接。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from xhs_pain_miner import __version__
from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.diagnostics import has_blocking_issue, run_diagnostics
from xhs_pain_miner.llm.base import LLMError
from xhs_pain_miner.models import MiningResult
from xhs_pain_miner.pain_miner import (
    PainMiner,
    PipelineNotAvailableError,
    normalize_keyword,
)

console = Console()
err_console = Console(stderr=True)

BACKEND_CHOICES = click.Choice(["fixture", "plugin", "mcp"], case_sensitive=False)

EXIT_ERROR = 1
EXIT_CONFIG = 2

_STATUS_STYLE = {"ok": ("✅", "green"), "warn": ("⚠️ ", "yellow"), "fail": ("❌", "red")}


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
"""C0 / C1 控制字符（保留 ``\\t`` 与 ``\\n``）。

只转义 rich markup 是不够的：``rich.escape()`` 只处理 ``[``，不管控制字符。
当输出被重定向到文件、管道或 CI 日志（非 TTY）时，裸 ``ESC`` 会被逐字节写出 ——
包括完整可点击的 OSC-8 超链接、``ESC[2J`` 清屏、光标移动，甚至写剪贴板（OSC-52）。
采集到的内容与用户关键词都是不可信输入，必须一并清洗。
"""


def _safe(value: Any) -> str:
    """净化不可信文本（用户输入 / 采集内容 / 异常信息），供 rich 输出使用。

    做两件事：剥离 C0/C1 控制字符，转义 rich markup。
    """
    return escape(_CONTROL_CHARS.sub("", str(value)))


def _validate_keyword(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """规范化并校验关键词。

    复用 :func:`~xhs_pain_miner.pain_miner.normalize_keyword`，保证 CLI 与
    Python API 的契约一致 —— 否则两层对「什么样的关键词算空」会给出不同答案，
    而空关键词会让样例后端静默回退到它自带的默认值。
    """
    try:
        return normalize_keyword(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from None


def _build_settings(**overrides: object) -> Settings:
    """构造配置，并把 ``--backend`` 之类的 CLI 覆盖项合并进去。"""
    return load_settings(**overrides)


def _fail(message: str, *, code: int = EXIT_ERROR) -> None:
    """打印错误并终止。"""
    err_console.print(f"[red]✖ {_safe(message)}[/red]")
    raise SystemExit(code)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="xhs-pain-miner")
def main() -> None:
    """🔍 XHS Pain Miner — 从用户痛点中发现可做的产品机会

    从小红书笔记与评论中聚类出真实痛点，调研现有竞品，输出带机会分的卡片。

    \b
    快速开始:
        xhs-pain-miner doctor                      # 先诊断环境
        xhs-pain-miner collect -k 防晒霜 --backend fixture
        xhs-pain-miner mine -k 防晒霜 --backend fixture
    """
    # 让 rich 在 Windows 等终端上也能正确输出 emoji 与中文
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (OSError, AttributeError):
            pass


# --------------------------------------------------------------------------- #
# mine
# --------------------------------------------------------------------------- #


@main.command()
@click.option(
    "--keyword",
    "-k",
    required=True,
    callback=_validate_keyword,
    help="品类关键词 (e.g., 防晒霜)",
)
@click.option("--notes", "-n", type=int, default=None, help="采集笔记数量")
@click.option(
    "--deep/--no-deep",
    default=False,
    help="开启 VLM 图片分析（成本与耗时显著上升）",
)
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="报告输出路径")
@click.option("--backend", type=BACKEND_CHOICES, default=None, help="覆盖采集后端")
@click.option("--yes", "-y", is_flag=True, help="跳过成本预估确认")
def mine(
    keyword: str,
    notes: int | None,
    deep: bool,
    output: str | None,
    backend: str | None,
    yes: bool,
) -> None:
    """📊 采集并分析一个品类，产出机会卡片。

    \b
    示例:
        xhs-pain-miner mine -k 防晒霜 -n 200
        xhs-pain-miner mine -k 防晒霜 --deep -o 防晒霜-机会卡片.html
    """
    settings = _build_settings(collector_backend=backend)

    console.print(f"🔍 [bold]正在分析「{_safe(keyword)}」[/bold]")
    console.print(f"   采集后端: {_safe(settings.collector_backend)}")
    console.print(f"   笔记数量: {notes if notes is not None else settings.max_notes}")
    console.print(f"   图片分析: {'✅ 开启 (--deep)' if deep else '❌ 关闭'}")

    miner = PainMiner(settings=settings)
    try:
        result = miner.mine(keyword, notes_count=notes, deep=deep)
    except PipelineNotAvailableError as exc:
        err_console.print(Panel(_safe(exc), title="⏳ 功能开发中", border_style="yellow"))
        raise SystemExit(EXIT_CONFIG) from None
    except CollectorError as exc:
        _fail(f"采集失败：{exc}")
        return
    except LLMError as exc:
        _fail(f"模型调用失败：{exc}")
        return

    _render_result(result)
    if output:
        console.print(f"\n✅ 报告已导出: {_safe(output)}")


def _render_result(result: MiningResult) -> None:
    """打印分析结果。

    完整的卡片渲染由 ``render`` 模块（M1 里程碑）负责，这里只给出终端摘要。
    """
    console.print(
        f"\n[bold]📊 {_safe(result.keyword)}[/bold] —— "
        f"{len(result.cards)} 张机会卡片 / {len(result.clusters)} 个痛点簇"
    )
    console.print(f"   样本: {result.total_notes} 篇笔记 / {result.total_comments} 条评论")
    console.print(f"   成本: {_safe(result.cost.summary())}")

    for message in result.notes:
        console.print(f"   [yellow]⚠️  {_safe(message)}[/yellow]")

    table = Table(title="机会卡片", show_lines=False)
    table.add_column("机会分", justify="right")
    table.add_column("方向", overflow="fold", max_width=48)
    table.add_column("提及", justify="right")
    table.add_column("活跃竞品", justify="right")

    for card in result.top_cards[:10]:
        active = sum(1 for c in card.competitors if not c.is_stale)
        table.add_row(
            _safe(f"{card.score:.0f}"),
            _safe(card.title),
            _safe(card.pain.size),
            _safe(active),
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #


@main.command()
@click.option(
    "--keyword",
    "-k",
    required=True,
    callback=_validate_keyword,
    help="品类关键词 (e.g., 防晒霜)",
)
@click.option("--notes", "-n", type=int, default=None, help="采集笔记数量")
@click.option("--comments", "-c", type=int, default=None, help="每篇笔记最多采集的评论数")
@click.option("--backend", type=BACKEND_CHOICES, default=None, help="覆盖采集后端")
@click.option(
    "--save",
    type=click.Path(dir_okay=False),
    default=None,
    help="把原始采集结果保存为 JSON（供调试或后续离线分析）",
)
def collect(
    keyword: str,
    notes: int | None,
    comments: int | None,
    backend: str | None,
    save: str | None,
) -> None:
    """📥 只做采集，打印统计结果（用于验证采集层与配置）。

    \b
    示例:
        xhs-pain-miner collect -k 防晒霜 --backend fixture
        xhs-pain-miner collect -k 防晒霜 --save corpus.json
    """
    settings = _build_settings(collector_backend=backend)
    miner = PainMiner(settings=settings)

    with console.status(f"正在采集「{_safe(keyword)}」…", spinner="dots"):
        try:
            corpus = miner.collect(keyword, limit=notes, max_comments_per_note=comments)
        except CollectorError as exc:
            _fail(str(exc))
            return

    console.print(f"✅ 采集完成：[bold]「{_safe(keyword)}」[/bold] {_safe(corpus.summary())}")

    if not corpus.notes:
        console.print(
            "[yellow]⚠️  没有采集到任何笔记。可能是关键词无结果，或采集后端未正常工作。[/yellow]"
        )

    if corpus.notes:
        table = Table(title="笔记样本", show_lines=False)
        table.add_column("标题", overflow="fold", max_width=48)
        table.add_column("点赞", justify="right")
        table.add_column("评论数", justify="right")
        table.add_column("图片", justify="right")
        for note in corpus.notes[:8]:
            table.add_row(
                _safe(note.title or note.desc[:40] or note.note_id),
                _safe(note.likes),
                _safe(note.comments_count),
                _safe(len(note.images)),
            )
        console.print(table)

    if corpus.comments:
        table = Table(title="评论样本", show_lines=False)
        table.add_column("内容", overflow="fold", max_width=56)
        table.add_column("点赞", justify="right")
        table.add_column("类型")
        for comment in corpus.comments[:10]:
            table.add_row(
                _safe(comment.content),
                _safe(comment.likes),
                "二级" if comment.parent_id else "一级",
            )
        console.print(table)

    if save is not None:
        # 用 `is not None` 而不是真值判断：`--save ""` 曾经静默什么都不做且 exit 0
        if not save.strip():
            _fail("--save 的路径不能为空", code=EXIT_CONFIG)
            return

        payload = asdict(corpus)
        try:
            Path(save).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            _fail(f"无法写入 {save}: {exc}")
            return
        console.print(f"💾 原始语料已保存: {_safe(save)}")


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


@main.command()
def doctor() -> None:
    """🩺 诊断环境配置（只做本地检查，不发起网络请求，也不会创建任何目录）。

    \b
    示例:
        xhs-pain-miner doctor
    """
    settings = _build_settings()
    checks = run_diagnostics(settings)

    table = Table(title="🩺 环境诊断", show_header=True, header_style="bold")
    table.add_column("", width=3, justify="center")
    table.add_column("检查项", style="bold", no_wrap=True)
    table.add_column("结果", overflow="fold")

    for check in checks:
        icon, style = _STATUS_STYLE[check.status]
        # escape 是必须的：诊断文本里含 `.[analysis]` 这类方括号，
        # 不转义会被 rich 当成样式标签吃掉。
        table.add_row(icon, _safe(check.name), f"[{style}]{_safe(check.detail)}[/{style}]")

    console.print(table)

    hints = [c for c in checks if c.hint and c.status != "ok"]
    if hints:
        console.print("\n[bold]修复建议[/bold]")
        for check in hints:
            console.print(f"  [yellow]•[/yellow] {_safe(check.name)}: {_safe(check.hint)}")

    if has_blocking_issue(checks):
        err_console.print("\n[red]存在阻塞项，请先按上面的建议修复。[/red]")
        raise SystemExit(EXIT_CONFIG)

    console.print("\n[green]✅ 环境可用。[/green]")


if __name__ == "__main__":
    main()
