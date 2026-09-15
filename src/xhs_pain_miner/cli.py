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
from rich.table import Table

from xhs_pain_miner import __version__
from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.diagnostics import has_blocking_issue, run_diagnostics
from xhs_pain_miner.llm.base import LLMError
from xhs_pain_miner.models import MiningResult
from xhs_pain_miner.pain_miner import PainMiner, normalize_keyword
from xhs_pain_miner.pipeline.deps import MissingDependencyError

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
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="报告输出路径，按扩展名决定格式（.html 或 .md）",
)
@click.option("--no-save", is_flag=True, help="只打印终端摘要，不写报告文件")
@click.option(
    "--weights",
    default=None,
    help='覆盖机会分权重，如 "gap=0.4,trend=0.3"',
)
@click.option("--backend", type=BACKEND_CHOICES, default=None, help="覆盖采集后端")
@click.option("--yes", "-y", is_flag=True, help="跳过 VLM 成本预估确认")
def mine(
    keyword: str,
    notes: int | None,
    deep: bool,
    output: str | None,
    no_save: bool,
    weights: str | None,
    backend: str | None,
    yes: bool,
) -> None:
    """📊 采集并分析一个品类，产出机会卡片。

    \b
    示例:
        xhs-pain-miner mine -k 防晒霜 -n 200
        xhs-pain-miner mine -k 防晒霜 --deep -o 防晒霜-机会卡片.html
        xhs-pain-miner mine -k 防晒霜 --weights "gap=0.4,trend=0.3"
    """
    try:
        settings = _build_settings(collector_backend=backend)
        if weights:
            settings = _apply_weights(settings, weights)
    except ValueError as exc:
        _fail(str(exc), code=EXIT_CONFIG)
        return

    console.print(f"🔍 [bold]正在分析「{_safe(keyword)}」[/bold]")
    console.print(f"   采集后端: {_safe(settings.collector_backend)}")
    console.print(f"   笔记数量: {notes if notes is not None else settings.max_notes}")
    console.print(f"   图片分析: {'✅ 开启 (--deep)' if deep else '❌ 关闭'}")

    try:
        # 报告与 VLM 缓存都要落盘，先把目录建好 —— 否则会等到写文件那一刻才失败，
        # 而那时一整轮 LLM 调用已经花掉了。
        settings.ensure_dirs()
    except OSError as exc:
        _fail(f"无法创建输出目录: {exc}")
        return

    # 快速失败：缺 API Key 是最常见的配置错误。不在这里拦住的话，用户会先看到
    # 「采集完成」的进度，跑完一遍采集才被告知缺 Key —— 白等一场，还会误以为
    # 采集环节有问题。依赖缺失同理，但它们由 mine() 内部按需报出（错误信息更具体）。
    if not settings.llm_api_key:
        _fail(
            "未配置 LLM API Key。请在 .env 或环境变量中设置 LLM_API_KEY"
            "（也兼容 DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY）。\n"
            "  可先执行 `xhs-pain-miner doctor` 检查完整配置。",
            code=EXIT_CONFIG,
        )
        return

    miner = PainMiner(settings=settings)
    try:
        # 采集放在 mine() 之外，是为了让「成本确认」发生在**任何 LLM 调用之前**，
        # 且预估基于真实语料而不是猜测。
        with console.status("正在采集…", spinner="dots"):
            corpus = miner.collect(keyword, limit=notes)

        if deep:
            deep = _confirm_vlm_cost(miner, corpus, assume_yes=yes)

        with console.status("分析中…", spinner="dots") as status:
            result = miner.mine(
                keyword,
                corpus=corpus,
                deep=deep,
                progress=lambda stage, ratio: status.update(f"{stage} {ratio:.0%}"),
            )
    except CollectorError as exc:
        _fail(f"采集失败：{exc}")
        return
    except LLMError as exc:
        _fail(f"模型调用失败：{exc}")
        return
    except MissingDependencyError as exc:
        _fail(str(exc), code=EXIT_CONFIG)
        return
    finally:
        miner.close()

    _render_result(result)

    if not no_save:
        _write_report(result, output, settings.output_dir)


def _confirm_vlm_cost(miner: PainMiner, corpus: object, *, assume_yes: bool) -> bool:
    """展示图片分析成本预估并征求确认。

    Returns:
        是否继续做图片分析。**拒绝不会中断整次运行** —— 文本分析的结论依然有价值。
    """
    if not getattr(corpus, "notes", None):
        console.print("[yellow]⚠️  没有采集到笔记，跳过图片分析。[/yellow]")
        return False

    try:
        with console.status("正在预估图片分析成本…", spinner="dots"):
            estimate = miner.estimate_vlm_cost(corpus)  # type: ignore[arg-type]
    except (MissingDependencyError, OSError) as exc:
        console.print(f"[yellow]⚠️  无法预估图片分析成本（{_safe(exc)}），跳过图片分析。[/yellow]")
        return False

    console.print(f"\n🖼  [bold]图片分析预估[/bold]  {_safe(estimate.summary())}")
    if assume_yes:
        return True
    if click.confirm("继续发起 VLM 调用吗？", default=True):
        return True
    console.print("   [dim]已跳过图片分析，本次只分析文本。[/dim]")
    return False


def _render_result(result: MiningResult) -> None:
    """打印终端摘要。

    完整报告（含证据链与因子溯源）由 ``render`` 模块产出的 HTML 文件承载，
    终端只给出够用的概览。
    """
    console.print(
        f"\n[bold]📊 {_safe(result.keyword)}[/bold] —— "
        f"{len(result.cards)} 张机会卡片 / {len(result.clusters)} 个痛点簇"
    )
    console.print(f"   样本: {result.total_notes} 篇笔记 / {result.total_comments} 条评论")
    console.print(f"   成本: {_safe(result.cost.summary())}")

    for message in result.notes:
        console.print(f"   [yellow]⚠️  {_safe(message)}[/yellow]")

    if not result.cards:
        console.print("\n[yellow]没有产出任何机会卡片。[/yellow]")
        return

    table = Table(title="机会卡片", show_lines=False)
    table.add_column("机会分", justify="right")
    table.add_column("方向", overflow="fold", max_width=40)
    table.add_column("痛点", overflow="fold", max_width=22)
    table.add_column("提及", justify="right")
    table.add_column("活跃竞品", justify="right")

    for card in result.top_cards[:15]:
        active = sum(1 for c in card.competitors if not c.is_stale)
        table.add_row(
            _safe(f"{card.score:.0f}"),
            _safe(card.title),
            _safe(card.pain.label),
            _safe(card.pain.size),
            _safe(active),
        )
    console.print(table)


_WEIGHT_FIELDS = {
    "pain": "weight_pain_strength",
    "pain_strength": "weight_pain_strength",
    "volume": "weight_mention_volume",
    "mention_volume": "weight_mention_volume",
    "trend": "weight_growth_trend",
    "growth_trend": "weight_growth_trend",
    "gap": "weight_competitor_gap",
    "competitor_gap": "weight_competitor_gap",
    "feasibility": "weight_feasibility",
}


def _apply_weights(settings: Settings, spec: str) -> Settings:
    """解析 ``--weights "gap=0.4,trend=0.3"`` 并写回配置。

    Raises:
        ValueError: 格式错误，或使用了未知的权重名。**未知键必须报错而不是忽略** ——
            用户拼错一个键却以为调整生效了，是最难排查的一类问题。
    """
    updates: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, separator, raw = part.partition("=")
        if not separator:
            raise ValueError(f"--weights 的格式应为 key=value，收到 {part!r}")
        field = _WEIGHT_FIELDS.get(key.strip().lower())
        if field is None:
            allowed = ", ".join(sorted(_WEIGHT_FIELDS))
            raise ValueError(f"未知的权重名 {key.strip()!r}，可用: {allowed}")
        try:
            updates[field] = float(raw)
        except ValueError:
            raise ValueError(f"权重 {key.strip()!r} 的值不是数字: {raw!r}") from None

    if not updates:
        raise ValueError("--weights 未指定任何权重")
    return settings.model_copy(update=updates)


def _report_path(result: MiningResult, output: str | None, output_dir: Path) -> Path | None:
    """决定报告写到哪。返回 ``None`` 表示格式无法识别（已打印错误）。"""
    if output:
        path = Path(output)
        if not path.suffix:
            path = path.with_suffix(".html")
        return path
    return output_dir / f"{_safe_filename(result.keyword)}-机会卡片.html"


def _safe_filename(name: str) -> str:
    """把关键词变成安全的文件名片段。

    关键词来自用户输入，可能含 ``/`` 或 ``..``。不处理的话会把报告写到意料之外
    的路径上去（``-k "../../etc/x"``）。
    """
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return (cleaned or "report")[:60]


def _write_report(result: MiningResult, output: str | None, output_dir: Path) -> None:
    """把报告写到磁盘。按扩展名选渲染器。"""
    path = _report_path(result, output, output_dir)
    if path is None:
        return

    suffix = path.suffix.lower()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".md":
            from xhs_pain_miner.render.markdown import render_markdown

            text = render_markdown(result)
        elif suffix in (".html", ".htm"):
            from xhs_pain_miner.render.html import render_html

            text = render_html(result)
        else:
            _fail(f"不支持的输出格式 {suffix!r}，请用 .html 或 .md", code=EXIT_CONFIG)
            return
        # 必须显式指定 utf-8：默认编码在 Windows 上会让中文报告变成乱码
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _fail(f"无法写入报告 {path}: {exc}")
        return

    console.print(f"\n✅ 报告已导出: {_safe(str(path))}")


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
