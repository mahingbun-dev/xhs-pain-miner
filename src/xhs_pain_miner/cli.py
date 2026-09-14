"""XHS Pain Miner CLI — Command line interface for pain point discovery."""

from __future__ import annotations

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="xhs-pain-miner")
def main() -> None:
    """🔍 XHS Pain Miner — 小红书用户痛点挖掘工具

    AI-powered discovery of real user pain points from Xiaohongshu (小红书/RedNote).

    Quick start:
        xhs-pain-miner analyze --keyword "防晒霜"
        xhs-pain-miner serve --port 8080
    """
    pass


@main.command()
@click.option("--keyword", "-k", required=True, help="品类关键词 (e.g., 防晒霜)")
@click.option("--notes", "-n", default=200, help="采集笔记数量")
@click.option("--no-comments", is_flag=True, help="不分析评论区")
@click.option("--no-images", is_flag=True, help="不分析图片")
@click.option("--output", "-o", default=None, help="输出文件路径")
def analyze(keyword: str, notes: int, no_comments: bool, no_images: bool, output: str | None) -> None:
    """📊 分析一个品类的用户痛点

    Example: xhs-pain-miner analyze -k "防晒霜" -n 200
    """
    from xhs_pain_miner import PainMiner

    click.echo(f"🔍 正在分析「{keyword}」的用户痛点...")
    click.echo(f"   📝 笔记数量: {notes}")
    click.echo(f"   💬 评论分析: {'✅' if not no_comments else '❌'}")
    click.echo(f"   🖼️  图片分析: {'✅' if not no_images else '❌'}")

    miner = PainMiner()
    report = miner.analyze(
        keyword=keyword,
        notes_count=notes,
        include_comments=not no_comments,
        include_images=not no_images,
    )

    report.print_pain_points()

    if output:
        if output.endswith(".json"):
            report.export_json(output)
        elif output.endswith(".html"):
            report.export_html(output)
        elif output.endswith(".pdf"):
            report.export_pdf(output)
        click.echo(f"\n✅ 报告已导出: {output}")


@main.command()
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", "-p", default=8080, help="端口号")
def serve(host: str, port: int) -> None:
    """🌐 启动 Web 仪表盘

    Example: xhs-pain-miner serve --port 8080
    """
    click.echo(f"🌐 启动 Web 仪表盘: http://{host}:{port}")
    # TODO: Phase 1 — FastAPI + 前端
    click.echo("⏳ Web 仪表盘功能开发中，敬请期待！")


if __name__ == "__main__":
    main()
