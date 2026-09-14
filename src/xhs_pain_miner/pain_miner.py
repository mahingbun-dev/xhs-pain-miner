"""XHS Pain Miner — Core pain point discovery engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PainPoint:
    """A single discovered pain point."""

    description: str
    frequency: int
    sentiment_score: float  # -1.0 (negative) to 1.0 (positive)
    examples: list[str] = field(default_factory=list)
    trend: str = "stable"  # rising | falling | stable
    category: str = ""


@dataclass
class AnalysisReport:
    """Complete pain point analysis report for a keyword."""

    keyword: str
    pain_points: list[PainPoint] = field(default_factory=list)
    total_notes: int = 0
    total_comments: int = 0

    def print_pain_points(self) -> None:
        """Print pain points to console with formatting."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"🔍 {self.keyword} — 用户痛点地图")
        table.add_column("排名", justify="center", width=4)
        table.add_column("痛点", min_width=20)
        table.add_column("提及次数", justify="right")
        table.add_column("情感", justify="center")
        table.add_column("趋势", justify="center")

        trend_icons = {"rising": "↗️", "falling": "↘️", "stable": "→"}

        for i, pp in enumerate(self.pain_points[:10], 1):
            sentiment = "🔴" if pp.sentiment_score < -0.3 else "🟡" if pp.sentiment_score < 0.3 else "🟢"
            table.add_row(
                str(i),
                pp.description,
                str(pp.frequency),
                sentiment,
                trend_icons.get(pp.trend, "→"),
            )

        console.print(table)

    def export_html(self, path: str) -> None:
        """Export report as interactive HTML."""
        # TODO: Phase 1 implement
        raise NotImplementedError

    def export_pdf(self, path: str) -> None:
        """Export report as PDF."""
        # TODO: Phase 1 implement
        raise NotImplementedError

    def export_json(self, path: str) -> None:
        """Export report as JSON data."""
        import json

        data = {
            "keyword": self.keyword,
            "total_notes": self.total_notes,
            "total_comments": self.total_comments,
            "pain_points": [
                {
                    "description": pp.description,
                    "frequency": pp.frequency,
                    "sentiment_score": pp.sentiment_score,
                    "trend": pp.trend,
                    "examples": pp.examples,
                    "category": pp.category,
                }
                for pp in self.pain_points
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class PainMiner:
    """Main entry point for pain point discovery.

    Usage::

        from xhs_pain_miner import PainMiner

        miner = PainMiner(api_key="your-key")
        report = miner.analyze(keyword="防晒霜", notes_count=200)
        report.print_pain_points()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_provider: str = "openai",
        model: str = "gpt-4o",
    ) -> None:
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model = model

    def analyze(
        self,
        keyword: str,
        notes_count: int = 200,
        include_comments: bool = True,
        include_images: bool = True,
    ) -> AnalysisReport:
        """Analyze a keyword and discover user pain points.

        Args:
            keyword: 品类关键词 (e.g., "防晒霜", "婴儿辅食")
            notes_count: 采集笔记数量
            include_comments: 是否分析评论区
            include_images: 是否分析图片内容

        Returns:
            AnalysisReport with discovered pain points
        """
        report = AnalysisReport(keyword=keyword)

        # TODO: Phase 1 — Wire up actual pipeline
        # 1. Crawl notes + comments via crawler module
        # 2. Analyze text via analysis module
        # 3. Analyze images via vlm module
        # 4. Discover pain points via miner module
        # 5. Generate visualizations via visualize module

        return report
