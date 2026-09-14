"""Tests for XHS Pain Miner core module."""

import pytest
from xhs_pain_miner import PainMiner
from xhs_pain_miner.pain_miner import AnalysisReport, PainPoint


class TestPainMiner:
    """Test PainMiner initialization."""

    def test_init_default(self):
        miner = PainMiner()
        assert miner.llm_provider == "openai"
        assert miner.model == "gpt-4o"

    def test_init_custom(self):
        miner = PainMiner(api_key="test", llm_provider="deepseek", model="deepseek-chat")
        assert miner.api_key == "test"
        assert miner.llm_provider == "deepseek"


class TestAnalysisReport:
    """Test AnalysisReport output."""

    def test_empty_report(self):
        report = AnalysisReport(keyword="测试")
        assert report.keyword == "测试"
        assert report.pain_points == []

    def test_export_json(self, tmp_path):
        report = AnalysisReport(keyword="测试")
        report.pain_points = [
            PainPoint(description="痛点1", frequency=10, sentiment_score=-0.8),
            PainPoint(description="痛点2", frequency=5, sentiment_score=-0.5),
        ]
        path = tmp_path / "test.json"
        report.export_json(str(path))
        assert path.exists()
