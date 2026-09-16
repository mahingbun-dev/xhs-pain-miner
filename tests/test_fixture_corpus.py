"""内置语料的结构与标注完整性测试。

语料是验收门③「频次误差 < 15%」唯一的 ground truth 来源，因此它自己的正确性必须
先被钉住。这里守的是四类**不会让任何代码报错、只会让验收数字失真**的问题：

1. **结构**：篇数、每篇评论数、主题分布 —— 分布塌了，"抽 3 个痛点核对频次"就抽不出
   有代表性的样本。
2. **标注**：``truth_label`` 的取值集合必须与 ``_meta.expected_clusters`` 完全一致，
   且每个主题的标注数与笔记数对得上。标签错位是最危险的一类缺陷。
3. **清洗口径**：噪声必须真的被 :func:`~xhs_pain_miner.pipeline.clean.is_noise` 拦下，
   而**主题文本一条都不能被误杀** —— 后者会让某个痛点的频次凭空偏低。
4. **确定性**：提交的 JSON 必须与 ``tools/build_fixture.py`` 当前的产出逐字节一致，
   否则"重新生成一遍"就会静默改掉验收基准。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from xhs_pain_miner.collectors.fixture import load_fixture_data, parse_corpus
from xhs_pain_miner.pipeline.clean import build_units, is_noise

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "src" / "xhs_pain_miner" / "data" / "fixture_corpus.json"
BUILD_TOOL_PATH = REPO_ROOT / "tools" / "build_fixture.py"

ANCHOR = datetime(2026, 9, 15, 12, 0, tzinfo=timezone(timedelta(hours=8)))
"""与生成脚本一致的验收时间锚点。"""

EXPECTED_NOTES = 201
MIN_COMMENTS = 800
COMMENTS_PER_NOTE = (3, 8)
NOTES_PER_TOPIC = (18, 22)
TOPIC_COUNT = 10
NOISE_RATIO = (0.10, 0.20)

_SYNTHETIC = re.compile(r"^synthetic://(?P<note_id>[^/]+)/(?P<index>\d+)$")


def _corpus() -> dict[str, Any]:
    """读取提交进仓库的语料文件。"""
    # json.loads 的返回类型是 Any，显式收窄才能通过 mypy 的 no-any-return
    return cast("dict[str, Any]", json.loads(CORPUS_PATH.read_text(encoding="utf-8")))


def _build_tool() -> Any:
    """按路径加载生成脚本（``tools/`` 不是包，只能这样导入）。"""
    spec = importlib.util.spec_from_file_location("build_fixture_tool", BUILD_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _days_ago(iso: str) -> float:
    """距验收锚点的天数。"""
    return (ANCHOR - datetime.fromisoformat(iso)).total_seconds() / 86400


class TestStructure:
    """语料规模与形状。"""

    def test_note_count(self):
        assert len(_corpus()["notes"]) == EXPECTED_NOTES

    def test_comment_count(self):
        assert len(_corpus()["comments"]) >= MIN_COMMENTS

    def test_comments_per_note(self):
        """每篇 3-8 条评论，且下限与上限都真的出现过。"""
        counts = Counter(c["note_id"] for c in _corpus()["comments"])
        per_note = [counts[note["note_id"]] for note in _corpus()["notes"]]
        assert min(per_note) == COMMENTS_PER_NOTE[0]
        assert max(per_note) == COMMENTS_PER_NOTE[1]

    def test_required_fields(self):
        data = _corpus()
        for note in data["notes"]:
            assert set(note) == {
                "note_id",
                "title",
                "desc",
                "url",
                "images",
                "likes",
                "collects",
                "comments_count",
                "publish_time",
                "author_hash",
                "truth_label",
            }
        for comment in data["comments"]:
            assert set(comment) == {
                "comment_id",
                "note_id",
                "content",
                "likes",
                "parent_id",
                "created_at",
                "user_hash",
                "truth_label",
            }

    def test_ids_are_unique(self):
        data = _corpus()
        assert len({n["note_id"] for n in data["notes"]}) == len(data["notes"])
        assert len({c["comment_id"] for c in data["comments"]}) == len(data["comments"])

    def test_comments_belong_to_existing_notes(self):
        note_ids = {n["note_id"] for n in _corpus()["notes"]}
        assert {c["note_id"] for c in _corpus()["comments"]} <= note_ids

    def test_has_second_level_comments(self):
        """二级评论是深层痛点的来源，语料里必须真的存在。"""
        data = _corpus()
        ids = {c["comment_id"] for c in data["comments"]}
        replies = [c for c in data["comments"] if c["parent_id"]]
        assert replies
        for reply in replies:
            assert reply["parent_id"] in ids


class TestTopicDistribution:
    """10 个痛点主题的分布。"""

    def test_ten_topics(self):
        labels = set(_corpus()["_meta"]["expected_clusters"])
        assert len(labels) == TOPIC_COUNT

    def test_each_topic_has_enough_notes(self):
        counts = Counter(n["truth_label"] for n in _corpus()["notes"])
        assert set(counts) == set(_corpus()["_meta"]["expected_clusters"])
        for topic, count in counts.items():
            assert NOTES_PER_TOPIC[0] <= count <= NOTES_PER_TOPIC[1], f"{topic}: {count}"

    def test_topic_note_counts_sum_to_all_notes(self):
        """★ 所有笔记都必须归属于某个主题：漏标会让总频次对不上。"""
        counts = Counter(n["truth_label"] for n in _corpus()["notes"])
        assert sum(counts.values()) == EXPECTED_NOTES

    def test_truth_labels_are_never_empty_in_notes(self):
        assert all(n["truth_label"] for n in _corpus()["notes"])

    def test_truth_label_values_are_exactly_the_meta_labels(self):
        """★ 取值集合一旦多出或少了标签，验收指标就会悄悄算错一个类别。"""
        data = _corpus()
        labels = set(data["_meta"]["expected_clusters"])
        used = {c["truth_label"] for c in data["comments"]} - {""}
        assert used == labels

    def test_comment_labels_match_their_note(self):
        labels = {n["note_id"]: n["truth_label"] for n in _corpus()["notes"]}
        for comment in _corpus()["comments"]:
            expected = labels[comment["note_id"]] if comment["truth_label"] else ""
            assert comment["truth_label"] == expected


class TestCleanAgreement:
    """语料与清洗口径必须一致 —— 这是"噪声能被过滤"这条承诺的实际检验。"""

    def test_noise_is_actually_removed_by_is_noise(self):
        """★ 标为噪声的文本必须真的被 is_noise 拦下，否则它们会在 embedding 里
        聚出一个"求链接"的假簇。"""
        noise = [c["content"] for c in _corpus()["comments"] if not c["truth_label"]]
        assert noise
        survivors = [text for text in noise if not is_noise(text)]
        assert survivors == []

    def test_topic_text_is_never_dropped_by_is_noise(self):
        """★ 误杀比漏杀更危险：一条被丢掉的真实抱怨会让痛点频次凭空偏低。"""
        data = _corpus()
        dropped = [
            note["title"] for note in data["notes"] if is_noise(note["title"] + " " + note["desc"])
        ]
        dropped += [
            c["content"] for c in data["comments"] if c["truth_label"] and is_noise(c["content"])
        ]
        assert dropped == []

    def test_noise_ratio(self):
        data = _corpus()
        noise = sum(1 for c in data["comments"] if not c["truth_label"])
        ratio = noise / len(data["comments"])
        assert NOISE_RATIO[0] <= ratio <= NOISE_RATIO[1], f"噪声比例 {ratio:.3f}"

    def test_topic_texts_are_distinct(self):
        """★ 同一主题内大量重复文本会让聚类质量虚高 —— 那不是聚类的功劳。"""
        comments = [c["content"] for c in _corpus()["comments"] if c["truth_label"]]
        assert len(set(comments)) / len(comments) > 0.95
        titles = [n["title"] for n in _corpus()["notes"]]
        assert len(set(titles)) == len(titles)
        descs = [n["desc"] for n in _corpus()["notes"]]
        assert len(set(descs)) / len(descs) > 0.9


class TestMediaAndPrivacy:
    """图片协议与合规。"""

    def test_images_use_the_synthetic_scheme(self):
        for note in _corpus()["notes"]:
            assert note["images"], f"{note['note_id']} 没有图片"
            for index, url in enumerate(note["images"]):
                match = _SYNTHETIC.match(url)
                assert match, f"图片地址不是合成图协议: {url}"
                assert match.group("note_id") == note["note_id"]
                assert int(match.group("index")) == index

    def test_urls_are_not_requestable(self):
        """样例数据不该让任何人真的去请求平台 CDN。"""
        for note in _corpus()["notes"]:
            assert ".invalid" in note["url"]

    def test_user_identifiers_are_hashed(self):
        """★ 合规：语料里不得出现未哈希的用户标识。"""
        data = _corpus()
        for note in data["notes"]:
            assert len(note["author_hash"]) == 16
            assert int(note["author_hash"], 16) >= 0
        for comment in data["comments"]:
            assert len(comment["user_hash"]) == 16
            assert int(comment["user_hash"], 16) >= 0


class TestSignals:
    """供「痛点强度」「增长趋势」两个因子使用的信号。"""

    def test_likes_have_spread(self):
        """★ 点赞数若都填同一个值，log 归一就测不出任何东西。"""
        data = _corpus()
        note_likes = [n["likes"] for n in data["notes"]]
        assert min(note_likes) == 0
        assert max(note_likes) > 1000
        assert len(set(note_likes)) > 100

        comment_likes = [c["likes"] for c in data["comments"]]
        assert min(comment_likes) == 0
        assert max(comment_likes) > 500
        assert len(set(comment_likes)) > 100

    def test_publish_times_are_not_in_the_future(self):
        for note in _corpus()["notes"]:
            assert datetime.fromisoformat(note["publish_time"]) <= ANCHOR

    def test_time_span_covers_about_eighteen_months(self):
        days = [_days_ago(n["publish_time"]) for n in _corpus()["notes"]]
        assert max(days) - min(days) > 500

    def test_some_topics_are_recent_heavy(self):
        """★ 「增长趋势」因子要求语料里同时存在"在变多"和"在变少"的主题。

        全部主题均匀铺开的话，stage 永远判成 stable，这个因子等于没实现。
        """
        by_topic: dict[str, list[float]] = {}
        for note in _corpus()["notes"]:
            by_topic.setdefault(note["truth_label"], []).append(_days_ago(note["publish_time"]))
        medians = {topic: statistics.median(days) for topic, days in by_topic.items()}

        recent = [topic for topic, median in medians.items() if median <= 180]
        old = [topic for topic, median in medians.items() if median >= 350]
        assert len(recent) >= 3, medians
        assert len(old) >= 3, medians
        assert max(medians[t] for t in recent) < min(medians[t] for t in old) - 150


class TestGeneratorDeterminism:
    """生成脚本必须可重复运行，且与提交的文件一致。"""

    def test_committed_file_matches_generator_output(self):
        """★ 提交的 JSON 与脚本产出不一致时，"重新生成一遍"会静默改掉验收基准。"""
        tool = _build_tool()
        expected = json.dumps(tool.build_corpus(), ensure_ascii=False, indent=2) + "\n"
        assert CORPUS_PATH.read_text(encoding="utf-8") == expected

    def test_generator_is_repeatable(self):
        tool = _build_tool()
        assert tool.build_corpus() == tool.build_corpus()

    def test_generator_output_passes_its_own_checks(self):
        """生成器自带的断言必须真的成立，而不是等着下游去发现。"""
        tool = _build_tool()
        tool._verify(tool.build_corpus())

    def test_anchor_is_fixed(self):
        """★ 锚点若改成读当前时间，"可重复运行"立刻失效。"""
        tool = _build_tool()
        assert tool.ANCHOR == ANCHOR

    def test_generator_never_reads_the_wall_clock(self):
        """★ 上一条断言在验收环境里可能漏判 —— 锚点就是按"当天"选的，机器时间
        恰好等于锚点时，读当前时间的实现也能通过。

        所以这里改为在语法树上找 ``.now()`` / ``.today()`` 调用：只要脚本在任何
        位置读了当前时间，产物就不再只由输入决定。字符串与注释不在 AST 的 Call
        节点里，因此文档里提到这些 API 不会误报。
        """
        tree = ast.parse(BUILD_TOOL_PATH.read_text(encoding="utf-8"))
        offenders = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"now", "today", "utcnow"}
        ]
        assert offenders == []


class TestEndToEndTruthLabels:
    """从 JSON 到文本单元的完整链路 —— 验收门③的前置条件。"""

    def test_truth_labels_reach_text_units(self):
        """★ 这是本次语料改造的**核心目的**：没有它，频次误差只能靠人工数原文。"""
        units = build_units(parse_corpus(load_fixture_data()))
        labelled = [unit for unit in units if unit.truth_label]
        assert labelled
        assert all(unit.text and unit.weight > 0 for unit in labelled)

    def test_every_topic_survives_cleaning(self):
        """★ 主题被清洗整段吃掉的话，那个痛点的频次会直接变成 0。"""
        data = _corpus()
        units = build_units(parse_corpus(load_fixture_data()))
        labels = Counter(unit.truth_label for unit in units)
        expected = set(data["_meta"]["expected_clusters"])
        assert set(labels) - {""} == expected
        for topic in expected:
            assert labels[topic] > 0

    def test_no_noise_unit_survives(self):
        """清洗后不该剩下任何一条无标注文本（fixture 里"无标注"就等于"噪声"）。"""
        units = build_units(parse_corpus(load_fixture_data()))
        assert [unit.text for unit in units if not unit.truth_label] == []

    def test_unit_count_matches_surviving_texts(self):
        data = _corpus()
        topic_comments = sum(1 for c in data["comments"] if c["truth_label"])
        units = build_units(parse_corpus(load_fixture_data()))
        assert len(units) == len(data["notes"]) + topic_comments

    def test_topic_frequency_is_measurable(self):
        """验收门③要抽 3 个痛点核对频次，前提是每个主题都有足够多的证据。"""
        units = build_units(parse_corpus(load_fixture_data()))
        labels = Counter(unit.truth_label for unit in units)
        smallest = min(labels[topic] for topic in _corpus()["_meta"]["expected_clusters"])
        assert smallest >= 40

    def test_corpus_keyword_is_preserved(self):
        corpus = parse_corpus(load_fixture_data())
        assert corpus.keyword == "防晒霜"
