"""内置脱敏样例后端 —— 让项目离线可跑、可测、可演示。

这是仓库内**唯一**随包分发的采集实现，数据是虚构的、已脱敏的样例语料，
不来自任何真实平台采集，也不产生任何网络请求。

它的存在解决了三个问题：

1. CI 无需网络与 API Key 即可端到端测试。
2. 新用户 ``pip install`` 后能立刻看到完整产物，不必先配好采集器。
3. 效果验收时可以固定输入，对比不同算法版本的输出差异。
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.models import RawComment, RawCorpus, RawNote

FIXTURE_RESOURCE = "data/fixture_corpus.json"
"""样例数据在包内的相对路径。"""


def _parse_dt(value: Any) -> datetime | None:
    """把 ISO 8601 字符串解析成 datetime，失败则返回 None。"""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _ensure_object(data: object, source: str) -> dict[str, Any]:
    """确认解析结果是一个 JSON 对象。

    ``json.loads`` 的返回类型是 ``Any``，显式收窄既能消除 mypy 的 ``no-any-return``，
    也能在样例文件被误改成数组或标量时给出可读错误，而不是后续的 ``KeyError``。

    Raises:
        CollectorError: 顶层不是 JSON 对象。
    """
    if not isinstance(data, dict):
        raise CollectorError(f"{source} 的顶层必须是 JSON 对象，实际是 {type(data).__name__}")
    return data


def load_fixture_data(path: Path | None = None) -> dict[str, Any]:
    """读取样例语料。

    Args:
        path: 自定义样例文件路径。为 ``None`` 时读取包内内置数据。

    Returns:
        解析后的 JSON 结构。

    Raises:
        CollectorError: 文件不存在、JSON 格式非法，或顶层不是对象。
    """
    if path is not None:
        if not path.is_file():
            raise CollectorError(f"样例数据文件不存在: {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CollectorError(f"样例数据不是合法 JSON: {path} ({exc})") from exc
        return _ensure_object(parsed, f"样例数据 {path}")

    try:
        resource = resources.files("xhs_pain_miner").joinpath(FIXTURE_RESOURCE)
        raw = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise CollectorError(f"无法读取内置样例数据 {FIXTURE_RESOURCE}: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CollectorError(f"内置样例数据不是合法 JSON ({FIXTURE_RESOURCE}): {exc}") from exc
    return _ensure_object(parsed, f"内置样例数据 {FIXTURE_RESOURCE}")


def _truth_label(item: dict[str, Any]) -> str:
    """取出样例语料里的 ``truth_label``（真实痛点标注）。

    这是验收门③「频次误差 < 15%」能**自动**计算的前提：没有它，聚类准不准只能靠
    人工逐条数原文。它随 :attr:`~xhs_pain_miner.models.RawNote.extra` 传递 ——
    ``RawNote`` / ``RawComment`` 是采集层的通用模型，不该为了一个只在 fixture 里
    存在的验收字段开新属性。

    非字符串一律当作没有标注：标注错了比没有标注更危险，它会让验收数字看起来
    正常却指向错误的结论。
    """
    value = item.get("truth_label", "")
    return value if isinstance(value, str) else ""


def parse_corpus(data: dict[str, Any], *, keyword: str | None = None) -> RawCorpus:
    """把样例 JSON 转换成 :class:`RawCorpus`。

    只搬运 ``truth_label`` 一个额外字段，不把整个 JSON 条目塞进 ``extra`` ——
    样例文件未来可能加入更多平台字段，无差别透传会把未经哈希的标识一路带到下游。

    Args:
        data: :func:`load_fixture_data` 的返回值。
        keyword: 覆盖语料中的关键词（用户查询的可能是别的品类）。

    Returns:
        转换后的语料对象。
    """
    notes = [
        RawNote(
            note_id=str(item.get("note_id", "")),
            title=item.get("title", ""),
            desc=item.get("desc", ""),
            url=item.get("url", ""),
            images=list(item.get("images", [])),
            likes=int(item.get("likes", 0)),
            collects=int(item.get("collects", 0)),
            comments_count=int(item.get("comments_count", 0)),
            publish_time=_parse_dt(item.get("publish_time")),
            author_hash=item.get("author_hash", ""),
            extra={"truth_label": _truth_label(item)},
        )
        for item in data.get("notes", [])
    ]
    comments = [
        RawComment(
            comment_id=str(item.get("comment_id", "")),
            content=item.get("content", ""),
            likes=int(item.get("likes", 0)),
            parent_id=item.get("parent_id"),
            note_id=str(item.get("note_id", "")),
            created_at=_parse_dt(item.get("created_at")),
            user_hash=item.get("user_hash", ""),
            extra={"truth_label": _truth_label(item)},
        )
        for item in data.get("comments", [])
    ]
    return RawCorpus(
        keyword=keyword or data.get("keyword", ""),
        notes=notes,
        comments=comments,
        backend="fixture",
    )


class FixtureBackend:
    """读取内置样例语料的采集后端。

    注意：它**不联网**，也不会按关键词真的去检索 —— 无论查询什么品类，
    返回的都是同一份样例语料。:attr:`name` 会如实标明这一点，避免用户误以为是真实数据。
    """

    name = "fixture（内置样例数据，非真实采集）"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        """惰性加载并缓存样例数据。"""
        if self._data is None:
            self._data = load_fixture_data(self._path)
        return self._data

    def collect(
        self,
        keyword: str,
        *,
        limit: int,
        max_comments_per_note: int = 20,
    ) -> RawCorpus:
        """返回样例语料（受 ``limit`` 与 ``max_comments_per_note`` 约束）。

        Args:
            keyword: 品类关键词，仅用于填充返回值的 ``keyword`` 字段。
            limit: 最多返回的笔记数。
            max_comments_per_note: 每篇笔记最多保留的评论数。

        Returns:
            样例语料。
        """
        corpus = parse_corpus(self._load(), keyword=keyword)

        notes = corpus.notes[: max(limit, 0)]
        kept_ids = {note.note_id for note in notes}

        per_note_count: dict[str, int] = {}
        comments: list[RawComment] = []
        for comment in corpus.comments:
            if comment.note_id not in kept_ids:
                continue
            used = per_note_count.get(comment.note_id, 0)
            if used >= max_comments_per_note:
                continue
            per_note_count[comment.note_id] = used + 1
            comments.append(comment)

        corpus.notes = notes
        corpus.comments = comments
        return corpus

    def available(self) -> bool:
        """样例文件可读即视为可用。"""
        try:
            self._load()
        except CollectorError:
            return False
        return True
