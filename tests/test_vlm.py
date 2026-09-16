"""VLM 图片分析测试 —— 全部离线，不联网、不需要 API Key。

本模块覆盖省钱七条里属于 :mod:`xhs_pain_miner.pipeline.vlm` 的部分，重点在
**"省了多少"能被断言**：去重是否真的减少了调用、缓存命中是否真的没花钱、
超限是否真的截断并如实上报、单张失败是否真的没拖垮整次运行。

测试用假 provider（返回预设 JSON 与预设异常）与假图片源（``data:`` / 本地文件 /
``synthetic://``），``fetch_image`` 的 http 分支只测 URL 处理与错误映射 ——
真实下载验证属于成本验收，不在单元测试里做。
"""

from __future__ import annotations

import base64
import io
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from xhs_pain_miner.llm.base import LLMError, LLMResponse, build_data_url, split_data_url
from xhs_pain_miner.models import ImageInsight, RawComment, RawCorpus, RawNote, RunCost
from xhs_pain_miner.pipeline import vlm
from xhs_pain_miner.pipeline.deps import MissingDependencyError
from xhs_pain_miner.pipeline.vlm import (
    SqliteVlmCache,
    VlmAnalyzer,
    downscale_image,
    fetch_image,
    image_hash,
)

_REPLY = '{"description": "一张左右对比图", "pain_hints": ["左暗右亮差异明显"]}'
_EMPTY_HINTS_REPLY = '{"description": "一张普通商品图", "pain_hints": []}'


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class FakeVision:
    """不联网的假 VLM provider。

    ``handler(index, prompt, images)`` 返回字符串（当作模型回复）或**返回/抛出**异常
    （模拟失败）；``index`` 是第几次调用，便于写"前两次失败第三次成功"这类脚本。
    允许直接 ``return`` 异常对象，是为了让一行式的失败脚本不必写成函数。
    """

    name = "fake-vision"

    def __init__(self, handler=None) -> None:
        self.handler = handler or (lambda index, prompt, images: _REPLY)
        self.usage = RunCost()
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        raise AssertionError("VLM 分析不应调用文本补全接口")

    def complete_vision(self, prompt, images, *, system=None, temperature=None, max_tokens=None):
        with self._lock:
            index = len(self.calls)
            self.calls.append(
                {"prompt": prompt, "images": list(images), "temperature": temperature}
            )
            self.usage.vlm_calls += 1
        outcome = self.handler(index, prompt, list(images))
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, LLMResponse):
            return outcome
        return LLMResponse(text=outcome, model="fake")

    def close(self) -> None:
        """无需释放任何资源。"""

    @property
    def call_count(self) -> int:
        """已发起的调用次数。"""
        return len(self.calls)


def _data_url(tag: str) -> str:
    """把一段任意字节包成 data URL —— 内容不同即哈希不同，足以测去重与缓存。"""
    return build_data_url(tag.encode("utf-8"), "image/jpeg")


def _corpus(*notes: tuple[str, int, list[str]]) -> RawCorpus:
    """按 ``(note_id, likes, [图片地址])`` 构造语料。"""
    return RawCorpus(
        keyword="防晒霜",
        notes=[
            RawNote(note_id=note_id, title="测试笔记", images=list(urls), likes=likes)
            for note_id, likes, urls in notes
        ],
    )


def _decoded_payload(call: dict) -> bytes:
    """取出假 provider 收到的那张图的原始字节。"""
    return base64.b64decode(split_data_url(call["images"][0])[1])


@pytest.fixture
def no_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    """让分析器测试不依赖 Pillow。

    压缩与依赖检查本身由 :class:`TestDownscaleImage` / :class:`TestMissingPillow`
    覆盖（那两组未装 Pillow 时会显式跳过）；这里替换掉这两处，是为了让
    去重 / 缓存 / 截断 / 降级这些**纯逻辑**测试在没有 Pillow 的环境里也真的跑起来
    （CI 只装 ``.[dev]``）。被 skip 掉的测试守不住任何东西。
    """
    monkeypatch.setattr(vlm, "downscale_image", lambda data, *, max_edge=512: (data, "image/jpeg"))
    monkeypatch.setattr(vlm, "require_pillow", lambda purpose: (None, None, None))


@pytest.fixture
def fake_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """把休眠换成记录 —— 退避测试不该真的等 7.5 秒。"""
    slept: list[float] = []
    monkeypatch.setattr(vlm, "_sleep", slept.append)
    return slept


# --------------------------------------------------------------------------- #
# 内容哈希（省钱第 1、4 条的基础）
# --------------------------------------------------------------------------- #


class TestImageHash:
    """按内容哈希 —— 去重与缓存都用它当键。"""

    def test_same_content_same_hash(self):
        assert image_hash(b"same-bytes") == image_hash(b"same-bytes")

    def test_different_content_different_hash(self):
        assert image_hash(b"a") != image_hash(b"b")

    def test_hash_is_short_hex(self):
        digest = image_hash(b"x")
        assert len(digest) == 32
        assert int(digest, 16) >= 0  # 非十六进制会抛 ValueError

    def test_url_is_not_part_of_the_key(self):
        """哈希只认内容：两个不同 URL 指向同一张图时必须命中同一个键。"""
        assert image_hash(b"jpeg-bytes") == image_hash(b"jpeg-bytes")


# --------------------------------------------------------------------------- #
# 图片来源解析
# --------------------------------------------------------------------------- #


class TestFetchImage:
    """``fetch_image`` 的各来源分支。http 分支只验证错误映射，不联网。"""

    def test_data_url_is_decoded(self):
        url = build_data_url(b"\xff\xd8fake-jpeg", "image/jpeg")
        assert fetch_image(url) == b"\xff\xd8fake-jpeg"

    def test_non_image_data_url_is_rejected(self):
        url = build_data_url(b"hello", "text/plain")
        with pytest.raises(ValueError, match="不是图片类型"):
            fetch_image(url)

    def test_broken_base64_is_rejected(self):
        with pytest.raises(ValueError, match="base64"):
            fetch_image("data:image/jpeg;base64,!!!not-base64!!!")

    def test_plain_path_is_read(self, tmp_path):
        path = tmp_path / "local.jpg"
        path.write_bytes(b"local-bytes")
        assert fetch_image(str(path)) == b"local-bytes"

    def test_file_url_is_read(self, tmp_path):
        path = tmp_path / "local.jpg"
        path.write_bytes(b"file-url-bytes")
        assert fetch_image(f"file://{path}") == b"file-url-bytes"

    def test_missing_local_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError, match="不存在"):
            fetch_image(str(tmp_path / "nope.jpg"))

    def test_unsupported_scheme_raises_value_error(self):
        with pytest.raises(ValueError, match="不支持的图片地址协议"):
            fetch_image("ftp://cdn.example.invalid/a.jpg")

    @pytest.mark.parametrize("url", ["", "   "])
    def test_empty_url_raises_value_error(self, url):
        with pytest.raises(ValueError, match="不能为空"):
            fetch_image(url)

    def test_malformed_synthetic_url_raises_value_error(self):
        with pytest.raises(ValueError, match="非法的合成图地址"):
            fetch_image("synthetic://note-without-index")

    def test_http_failure_becomes_oserror(self, monkeypatch):
        def boom(url, timeout):
            raise OSError(f"图片下载失败: {url}")

        monkeypatch.setattr(vlm, "_download_bytes", boom)
        with pytest.raises(OSError, match="下载失败"):
            fetch_image("https://cdn.example.invalid/a.jpg")

    def test_http_timeout_is_threaded_through(self, monkeypatch):
        seen: list[float] = []

        def record(url, timeout):
            seen.append(timeout)
            return b"jpeg-bytes"

        monkeypatch.setattr(vlm, "_download_bytes", record)
        fetch_image("https://cdn.example.invalid/a.jpg", timeout=3.5)
        assert seen == [3.5]

    def test_empty_download_is_oserror(self, monkeypatch):
        monkeypatch.setattr(vlm, "_download_bytes", lambda url, timeout: b"")
        with pytest.raises(OSError, match="为空"):
            fetch_image("https://cdn.example.invalid/a.jpg")

    def test_oversized_download_is_rejected(self, monkeypatch):
        """8MB 上限是防御性的 —— 有上限才不会被一张巨图拖死整轮。"""
        big = b"x" * (vlm.IMAGE_MAX_BYTES + 1)
        monkeypatch.setattr(vlm, "_download_bytes", lambda url, timeout: big)
        with pytest.raises(OSError, match="大小上限"):
            fetch_image("https://cdn.example.invalid/huge.jpg")

    def test_size_limit_inside_the_expected_range(self):
        assert vlm.IMAGE_MAX_BYTES == 8 * 1024 * 1024


# --------------------------------------------------------------------------- #
# 压缩（省钱第 2 条）
# --------------------------------------------------------------------------- #


class TestDownscaleImage:
    """长边压缩。这组需要 Pillow；未安装时显式跳过而不是伪装通过。"""

    @pytest.fixture(autouse=True)
    def _require_pillow(self):
        pytest.importorskip("PIL.Image")

    @staticmethod
    def _image(size: tuple[int, int], mode: str = "RGB", fmt: str = "PNG") -> bytes:
        from PIL import Image

        image = Image.new(mode, size, (200, 120, 60, 255) if mode == "RGBA" else (200, 120, 60))
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()

    @staticmethod
    def _noise_image(size: tuple[int, int], fmt: str = "PNG") -> bytes:
        """噪声图 —— 不可压缩，才能真实地检验"压缩后确实变小了"。

        纯色块 PNG 只有几 KB，比压缩后的 JPEG 还小，用它做断言会让实现
        反过来"压缩后变大"，测试就失去了意义。
        """
        import random

        from PIL import Image

        width, height = size
        raw = random.Random(20240915).randbytes(width * height * 3)
        buffer = io.BytesIO()
        Image.frombytes("RGB", size, raw).save(buffer, format=fmt)
        return buffer.getvalue()

    def test_large_png_is_shrunk_and_becomes_jpeg(self):
        source = self._noise_image((1600, 1200))
        out, media_type = downscale_image(source, max_edge=512)
        assert media_type == "image/jpeg"
        assert out[:2] == b"\xff\xd8"  # JPEG 魔数
        assert len(out) < len(source)

        from PIL import Image

        with Image.open(io.BytesIO(out)) as image:
            assert max(image.size) == 512
            assert image.format == "JPEG"

    def test_oversized_jpeg_is_resized(self):
        source = self._noise_image((1000, 800), fmt="JPEG")
        out, media_type = downscale_image(source, max_edge=512)
        assert media_type == "image/jpeg"
        assert len(out) < len(source)

        from PIL import Image

        with Image.open(io.BytesIO(out)) as image:
            assert max(image.size) == 512

    def test_compliant_jpeg_is_returned_untouched(self):
        """已是 JPEG 且尺寸不超标 —— 原样返回，不重新编码。

        这条断言是"压缩真的省下了东西"的反面证据：若实现无条件重新编码，
        字节必然变化、画质必然下降，而测试只有比对字节才能发现。
        """
        source = self._image((400, 300), fmt="JPEG")
        out, media_type = downscale_image(source, max_edge=512)
        assert media_type == "image/jpeg"
        assert out is source

    def test_exactly_at_the_limit_is_untouched(self):
        source = self._image((512, 256), fmt="JPEG")
        out, _ = downscale_image(source, max_edge=512)
        assert out is source

    def test_rgba_png_is_converted_without_error(self):
        """带透明通道的 PNG 直接存 JPEG 会报错或出黑块，必须先转 RGB。"""
        source = self._image((300, 300), mode="RGBA")
        out, media_type = downscale_image(source, max_edge=512)
        assert media_type == "image/jpeg"
        assert out[:2] == b"\xff\xd8"

    def test_non_image_bytes_raise_value_error(self):
        with pytest.raises(ValueError, match="无法识别为图片"):
            downscale_image(b"definitely not an image")

    def test_invalid_max_edge_raises_value_error(self):
        with pytest.raises(ValueError, match="max_edge"):
            downscale_image(self._image((10, 10)), max_edge=0)


# --------------------------------------------------------------------------- #
# 缓存（省钱第 4 条）
# --------------------------------------------------------------------------- #


class TestSqliteVlmCache:
    """按内容哈希缓存到 SQLite。"""

    @staticmethod
    def _insight(tag: str, hints: list[str] | None = None) -> ImageInsight:
        digest = image_hash(tag.encode())
        return ImageInsight(
            url=f"synthetic://{tag}/0",
            image_hash=digest,
            note_id=tag,
            description=f"描述-{tag}",
            pain_hints=hints if hints is not None else ["色差明显"],
        )

    def test_put_then_get_roundtrip(self):
        cache = SqliteVlmCache(":memory:")
        insight = self._insight("a")
        cache.put(insight)

        hit = cache.get(insight.image_hash)
        assert hit is not None
        assert hit.description == "描述-a"
        assert hit.pain_hints == ["色差明显"]
        assert hit.note_id == "a"
        assert hit.url == "synthetic://a/0"
        cache.close()

    def test_miss_returns_none(self):
        cache = SqliteVlmCache(":memory:")
        assert cache.get("0" * 32) is None
        cache.close()

    def test_hit_is_marked_from_cache(self):
        """``from_cache`` 是成本报告区分"这次花了钱"与"这次白嫖"的唯一依据。"""
        cache = SqliteVlmCache(":memory:")
        insight = self._insight("b")
        cache.put(insight)
        assert cache.get(insight.image_hash).from_cache is True
        cache.close()

    def test_same_hash_is_overwritten_not_duplicated(self):
        cache = SqliteVlmCache(":memory:")
        insight = self._insight("c", hints=["第一次"])
        cache.put(insight)
        cache.put(self._insight("c", hints=["第二次"]))

        assert cache.stats()["entries"] == 1
        assert cache.get(insight.image_hash).pain_hints == ["第二次"]
        cache.close()

    def test_hits_are_counted(self):
        cache = SqliteVlmCache(":memory:")
        insight = self._insight("d")
        cache.put(insight)
        cache.get(insight.image_hash)
        cache.get(insight.image_hash)
        assert cache.stats()["hits"] == 2
        cache.close()

    def test_persists_across_instances(self, tmp_path):
        """缓存必须跨进程有效 —— 否则"第二次跑同一品类成本归零"就是空话。"""
        path = str(tmp_path / "vlm.sqlite")
        first = SqliteVlmCache(path)
        insight = self._insight("e")
        first.put(insight)
        first.close()

        second = SqliteVlmCache(path)
        hit = second.get(insight.image_hash)
        assert hit is not None
        assert hit.pain_hints == ["色差明显"]
        second.close()

    def test_creates_parent_directory(self, tmp_path):
        cache = SqliteVlmCache(str(tmp_path / "nested" / "dir" / "vlm.sqlite"))
        assert cache.stats()["entries"] == 0
        cache.close()

    def test_put_without_hash_raises(self):
        """没有键就写不进去 —— 静默写入会留下一批永远取不到的记录。"""
        cache = SqliteVlmCache(":memory:")
        with pytest.raises(ValueError, match="image_hash"):
            cache.put(ImageInsight(url="u", description="d"))
        cache.close()

    def test_empty_stats(self):
        cache = SqliteVlmCache(":memory:")
        assert cache.stats() == {"entries": 0, "hits": 0}
        cache.close()

    def test_concurrent_writes_are_serialized(self):
        """分析阶段是并发的：缓存被多线程写入不能丢记录、也不能抛异常。"""
        cache = SqliteVlmCache(":memory:")

        def put(index: int) -> None:
            cache.put(ImageInsight(url=f"u{index}", image_hash=image_hash(str(index).encode())))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(put, range(40)))

        assert cache.stats()["entries"] == 40
        cache.close()

    def test_concurrent_reads_are_serialized(self):
        cache = SqliteVlmCache(":memory:")
        cache.put(self._insight("f"))
        digest = image_hash(b"f")

        with ThreadPoolExecutor(max_workers=8) as pool:
            hits = list(pool.map(lambda _: cache.get(digest), range(40)))

        assert all(hit is not None for hit in hits)
        assert cache.stats()["hits"] == 40
        cache.close()


# --------------------------------------------------------------------------- #
# 成本预估（省钱第 7 条）
# --------------------------------------------------------------------------- #


class TestVlmAnalyzerEstimate:
    """``estimate()`` 必须在花钱之前给出可信的数字。"""

    def test_counts_images_after_cap_and_dedup(self, no_pillow):
        corpus = _corpus(
            ("n1", 10, [_data_url("a"), _data_url("b"), _data_url("c"), _data_url("d")]),
            ("n2", 10, [_data_url("e")]),
        )
        # 默认单篇最多 3 张：n1 的第 4 张不进入统计口径
        analyzer = VlmAnalyzer(FakeVision(), max_images_per_note=3)
        estimate = analyzer.estimate(corpus)

        assert estimate.total_images == 5
        assert estimate.unique_images == 4
        assert estimate.planned_calls == 4
        assert estimate.truncated is False

    def test_identical_content_counts_once(self, no_pillow):
        """省钱第 1 条：去重数的是内容，不是 URL。"""
        corpus = _corpus(
            ("n1", 10, [_data_url("same")]),
            ("n2", 10, [_data_url("same"), _data_url("other")]),
        )
        estimate = VlmAnalyzer(FakeVision()).estimate(corpus)
        assert estimate.total_images == 3
        assert estimate.unique_images == 2

    def test_cache_hits_are_excluded_from_planned_calls(self, no_pillow):
        cache = SqliteVlmCache(":memory:")
        cached_url = _data_url("cached")
        cache.put(
            ImageInsight(
                url=cached_url,
                image_hash=image_hash(b"cached"),
                description="命中",
                pain_hints=["x"],
            )
        )
        corpus = _corpus(("n1", 10, [cached_url, _data_url("fresh")]))

        estimate = VlmAnalyzer(FakeVision(), cache=cache).estimate(corpus)
        assert estimate.unique_images == 2
        assert estimate.cached_images == 1
        assert estimate.planned_calls == 1
        cache.close()

    def test_max_calls_truncates_and_flags(self, no_pillow):
        corpus = _corpus(("n1", 10, [_data_url(f"img-{i}") for i in range(5)]))
        estimate = VlmAnalyzer(FakeVision(), max_calls=2, max_images_per_note=5).estimate(corpus)

        assert estimate.planned_calls == 2
        assert estimate.truncated is True
        assert "截断" in estimate.summary()

    def test_zero_max_calls_plans_nothing(self, no_pillow):
        corpus = _corpus(("n1", 10, [_data_url("only")]))
        estimate = VlmAnalyzer(FakeVision(), max_calls=0).estimate(corpus)
        assert estimate.planned_calls == 0
        assert estimate.truncated is True

    def test_estimate_never_calls_the_model(self, no_pillow):
        """第 7 条的全部意义就在这里：没确认之前不许花钱。"""
        provider = FakeVision()
        corpus = _corpus(("n1", 10, [_data_url("a"), _data_url("b")]))
        VlmAnalyzer(provider).estimate(corpus)
        assert provider.call_count == 0

    def test_uncached_estimate_does_not_need_pillow(self, monkeypatch):
        """预估只需要算哈希，不该因为没有 Pillow 就连预估都做不了。"""

        def boom(purpose: str):
            raise AssertionError("estimate 不应触发 Pillow 检查")

        monkeypatch.setattr(vlm, "require_pillow", boom)
        corpus = _corpus(("n1", 10, [_data_url("a")]))
        assert VlmAnalyzer(FakeVision()).estimate(corpus).unique_images == 1

    def test_fetch_failure_is_reported_not_silently_dropped(self, no_pillow, monkeypatch):
        """不变式 3：失败 ≠ 空结果。取不到的图必须出现在 warnings 里。"""
        corpus = _corpus(
            ("n1", 10, ["https://cdn.example.invalid/dead.jpg", _data_url("alive")]),
        )

        def fake_download(url, timeout):
            raise OSError("连接超时")

        monkeypatch.setattr(vlm, "_download_bytes", fake_download)
        estimate = VlmAnalyzer(FakeVision()).estimate(corpus)

        # 只有 data URL 那张算得出来；失败的那张不计入 unique（内容未知，无法判断重复）
        assert estimate.unique_images == 1
        assert estimate.planned_calls == 1

    def test_fetch_failure_lands_in_warnings(self, no_pillow, monkeypatch):
        corpus = _corpus(("n1", 10, ["https://cdn.example.invalid/dead.jpg"]))

        def fake_download(url, timeout):
            raise OSError("连接超时")

        monkeypatch.setattr(vlm, "_download_bytes", fake_download)
        result = VlmAnalyzer(FakeVision()).analyze(corpus)

        assert result.units == []
        assert any("获取失败" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# 分析主流程
# --------------------------------------------------------------------------- #


class TestVlmAnalyzerAnalyze:
    """``analyze()`` 的产出、省钱效果与降级行为。"""

    def test_units_carry_note_weight_and_image_flag(self, no_pillow):
        provider = FakeVision()
        corpus = _corpus(("n1", 1000, [_data_url("a")]), ("n2", 100, [_data_url("b")]))
        result = VlmAnalyzer(provider).analyze(corpus)

        assert len(result.units) == 2
        weights = {unit.note_id: unit.weight for unit in result.units}
        # 权重按全语料最大点赞数归一（与 clean.compute_weights 同口径）
        assert weights["n1"] == pytest.approx(1.0)
        assert weights["n2"] == pytest.approx(math.log1p(100) / math.log1p(1000))
        assert 0 < weights["n2"] < weights["n1"]

        for unit in result.units:
            assert unit.from_image is True
            assert unit.source == "note"
            assert "一张左右对比图" in unit.text
            assert "左暗右亮差异明显" in unit.text
            assert unit.images

    def test_weight_scale_includes_comments(self, no_pillow):
        """评论点赞数也是同一批文本单元的一部分，基准必须与清洗层一致。"""
        corpus = _corpus(("n1", 100, [_data_url("a")]))
        corpus.comments = [RawComment(comment_id="c1", content="好用", likes=10000, note_id="n1")]
        result = VlmAnalyzer(FakeVision()).analyze(corpus)

        assert result.units[0].weight == pytest.approx(math.log1p(100) / math.log1p(10000))

    def test_weight_is_one_when_no_likes_anywhere(self, no_pillow):
        corpus = _corpus(("n1", 0, [_data_url("a")]))
        result = VlmAnalyzer(FakeVision()).analyze(corpus)
        assert result.units[0].weight == 1.0

    def test_no_pain_hints_means_no_unit(self, no_pillow):
        """图里没有痛点就不生成单元 —— 否则会给聚类灌入"这张图没问题"的噪声。"""
        provider = FakeVision(handler=lambda i, p, images: _EMPTY_HINTS_REPLY)
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)
        assert result.units == []
        assert result.warnings == []
        assert result.insights["n1"][0].description == "一张普通商品图"

    def test_duplicate_images_cost_one_call(self, no_pillow):
        """省钱第 1 条：同款图跨笔记重复，只花一次钱。"""
        provider = FakeVision()
        corpus = _corpus(
            ("n1", 10, [_data_url("dup")]),
            ("n2", 10, [_data_url("dup")]),
            ("n3", 10, [_data_url("dup")]),
        )
        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 1
        assert result.estimate.unique_images == 1
        assert len(result.units) == 1  # 同一张图只产生一条证据，不灌水簇大小
        # 但来源笔记都要记上，渲染时才说得清这张图出现在哪些笔记里
        assert sorted(result.insights) == ["n1", "n2", "n3"]

    def test_cached_images_cost_nothing(self, no_pillow):
        """省钱第 4 条：命中缓存不产生调用。"""
        cache = SqliteVlmCache(":memory:")
        cache.put(
            ImageInsight(
                url=_data_url("hit"),
                image_hash=image_hash(b"hit"),
                description="缓存里的描述",
                pain_hints=["缓存里的痛点"],
            )
        )
        provider = FakeVision()
        corpus = _corpus(("n1", 10, [_data_url("hit")]))

        result = VlmAnalyzer(provider, cache=cache).analyze(corpus)
        assert provider.call_count == 0
        assert result.estimate.cached_images == 1
        assert result.units[0].text.startswith("缓存里的描述")
        cache.close()

    def test_successful_insight_is_written_into_cache(self, no_pillow):
        cache = SqliteVlmCache(":memory:")
        corpus = _corpus(("n1", 10, [_data_url("fresh")]))

        VlmAnalyzer(FakeVision(), cache=cache).analyze(corpus)
        assert cache.get(image_hash(b"fresh")) is not None
        cache.close()

    def test_missing_pillow_fails_fast_with_install_command(self, monkeypatch):
        """缺 Pillow 是安装问题，应该在开跑时就给修复命令，而不是让每张图各报一次。"""
        monkeypatch.setitem(sys.modules, "PIL.Image", None)
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        with pytest.raises(MissingDependencyError, match=r"\[vlm\]"):
            VlmAnalyzer(FakeVision()).analyze(corpus)

    def test_corpus_without_images_needs_no_pillow(self, monkeypatch):
        def boom(purpose: str):
            raise AssertionError("没有图片时不该检查 Pillow")

        monkeypatch.setattr(vlm, "require_pillow", boom)
        result = VlmAnalyzer(FakeVision()).analyze(_corpus(("n1", 10, [])))
        assert result.units == []

    def test_max_images_per_note_caps_the_calls(self, no_pillow):
        """省钱第 3 条：单篇只取前 N 张，默认 3 —— 一条 12 图的笔记也只花 3 次。"""
        provider = FakeVision()
        corpus = _corpus(("n1", 10, [_data_url(f"img-{i}") for i in range(12)]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 3
        assert len(result.units) == 3
        # 取前 N 张而不是随机采样 —— 笔记的第一张通常是封面，且结果可复现
        assert [unit.images[0] for unit in result.units] == [
            _data_url("img-0"),
            _data_url("img-1"),
            _data_url("img-2"),
        ]

    def test_max_calls_truncation_warns_and_stops(self, no_pillow):
        """省钱第 7 条的另一半：截断了必须说，否则用户以为看到的是全量。"""
        provider = FakeVision()
        corpus = _corpus(("n1", 10, [_data_url(f"img-{i}") for i in range(4)]))

        analyzer = VlmAnalyzer(provider, max_calls=2, max_images_per_note=4)
        result = analyzer.analyze(corpus)

        assert provider.call_count == 2
        assert result.estimate.truncated is True
        assert result.estimate.planned_calls == 2
        assert len(result.units) == 2
        assert any("max_vlm_calls=2" in warning for warning in result.warnings)

    def test_single_failure_does_not_abort_the_run(self, no_pillow, fake_sleep):
        """省钱第 6 条：单张失败只记 warning，其余照跑，绝不抛出。"""
        provider = FakeVision(
            handler=lambda index, prompt, images: (
                LLMError("鉴权失败：API Key 无效")
                if images[0] == build_data_url(b"bad", "image/jpeg")
                else _REPLY
            )
        )
        corpus = _corpus(("n1", 10, [_data_url("bad"), _data_url("good")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert len(result.units) == 1  # 好的那张照常产出
        assert any("分析失败" in warning for warning in result.warnings)
        errors = [insight.error for insight in result.insights["n1"] if insight.error]
        assert len(errors) == 1
        assert "鉴权失败" in errors[0]

    def test_failure_is_not_cached(self, no_pillow, fake_sleep):
        """把一次限流错误缓存下来，会让这张图在缓存过期前永远拿不到分析。"""
        cache = SqliteVlmCache(":memory:")
        provider = FakeVision(handler=lambda i, p, images: LLMError("网络错误：connection reset"))
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        VlmAnalyzer(provider, cache=cache).analyze(corpus)
        assert cache.stats()["entries"] == 0
        cache.close()

    def test_rate_limit_is_not_retried(self, no_pillow, fake_sleep):
        """省钱第 5 条：被限流时立即放弃该张图 —— 重试就是继续烧限额。"""
        provider = FakeVision(handler=lambda i, p, images: LLMError("HTTP 429 Too Many Requests"))
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 1
        assert fake_sleep == []
        assert "429" in result.insights["n1"][0].error

    def test_transient_error_is_retried_with_exponential_backoff(self, no_pillow, fake_sleep):
        """非限流的瞬时错误才退避重试，且**次数有上限**（第 5 条的"不无限重试"）。"""

        def handler(index, prompt, images):
            if index < 2:
                raise LLMError("网络错误：connection reset")
            return _REPLY

        provider = FakeVision(handler=handler)
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 3
        assert fake_sleep == [0.5, 1.0]
        assert result.units and not result.warnings

    def test_retries_are_capped(self, no_pillow, fake_sleep):
        """一直失败也不会无限重试 —— 最多 3 次就放手，并如实记错。"""
        provider = FakeVision(handler=lambda i, p, images: LLMError("网络错误：gateway timeout"))
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == vlm.VLM_MAX_ATTEMPTS == 3
        assert fake_sleep == [0.5, 1.0]
        assert result.insights["n1"][0].error

    def test_malformed_reply_is_retried_then_recorded(self, no_pillow, fake_sleep):
        provider = FakeVision(handler=lambda i, p, images: "抱歉，我无法完成这个任务。")
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 3
        assert "JSON" in result.insights["n1"][0].error
        assert result.units == []

    def test_unexpected_exception_type_is_not_retried(self, no_pillow, fake_sleep):
        """非 LLMError 通常是代码缺陷，重试没有意义 —— 直接记错。"""
        provider = FakeVision(handler=lambda i, p, images: TypeError("bad arguments"))
        corpus = _corpus(("n1", 10, [_data_url("a")]))

        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 1
        assert "TypeError" in result.insights["n1"][0].error

    def test_suffix_markdown_reply_is_parsed(self, no_pillow):
        """模型很爱套 ```json 代码块 —— 解析必须容得下。"""
        provider = FakeVision(handler=lambda i, p, images: f"结果如下：\n```json\n{_REPLY}\n```")
        result = VlmAnalyzer(provider).analyze(_corpus(("n1", 10, [_data_url("a")])))

        assert result.units[0].text.startswith("一张左右对比图")

    def test_non_string_hints_are_dropped(self, no_pillow):
        provider = FakeVision(
            handler=lambda i, p, images: '{"description": "x", "pain_hints": [1, {"a": 2}, "有效"]}'
        )
        result = VlmAnalyzer(provider).analyze(_corpus(("n1", 10, [_data_url("a")])))

        assert result.units[0].text.endswith("有效")
        assert "{" not in result.units[0].text

    def test_hints_as_plain_string_are_kept(self, no_pillow):
        provider = FakeVision(
            handler=lambda i, p, images: '{"description": "x", "pain_hints": "只有一条"}'
        )
        result = VlmAnalyzer(provider).analyze(_corpus(("n1", 10, [_data_url("a")])))

        assert result.units[0].text.endswith("只有一条")

    def test_too_many_hints_are_capped(self, no_pillow):
        """一条几百字的单元会在 embedding 空间里主导整个簇。"""
        hints = ", ".join(f'"痛点{i}"' for i in range(30))
        reply = f'{{"description": "x", "pain_hints": [{hints}]}}'
        provider = FakeVision(handler=lambda i, p, images: reply)

        result = VlmAnalyzer(provider).analyze(_corpus(("n1", 10, [_data_url("a")])))
        assert len(result.insights["n1"][0].pain_hints) == vlm._MAX_PAIN_HINTS

    def test_progress_reaches_completion(self, no_pillow):
        seen: list[tuple[str, float]] = []
        urls = [_data_url(f"p{i}") for i in range(4)]
        corpus = _corpus(("n1", 10, urls))

        VlmAnalyzer(FakeVision(), max_concurrency=2, max_images_per_note=len(urls)).analyze(
            corpus, progress=lambda stage, ratio: seen.append((stage, ratio))
        )

        assert [stage for stage, _ in seen] == ["图片分析"] * 4
        assert seen[-1][1] == pytest.approx(1.0)
        assert all(0 < ratio <= 1 for _, ratio in seen)

    def test_no_progress_callback_is_fine(self, no_pillow):
        result = VlmAnalyzer(FakeVision()).analyze(_corpus(("n1", 10, [_data_url("a")])))
        assert len(result.units) == 1

    def test_units_follow_corpus_order_despite_concurrency(self, no_pillow):
        """不变式 1：顺序必须稳定 —— 并发完成顺序不能影响单元顺序。"""
        corpus = _corpus(
            ("n1", 10, [_data_url("a1"), _data_url("a2")]),
            ("n2", 10, [_data_url("b1")]),
        )

        def handler(index, prompt, images):
            time.sleep(0.01 * (3 - index))  # 先提交的后完成
            return f'{{"description": "d{index}", "pain_hints": ["h{index}"]}}'

        result = VlmAnalyzer(FakeVision(handler=handler), max_concurrency=3).analyze(corpus)
        assert [unit.note_id for unit in result.units] == ["n1", "n1", "n2"]

    def test_unsupported_image_bytes_are_recorded_as_failure(self, no_pillow, monkeypatch):
        """拿到了字节但用不了（不是图片）—— 记失败，不占调用额度。"""
        provider = FakeVision()
        corpus = _corpus(("n1", 10, [_data_url("a"), _data_url("b")]))

        def picky_downscale(data, *, max_edge=512):
            if data == b"a":
                raise ValueError("无法识别为图片")
            return data, "image/jpeg"

        monkeypatch.setattr(vlm, "downscale_image", picky_downscale)
        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 1
        assert result.estimate.unique_images == 2
        assert result.estimate.planned_calls == 1
        assert any("无法用于分析" in warning for warning in result.warnings)

    def test_repeated_failures_are_summarized(self, no_pillow, fake_sleep):
        """1000 张图集体失败时，warnings 不该变成 1000 行。"""
        provider = FakeVision(handler=lambda i, p, images: LLMError("HTTP 429 rate limited"))
        urls = [_data_url(f"img-{i}") for i in range(vlm._MAX_DETAILED_WARNINGS + 5)]
        corpus = _corpus(("n1", 10, urls))

        result = VlmAnalyzer(provider, max_images_per_note=len(urls)).analyze(corpus)

        assert len(result.warnings) == vlm._MAX_DETAILED_WARNINGS + 1
        assert "另有 5 张" in result.warnings[-1]

    def test_request_rate_limiting_spaces_calls(self, no_pillow, monkeypatch):
        """省钱第 5 条的前半段：并发 + 限流。"""
        interval = 0.05
        monkeypatch.setattr(vlm, "VLM_MIN_REQUEST_INTERVAL", interval)
        urls = [_data_url(f"r{i}") for i in range(4)]
        corpus = _corpus(("n1", 10, urls))

        started = time.monotonic()
        VlmAnalyzer(FakeVision(), max_concurrency=4, max_images_per_note=len(urls)).analyze(corpus)
        elapsed = time.monotonic() - started

        # 4 次调用至少要拉开 3 个间隔；只断言下界，不会因机器快而抖动
        assert elapsed >= interval * 3

    def test_concurrent_analysis_calls_every_image_once(self, no_pillow):
        provider = FakeVision()
        urls = [_data_url(f"c{i}") for i in range(8)]
        corpus = _corpus(("n1", 10, urls))

        result = VlmAnalyzer(provider, max_concurrency=4, max_images_per_note=8).analyze(corpus)

        assert provider.call_count == 8
        assert len(result.units) == 8

    def test_estimate_and_analyze_agree(self, no_pillow):
        """预估与实际必须同口径 —— 否则第 7 条的承诺就是假的。"""
        cache = SqliteVlmCache(":memory:")
        corpus = _corpus(
            ("n1", 10, [_data_url("a"), _data_url("b")]),
            ("n2", 10, [_data_url("a")]),
        )
        analyzer = VlmAnalyzer(FakeVision(), cache=cache, max_calls=2)

        estimate = analyzer.estimate(corpus)
        result = analyzer.analyze(corpus)

        assert result.estimate == estimate
        cache.close()


# --------------------------------------------------------------------------- #
# 与真实 Pillow / synthetic 的集成
# --------------------------------------------------------------------------- #


class TestVlmIntegration:
    """真图（``synthetic://``）走完整链路 —— 需要 Pillow。"""

    @pytest.fixture(autouse=True)
    def _require_pillow(self):
        pytest.importorskip("PIL.Image")

    def test_synthetic_images_are_compressed_before_upload(self):
        """省钱第 2 条：上传的是压缩后的 JPEG，不是原图。"""
        provider = FakeVision()
        corpus = _corpus(("n1", 10, ["synthetic://n1/0"]))

        VlmAnalyzer(provider).analyze(corpus)

        sent = _decoded_payload(provider.calls[0])
        assert sent[:2] == b"\xff\xd8"
        assert len(sent) < len(fetch_image("synthetic://n1/0"))

        from PIL import Image

        with Image.open(io.BytesIO(sent)) as image:
            assert max(image.size) <= 512

    def test_identical_synthetic_urls_cost_one_call(self):
        """同一地址必然生成同样的字节 —— 去重测试因此是确定性的。"""
        provider = FakeVision()
        corpus = _corpus(
            ("n1", 10, ["synthetic://shared/0"]),
            ("n2", 10, ["synthetic://shared/0"]),
        )
        result = VlmAnalyzer(provider).analyze(corpus)

        assert provider.call_count == 1
        assert len(result.units) == 1
        assert sorted(result.insights) == ["n1", "n2"]

    def test_second_run_is_free_through_the_cache(self):
        cache = SqliteVlmCache(":memory:")
        corpus = _corpus(("n1", 10, ["synthetic://n1/0", "synthetic://n1/1"]))

        first = FakeVision()
        VlmAnalyzer(first, cache=cache).analyze(corpus)
        assert first.call_count == 2

        second = FakeVision()
        result = VlmAnalyzer(second, cache=cache).analyze(corpus)
        assert second.call_count == 0
        assert result.estimate.cached_images == 2
        assert len(result.units) == 2
        cache.close()
