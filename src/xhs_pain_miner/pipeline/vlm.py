"""VLM 图片分析（``--deep``）。

这是整条流水线里**最贵、最慢**的一环：1000 张图的调用成本可达 ¥15-100、耗时
10-30 分钟，是纯文本的 5-10 倍。因此本模块的绝大部分复杂度都在**省钱**上：

省钱七条（缺一条都会让成本失控）
--------------------------------
1. **URL 去重** —— 同款商品图跨笔记大量重复，实测去重率常在 40% 以上。
2. **长边压缩到 512px** —— 直接决定图片 token 数量。
3. **单篇最多取前 N 张** —— ``Settings.vlm_max_images_per_note``，默认 3。
4. **结果按图片内容哈希缓存到 SQLite** —— 第二次跑同一品类时成本近乎归零。
5. **并发 + 限流 + 指数退避** —— 但**不无限重试**，被限流时立即放弃该张图。
6. **失败降级** —— 单张图失败不阻塞主流程，但必须在产物上如实标注。
7. **开跑前打印成本预估** —— :class:`VlmEstimate`，需用户确认后才真正花钱。

价值边界（不要夸大）
--------------------
VLM 的产出会作为额外的 :class:`~xhs_pain_miner.models.TextUnit` 进入聚类，
因此**视觉痛点可以形成独立的簇**（例如"宣传图色差"这类只有看图才知道的问题）。
代价是流水线从「clean → embed」变成「clean → vlm → embed」，VLM 成为串行瓶颈。

实现上的两条取舍（改代码前先读）
--------------------------------
* **去重按内容，不按 URL。** 一个 URL 可能返回不同内容（CDN 变体），不同 URL
  也可能是同一张图（同款图重发）。因此去重必须先拿到字节 —— 这是
  :meth:`VlmAnalyzer.estimate` 也要下载图片的原因：它省的是 VLM 调用，
  不是带宽。真正的带宽节省来自"每张图只下载一次"：计划阶段下完字节就地压缩，
  待调用队列直接拿着压缩后的字节去调用。
* **同一张图跨笔记重复时，只生成一条 TextUnit。** 调用只花一次，但
  :attr:`VlmResult.insights` 仍会为每篇用到它的笔记记录来源。若同一张图在
  10 篇笔记里重复出现就产生 10 条内容完全相同的单元，簇的 ``size`` 会被灌水 ——
  而 ``size`` 是「提及次数」的唯一事实来源，宁可少算也不能造假。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from xhs_pain_miner.llm.base import (
    LLMError,
    LLMProvider,
    build_data_url,
    extract_json,
    split_data_url,
)
from xhs_pain_miner.models import ImageInsight, RawCorpus, TextUnit, VlmEstimate
from xhs_pain_miner.synthetic import (
    SYNTHETIC_SCHEME,
    parse_synthetic_url,
    require_pillow,
    synthesize,
)

IMAGE_FETCH_TIMEOUT = 15.0
"""单张图片的下载超时（秒）。"""

IMAGE_MAX_BYTES = 8 * 1024 * 1024
"""单张图片的大小上限（8MB）。超过则放弃 —— 这是防御性的，不是容量规划。"""

VLM_MAX_ATTEMPTS = 3
"""单张图最多尝试几次（含首次）。**不允许无限重试** —— 被限流时更要立刻放手。"""

VLM_RETRY_BACKOFF_BASE = 0.5
"""指数退避基数（秒）：第 n 次重试前等 ``BASE * 2**(n-1)``。"""

VLM_RETRY_BACKOFF_MAX = 4.0
"""单次退避的上限（秒）。"""

VLM_MIN_REQUEST_INTERVAL = 0.0
"""两次 VLM 调用之间的最小间隔（秒）。0 表示不额外限流。

默认 0 是刻意的：多数供应商自己会返回 429，本地再限流只会拖长耗时。当用户
用的是严格限额的服务（或本地模型）时，把它调大即可获得全局串行节流。
"""

VISION_PROMPT = """这是一张来自小红书笔记的图片。请分析：

1. description：客观描述图片内容（是什么产品、什么场景、有没有对比或标注文字）。
2. pain_hints：图片中体现出的、**用户可能不满意的点**。只提取你能从图里直接看到的，
   例如：宣传图与实际效果的差异、包装上的使用说明混乱、色号与描述不符、
   对比图中暴露的缺陷。**没有就返回空数组，不要为了凑数而编造。**

只输出 JSON：{"description": "...", "pain_hints": ["...", "..."]}
"""

_HASH_CHARS = 32
"""缓存键取 sha256 的前 32 个十六进制字符（128 bit）。

在这个量级（万张图）碰撞概率可以忽略，而短键让缓存表、日志与告警都可读。
"""

_MAX_DESCRIPTION_CHARS = 300
_MAX_PAIN_HINTS = 8
"""单张图进入聚类的文本上限。

模型偶尔会把整段 OCR 结果倒出来，一条几百字的单元会在 embedding 空间里
主导整个簇。截断会丢信息，但比让一张图代表一个簇要好。
"""

_MAX_DETAILED_WARNINGS = 10
"""逐条列出的失败上限。超过后折叠成一行 —— 1000 张图集体失败时，
产物里的 ``notes`` 不该被 1000 行同样的错误淹没。"""

_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "限流",
    "请求过于频繁",
)

_HEADERS = {"User-Agent": "xhs-pain-miner/0.1 (image fetch)"}

_VlmStatus = Literal["cached", "pending", "skipped", "failed"]
"""计划中每张图的状态：

* ``cached`` —— 命中本地缓存，不产生调用。
* ``pending`` —— 需要发起调用。
* ``skipped`` —— 超出 ``max_calls``，本次不分析（必须告知用户）。
* ``failed`` —— 字节拿到了但无法用于调用（如不是图片）。
"""


@runtime_checkable
class VlmCache(Protocol):
    """VLM 结果缓存。"""

    def get(self, image_hash: str) -> ImageInsight | None:
        """按图片内容哈希取缓存。未命中返回 ``None``。"""
        ...

    def put(self, insight: ImageInsight) -> None:
        """写入缓存。"""
        ...


@dataclass(slots=True)
class VlmResult:
    """一次 VLM 分析的全部产出。"""

    units: list[TextUnit] = field(default_factory=list)
    """图片派生的文本单元，与文本单元一起进入聚类。"""

    insights: dict[str, list[ImageInsight]] = field(default_factory=dict)
    """``note_id`` → 该笔记图片的分析结果，供渲染时展示图片来源。"""

    estimate: VlmEstimate = field(default_factory=VlmEstimate)
    warnings: list[str] = field(default_factory=list)
    """降级与失败信息。**必须透传到** :attr:`MiningResult.notes`。"""


def _sleep(seconds: float) -> None:
    """休眠 —— 抽成函数是为了让测试能替换掉它（退避测试不该真的等 7.5 秒）。"""
    if seconds > 0:
        time.sleep(seconds)


class _RateLimiter:
    """跨线程的最小调用间隔限流器。

    锁住整个"等待 + 预约下一个时隙"的过程，因此并发数再大也保证全局间隔 ——
    若只在取时间戳时加锁，多个线程会同时读到同一个"下一个可用时刻"，
    限流等于没做。
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(min_interval, 0.0)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        """等待到允许发起下一次调用的时刻。"""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                _sleep(wait)
            self._next_at = max(time.monotonic(), self._next_at) + self._min_interval


def _is_rate_limited(exc: BaseException) -> bool:
    """判断异常是否是「被限流」。

    三家协议的限流异常类型各不相同，``LLMProvider`` 只保证抛 :class:`LLMError`，
    因此只能按消息判断。宁可把普通错误误判成限流（少重试几次，仍然会记进
    warnings），也不要把限流当普通错误重试 —— 那正是把限额烧光的方式。
    """
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def image_hash(data: bytes) -> str:
    """按图片内容算哈希，用于去重与缓存键。

    用内容哈希而不是 URL：同一个 URL 可能返回不同内容（CDN 变体），
    不同 URL 也可能是同一张图（同款商品图重发）。内容哈希才能同时覆盖两种情况。
    """
    return hashlib.sha256(data).hexdigest()[:_HASH_CHARS]


def _download_bytes(url: str, timeout: float) -> bytes:
    """下载图片并累计字节数，超过 :data:`IMAGE_MAX_BYTES` 立即中断。

    抽成独立函数有两个好处：流式累计可以**在超限时断开连接**（而不是下完
    8MB 再判断），并且测试可以替换它 —— 图片链路的测试一律不联网。
    """
    import httpx

    chunks: list[bytes] = []
    total = 0
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url, headers=_HEADERS) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > IMAGE_MAX_BYTES:
                        limit_mb = IMAGE_MAX_BYTES // (1024 * 1024)
                        raise OSError(f"图片超过大小上限 {limit_mb}MB，已中断下载: {url}")
                    chunks.append(chunk)
    except httpx.HTTPStatusError as exc:
        raise OSError(f"图片下载失败 HTTP {exc.response.status_code}: {url}") from exc
    except httpx.HTTPError as exc:
        raise OSError(f"图片下载失败: {url}（{exc}）") from exc
    return b"".join(chunks)


def fetch_image(url: str, *, timeout: float = IMAGE_FETCH_TIMEOUT) -> bytes:
    """把图片地址解析成字节。

    支持的来源：

    * ``http(s)://`` —— 下载，受 ``timeout`` 与 :data:`IMAGE_MAX_BYTES` 约束。
    * ``file://`` 或本地路径 —— 读文件（便于离线测试与自定义语料）。
    * ``data:image/...;base64,...`` —— 直接解码。
    * ``synthetic://<note_id>/<index>`` —— **由**
      :mod:`xhs_pain_miner.synthetic` **生成的确定性模拟图片**，仅用于内置
      fixture 与测试。生产语料不会出现这个 scheme。

    Args:
        url: 图片地址。
        timeout: 下载超时。

    Returns:
        图片字节。

    Raises:
        ValueError: 地址格式不合法。
        OSError: 下载或读取失败。
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"图片地址不能为空: {url!r}")
    candidate = url.strip()

    if candidate.startswith("data:"):
        return _decode_data_url(candidate)

    if candidate.startswith(("http://", "https://")):
        data = _download_bytes(candidate, timeout)
        if not data:
            raise OSError(f"下载到的图片为空: {candidate}")
        # 流式下载已经拦过一次；这里再拦一次是因为 ``_download_bytes`` 可能被
        # 替换（测试）或未来换成别的实现 —— 上限必须由本函数自己兜底。
        if len(data) > IMAGE_MAX_BYTES:
            raise OSError(f"图片超过大小上限 {IMAGE_MAX_BYTES // (1024 * 1024)}MB: {candidate}")
        return data

    if candidate.startswith(SYNTHETIC_SCHEME):
        if parse_synthetic_url(candidate) is None:
            raise ValueError(
                f"非法的合成图地址: {candidate!r}（应形如 {SYNTHETIC_SCHEME}<note_id>/<index>）"
            )
        return synthesize(candidate)

    if candidate.startswith("file://"):
        path = Path(candidate[len("file://") :])
        if not path.is_file():
            raise OSError(f"图片文件不存在: {path}")
        return path.read_bytes()

    if "://" in candidate:
        scheme = candidate.split("://", 1)[0]
        raise ValueError(f"不支持的图片地址协议: {scheme}://")

    # 其余一律按本地路径处理 —— 自定义语料里写相对路径是最省事的用法。
    path = Path(candidate)
    if not path.is_file():
        raise OSError(f"图片文件不存在: {path}")
    return path.read_bytes()


def _decode_data_url(url: str) -> bytes:
    """解码 ``data:image/...;base64,...``。"""
    import base64
    import binascii

    media_type, payload = split_data_url(url)
    if not media_type.startswith("image/"):
        raise ValueError(f"Data URL 不是图片类型: {media_type}")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Data URL 的 base64 内容无法解码: {exc}") from exc


def downscale_image(data: bytes, *, max_edge: int = 512) -> tuple[bytes, str]:
    """把图片压缩到长边不超过 ``max_edge``，并转成统一的 JPEG。

    统一转 JPEG 有两个原因：压缩率最高；三种 LLM 协议对 JPEG 的支持最一致
    （PNG 透明通道在部分服务上会被拒绝）。

    已经是 JPEG 且尺寸不超标时**原样返回**，避免无意义的重新编码损失画质 ——
    这也让"压缩是否真的省钱"这件事可以被测试断言。

    Args:
        data: 原始图片字节。
        max_edge: 长边上限。

    Returns:
        ``(压缩后的字节, media_type)``。

    Raises:
        xhs_pain_miner.pipeline.deps.MissingDependencyError: 缺少 Pillow。
        ValueError: 无法识别为图片。
    """
    if max_edge < 1:
        raise ValueError(f"max_edge 必须为正整数，收到 {max_edge}")

    pil_image, _, _ = require_pillow("VLM 图片压缩")
    try:
        image = pil_image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # Pillow 的异常类型很多（UnidentifiedImageError / OSError / ...）
        raise ValueError(f"无法识别为图片: {exc}") from exc

    with image:
        width, height = image.size
        if image.format == "JPEG" and max(width, height) <= max_edge:
            return data, "image/jpeg"

        # RGBA / P / LA 直接存 JPEG 会报错或出黑块，必须先落到 RGB。
        rgb = image.convert("RGB")
        longest = max(rgb.size)
        if longest > max_edge:
            scale = max_edge / longest
            target = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(target, pil_image.LANCZOS)

        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue(), "image/jpeg"


class SqliteVlmCache:
    """把 VLM 结果缓存到 SQLite。

    放在 ``Settings.db_path`` 的同目录下。缓存键是图片内容哈希，
    因此换一个品类重新跑时，重复出现的商品图不会二次计费。

    Args:
        path: SQLite 文件路径。``":memory:"`` 表示内存库（测试用）。
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS vlm_cache (
        image_hash  TEXT PRIMARY KEY,
        url         TEXT NOT NULL DEFAULT '',
        note_id     TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        pain_hints  TEXT NOT NULL DEFAULT '[]',
        created_at  TEXT NOT NULL DEFAULT '',
        hits        INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + 显式锁：分析阶段是并发的，缓存必须能被工作线程
        # 读写；只用其中任何一个都会在并发下出错（抛异常 / 数据竞争）。
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(self._SCHEMA)
            self._conn.commit()

    def get(self, image_hash: str) -> ImageInsight | None:
        """按哈希取缓存。

        命中时把 ``hits`` 计数加一 —— 成本报告要能回答"缓存帮我省了多少次调用"，
        而这个数字跨运行才有意义，所以写进库里而不是内存。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT url, note_id, description, pain_hints FROM vlm_cache WHERE image_hash = ?",
                (image_hash,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE vlm_cache SET hits = hits + 1 WHERE image_hash = ?", (image_hash,)
            )
            self._conn.commit()

        url, note_id, description, pain_hints = row
        try:
            hints = json.loads(pain_hints)
        except json.JSONDecodeError:
            hints = []
        return ImageInsight(
            url=url,
            image_hash=image_hash,
            note_id=note_id,
            description=description,
            pain_hints=[str(h) for h in hints] if isinstance(hints, list) else [],
            from_cache=True,
        )

    def put(self, insight: ImageInsight) -> None:
        """写入缓存（同哈希覆盖）。

        Raises:
            ValueError: ``insight.image_hash`` 为空 —— 没有键就没法缓存，
                静默写入会产生一批永远取不到的记录。
        """
        if not insight.image_hash:
            raise ValueError("写入 VLM 缓存前必须设置 image_hash")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO vlm_cache "
                "(image_hash, url, note_id, description, pain_hints, created_at, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "  COALESCE((SELECT hits FROM vlm_cache WHERE image_hash = ?), 0))",
                (
                    insight.image_hash,
                    insight.url,
                    insight.note_id,
                    insight.description,
                    json.dumps(insight.pain_hints, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    insight.image_hash,
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        """返回缓存统计（条数等），供成本报告使用。"""
        with self._lock:
            entries = self._conn.execute("SELECT COUNT(*) FROM vlm_cache").fetchone()[0]
            hits = self._conn.execute("SELECT COALESCE(SUM(hits), 0) FROM vlm_cache").fetchone()[0]
        return {"entries": int(entries), "hits": int(hits)}

    def close(self) -> None:
        """关闭连接。"""
        with self._lock:
            self._conn.close()


@dataclass(slots=True)
class _PlannedImage:
    """一张（去重后的）待处理图片。"""

    url: str
    note_id: str
    image_hash: str
    likes: int
    status: _VlmStatus
    publish_time: datetime | None = None
    media_type: str = "image/jpeg"
    payload: bytes = b""
    """压缩后的字节。只有 ``pending`` 的图片会持有它。"""
    duplicate_note_ids: list[str] = field(default_factory=list)
    """同样用到这张图的其它笔记（内容完全相同），用于把来源记进 insights。"""
    insight: ImageInsight | None = None


@dataclass(slots=True)
class _Plan:
    """一次运行的计划 —— ``estimate()`` 与 ``analyze()`` 共用同一份统计口径。"""

    total_images: int
    images: list[_PlannedImage]
    skipped_by_limit: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def cached(self) -> int:
        """命中缓存的图片数。"""
        return sum(1 for item in self.images if item.status == "cached")

    @property
    def planned_calls(self) -> int:
        """将要真正发起的调用数。"""
        return sum(1 for item in self.images if item.status == "pending")


class VlmAnalyzer:
    """图片分析器。

    Args:
        provider: 支持视觉的 LLM 供应商。
        *,
        cache: 缓存实现。``None`` 表示不缓存（会在成本报告里体现为全量调用）。
        max_edge: 压缩后的长边。
        max_images_per_note: 单篇笔记最多分析多少张图。
        max_calls: 本次运行的调用上限。``None`` 表示不设上限。
        max_concurrency: 并发数。
        temperature: 采样温度。分析任务建议低温。
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        cache: VlmCache | None = None,
        max_edge: int = 512,
        max_images_per_note: int = 3,
        max_calls: int | None = None,
        max_concurrency: int = 2,
        temperature: float = 0.2,
    ) -> None:
        if max_edge < 1:
            raise ValueError(f"max_edge 必须为正整数，收到 {max_edge}")
        if max_images_per_note < 0:
            raise ValueError(f"max_images_per_note 不能为负，收到 {max_images_per_note}")
        if max_calls is not None and max_calls < 0:
            raise ValueError(f"max_calls 不能为负，收到 {max_calls}")

        self.provider = provider
        self.cache = cache
        self.max_edge = max_edge
        self.max_images_per_note = max_images_per_note
        self.max_calls = max_calls
        self.max_concurrency = max(max_concurrency, 1)
        self.temperature = temperature

    # ------------------------------------------------------------------ 预估 --
    def estimate(self, corpus: RawCorpus) -> VlmEstimate:
        """**不产生任何调用**地预估成本。

        需要遍历语料统计图片总数、按内容哈希去重、查缓存命中数，并按
        ``max_calls`` 截断。注意为了算内容哈希，这一步必须下载图片 —— 因此
        "不产生调用"指的是不产生 **VLM** 调用，下载仍会发生。若连下载也要避免，
        调用方应先检查 :attr:`RawCorpus.total_images`。

        Args:
            corpus: 原始语料。

        Returns:
            成本预估。
        """
        return self._estimate_of(self._plan(corpus, with_payload=False))

    @staticmethod
    def _estimate_of(plan: _Plan) -> VlmEstimate:
        """把计划折算成预估。

        ``planned_calls`` 不写成 ``unique - cached``：无法解码的图片
        （``failed``）也占着去重位，却永远不会产生调用，相减会多报。
        """
        return VlmEstimate(
            total_images=plan.total_images,
            unique_images=len(plan.images),
            cached_images=plan.cached,
            planned_calls=plan.planned_calls,
            truncated=plan.skipped_by_limit > 0,
        )

    # ------------------------------------------------------------------ 分析 --
    def analyze(
        self,
        corpus: RawCorpus,
        *,
        progress: Callable[[str, float], None] | None = None,
    ) -> VlmResult:
        """分析语料中的图片。

        Args:
            corpus: 原始语料。
            progress: 进度回调 ``("图片分析", 完成比例)``。

        Returns:
            分析结果。**任何单张图片的失败都只记进** ``warnings`` **而不抛出** ——
            图片分析是增量信息，不该让整次运行白跑。

        Note:
            图片洞察转成 :class:`TextUnit` 时，``text`` 由 ``description`` 与
            ``pain_hints`` 拼接而成，``from_image=True``，``weight`` 取该笔记
            点赞数算出的权重而非固定值 —— 否则视觉证据会与文本证据在同一尺度上
            不可比。
        """
        # 缺 Pillow 时立刻失败，而不是让上千张图逐张报同一句错。这是**安装问题**
        # 而不是单张图的偶发失败，属于第 6 条「失败降级」的适用范围之外 ——
        # 用户需要看到的是可复制执行的修复命令。
        if any(note.images for note in corpus.notes):
            require_pillow("VLM 图片分析")

        plan = self._plan(corpus, with_payload=True)
        result = VlmResult(estimate=self._estimate_of(plan), warnings=list(plan.warnings))

        if plan.skipped_by_limit:
            result.warnings.append(
                f"已达调用上限 max_vlm_calls={self.max_calls}，"
                f"{plan.skipped_by_limit} 张图片未分析 —— 本次结果是部分覆盖，"
                "不代表语料里只有这些视觉痛点。"
            )

        pending = [item for item in plan.images if item.status == "pending"]
        completed = self._run(pending, progress)

        # 只缓存成功的结果：把一次限流错误缓存下来，会让这张图在缓存过期前
        # 永远拿不到分析（而且不会有人发现，因为"有缓存"看起来是正常的）。
        if self.cache is not None:
            for item in pending:
                insight = completed.get(item.image_hash)
                if insight is not None and not insight.error:
                    self.cache.put(insight)

        scale = _likes_scale(corpus)
        failures: list[tuple[str, str]] = []
        for item in plan.images:
            if item.status == "skipped":
                continue
            # ``cached`` 与 ``failed`` 在计划阶段就已经有洞察了（缓存命中 / 图片不可用），
            # 只有 ``pending`` 的洞察来自这一轮的调用结果。
            insight = item.insight or completed.get(item.image_hash)
            if insight is None:
                continue

            result.insights.setdefault(item.note_id, []).append(insight)
            for other_note in item.duplicate_note_ids:
                result.insights.setdefault(other_note, []).append(
                    replace(insight, note_id=other_note)
                )

            if insight.error:
                failures.append((item.url, insight.error))
                continue

            unit = _to_unit(insight, item, scale)
            if unit is not None:
                result.units.append(unit)

        result.warnings.extend(_summarize_failures("分析失败", failures))
        return result

    def _run(
        self,
        pending: list[_PlannedImage],
        progress: Callable[[str, float], None] | None,
    ) -> dict[str, ImageInsight]:
        """并发执行调用，按图片哈希返回结果。"""
        if not pending:
            return {}

        limiter = _RateLimiter(VLM_MIN_REQUEST_INTERVAL)
        completed: dict[str, ImageInsight] = {}
        workers = min(self.max_concurrency, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._analyze_one, item, limiter): item for item in pending}
            done = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    completed[item.image_hash] = future.result()
                except Exception as exc:  # noqa: BLE001
                    # 兜底：_analyze_one 自己保证不抛，这里拦住的是它内部
                    # 意料之外的缺陷。即便如此也不能让整次运行白跑。
                    completed[item.image_hash] = ImageInsight(
                        url=item.url,
                        image_hash=item.image_hash,
                        note_id=item.note_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                done += 1
                if progress is not None:
                    progress("图片分析", done / len(pending))
        return completed

    def _analyze_one(self, item: _PlannedImage, limiter: _RateLimiter) -> ImageInsight:
        """分析一张图。任何失败都返回带 ``error`` 的洞察，**绝不抛出**。"""
        data_url = build_data_url(item.payload, item.media_type)
        last_error = ""
        for attempt in range(VLM_MAX_ATTEMPTS):
            limiter.acquire()
            try:
                response = self.provider.complete_vision(
                    VISION_PROMPT,
                    [data_url],
                    temperature=self.temperature,
                )
                return self._parse_insight(item, response.text)
            except LLMError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if _is_rate_limited(exc):
                    # 被限流时立刻放手：继续重试只会把限额烧得更彻底，
                    # 而且整轮耗时会被拉长到用户以为程序卡死。
                    break
                if attempt + 1 < VLM_MAX_ATTEMPTS:
                    _sleep(min(VLM_RETRY_BACKOFF_BASE * (2**attempt), VLM_RETRY_BACKOFF_MAX))
            except Exception as exc:  # noqa: BLE001
                # 非 LLMError 视为代码/配置缺陷（如传错的参数），重试没有意义。
                last_error = f"{type(exc).__name__}: {exc}"
                break
        return ImageInsight(
            url=item.url,
            image_hash=item.image_hash,
            note_id=item.note_id,
            error=last_error or "未知错误",
        )

    @staticmethod
    def _parse_insight(item: _PlannedImage, text: str) -> ImageInsight:
        """把模型回复解析成洞察。

        Raises:
            LLMError: 回复里没有可解析的 JSON 对象（会被当作可重试的调用失败）。
        """
        data = extract_json(text)
        if not isinstance(data, dict):
            raise LLMError(f"VLM 返回的 JSON 不是对象，而是 {type(data).__name__}")

        description = str(data.get("description") or "").strip()[:_MAX_DESCRIPTION_CHARS]
        raw_hints = data.get("pain_hints")
        if isinstance(raw_hints, str):
            # 模型偶尔把数组写成字符串 —— 当成单条处理比整张图判失败要好。
            raw_hints = [raw_hints]
        if not isinstance(raw_hints, list):
            raw_hints = []
        # 只保留字符串元素：把 dict/int 拼进文本会污染 embedding，
        # 而这类回复本身说明模型没按格式走，不该假装它有效。
        pain_hints = [h.strip() for h in raw_hints if isinstance(h, str) and h.strip()]
        return ImageInsight(
            url=item.url,
            image_hash=item.image_hash,
            note_id=item.note_id,
            description=description,
            pain_hints=pain_hints[:_MAX_PAIN_HINTS],
        )

    # ------------------------------------------------------------------ 计划 --
    def _plan(self, corpus: RawCorpus, *, with_payload: bool) -> _Plan:
        """遍历语料，产出"要分析哪些图、各自什么状态"。

        七条省钱策略里有四条在这里落地：单篇上限、内容去重、缓存命中、
        调用上限截断，外加每次运行只下载一次的字节复用。

        Args:
            corpus: 原始语料。
            with_payload: 是否保留压缩后的字节（``analyze`` 需要，``estimate`` 不需要）。
        """
        images: list[_PlannedImage] = []
        by_hash: dict[str, _PlannedImage] = {}
        fetch_failures: list[tuple[str, str]] = []
        skipped = 0
        pending_count = 0

        for note in corpus.notes:
            # 第 3 条：单篇最多取前 N 张。取前 N 而非抽样 —— 笔记的第一张通常是
            # 封面，信息密度最高，而且"取前 N"是可复现的。
            for url in note.images[: self.max_images_per_note]:
                try:
                    data = fetch_image(url)
                except (OSError, ValueError) as exc:
                    fetch_failures.append((url, str(exc)))
                    continue

                digest = image_hash(data)

                # 第 1 条：内容去重。
                existing = by_hash.get(digest)
                if existing is not None:
                    if note.note_id != existing.note_id and note.note_id not in (
                        existing.duplicate_note_ids
                    ):
                        existing.duplicate_note_ids.append(note.note_id)
                    continue

                item = _PlannedImage(
                    url=url,
                    note_id=note.note_id,
                    image_hash=digest,
                    likes=note.likes,
                    publish_time=note.publish_time,
                    status="pending",
                )
                by_hash[digest] = item
                images.append(item)

                # 第 4 条：按内容哈希查缓存。
                cached = self.cache.get(digest) if self.cache is not None else None
                if cached is not None:
                    cached.from_cache = True
                    cached.image_hash = digest
                    item.insight = cached
                    item.status = "cached"
                    continue

                # 第 7 条的一部分：max_calls 截断。缓存命中不占额度。
                if self.max_calls is not None and pending_count >= self.max_calls:
                    item.status = "skipped"
                    skipped += 1
                    continue

                # 第 2 条：长边压缩。压缩在计划阶段就地完成，字节直接留给调用用，
                # 避免"哈希时下一次、调用前再下一次"的重复下载。
                if with_payload:
                    try:
                        payload, media_type = downscale_image(data, max_edge=self.max_edge)
                    except ValueError as exc:
                        item.status = "failed"
                        item.insight = ImageInsight(
                            url=url,
                            image_hash=digest,
                            note_id=note.note_id,
                            error=f"图片无法用于分析：{exc}",
                        )
                        continue
                    item.payload = payload
                    item.media_type = media_type

                pending_count += 1

        warnings = _summarize_failures("获取失败，已跳过", fetch_failures)
        return _Plan(
            total_images=corpus.total_images,
            images=images,
            skipped_by_limit=skipped,
            warnings=warnings,
        )


def _summarize_failures(subject: str, failures: list[tuple[str, str]]) -> list[str]:
    """把失败清单转成 warnings，逐条列出但不超过 :data:`_MAX_DETAILED_WARNINGS` 条。"""
    lines = [f"图片{subject}：{url}（{error}）" for url, error in failures[:_MAX_DETAILED_WARNINGS]]
    hidden = len(failures) - _MAX_DETAILED_WARNINGS
    if hidden > 0:
        example = failures[_MAX_DETAILED_WARNINGS][1]
        lines.append(f"另有 {hidden} 张图片{subject}（错误已折叠，例如：{example}）")
    return lines


def _likes_scale(corpus: RawCorpus) -> float:
    """点赞数归一化基准 —— 与 ``clean.compute_weights`` 的基准保持一致。

    必须把评论也算进来：``compute_weights`` 拿到的是**全部文本单元**的点赞数，
    少算评论会让基准偏小，视觉证据的权重整体被抬高。
    """
    likes = [note.likes for note in corpus.notes] + [comment.likes for comment in corpus.comments]
    return float(max(likes)) if likes else 0.0


def _weight_for(likes: int, scale: float) -> float:
    """把点赞数压成 ``(0, 1]`` 的权重。

    这里是 ``pipeline/clean.compute_weights`` 的**等价实现**，刻意没有复用：

    * 那个文件的实现由另一个工作流负责，直接 import 会让两个模块在合并前互相
      阻塞，也会让"每个模块的测试不依赖其他流的实现"这条约定失效；
    * 本模块只需要「单条」权重，而 ``compute_weights`` 是「整批」接口 ——
      为了算一条而构造一整批输入，反而更容易出现两处口径不一致。

    两处公式必须保持一致（``log1p`` 压缩后按最大值归一），否则图片证据与文本
    证据不在同一尺度上，「痛点强度」因子就会偏向其中一边。

    Note:
        公式与 ``clean.compute_weights`` 一样会把 0 赞压成 0.0，而
        :class:`~xhs_pain_miner.models.TextUnit` 的 ``weight`` 注释写的是
        ``(0, 1]``。这里选择与清洗层对齐而不是各自补一个地板值 —— 不一致的
        尺度比边界取值更危险。已在验收报告中作为接口问题上报。
    """
    if scale <= 0:
        # 全语料都没有点赞信号时一视同仁，而不是全 0。
        return 1.0
    return math.log1p(max(likes, 0)) / math.log1p(scale)


def _to_unit(insight: ImageInsight, item: _PlannedImage, scale: float) -> TextUnit | None:
    """把图片洞察转成文本单元。

    ``pain_hints`` 为空时返回 ``None`` —— 图里没有痛点。这类图片在语料里占多数
    （正常的商品图），放它们进聚类会灌入大量"这张图没什么问题"的噪声，
    把真正的问题表达稀释掉。
    """
    if not insight.pain_hints:
        return None

    parts = [insight.description, "；".join(insight.pain_hints)]
    text = " ".join(part for part in parts if part)
    return TextUnit(
        text=text,
        source="note",
        likes=item.likes,
        weight=_weight_for(item.likes, scale),
        note_id=item.note_id,
        created_at=item.publish_time,
        images=[item.url],
        from_image=True,
    )
