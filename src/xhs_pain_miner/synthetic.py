"""确定性的模拟图片生成器 —— **仅供内置 fixture、测试与演示**。

生产路径永远不会调用本模块：真实语料的图片地址是 ``http(s)://``，由
:func:`xhs_pain_miner.pipeline.vlm.fetch_image` 下载。

存在的理由：内置 fixture 里的图片 URL 指向 ``.invalid`` 假域名（这是刻意的 ——
样例数据不该让任何人真的去请求平台 CDN），因此 VLM 链路无法用真实地址测试。
本模块按固定种子生成图片，让「去重 / 压缩 / 缓存 / 并发 / 降级」这五条省钱
策略可以被**确定性**地测试和压测，而不必依赖外网。

.. warning::
   请不要为了让 demo 好看而在这里生成"看起来像真实笔记"的图片。
   合成图的用途是验证代码，不是伪装成真实数据。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xhs_pain_miner.pipeline.deps import MissingDependencyError, require

SYNTHETIC_SCHEME = "synthetic://"
"""合成图的地址前缀，形如 ``synthetic://note-42/0``。"""

VLM_EXTRA = 'pip install -e ".[vlm]"'
"""缺失 Pillow 时的修复命令。

``pipeline/deps.require()`` 给出的提示固定指向 ``[analysis]`` extra
（numpy/sklearn），但图片链路缺的是 ``[vlm]``。按 deps 模块自己的设计原则
——错误消息必须是**可直接复制执行**的修复命令——这里重新包装一次。
"""

_IMAGE_SIZE = (512, 640)

# 生成时使用的中文字体候选。找不到就退回 Pillow 内置字体并改用英文 ——
# 内置字体没有 CJK 字形，硬写中文会渲染成一排方框，让 VLM 完全读不出东西。
_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
)

# --------------------------------------------------------------------------- #
# 版式常量
#
# 集中放在这里而不是散落在绘制代码里：测试要按这些坐标采样像素来断言
# 「左暗右亮」，draw 与 assert 必须引用同一组数字，否则改版式会静默放过回归。
# --------------------------------------------------------------------------- #

_MARGIN = 24
_HEADER_TOP = 16
_FOOTER_TOP = 536
_BLOCK_TOP = 130
_BLOCK_BOTTOM = 500
_LEFT_BLOCK = (24, 240)
_RIGHT_BLOCK = (272, 488)
_DIVIDER_X = 256

_BG = (246, 246, 248)
_INK = (32, 32, 36)

# 对比图的左右明暗 —— 这是 ``pain_hints`` 能被提取出来的前提：
# 纯色块里 VLM 什么也看不出来，左右对比才让它有东西可讲。
_COMPARE_DARK = (58, 62, 74)
_COMPARE_LIGHT = (232, 236, 240)

_HEADLINE_SIZES = (30, 26, 22, 18, 14)
_CAPTION_SIZE = 20
_JPEG_QUALITY = 85

_DEFAULT_SUBJECT = "好物记录"

_HEADLINE_SUFFIXES = (
    "真实记录",
    "第 3 次回购",
    "踩雷实录",
    "开箱实测",
    "前后对比",
)
_CAPTIONS_PLAIN = (
    "室内自然光下拍摄",
    "用了一个月后的样子",
    "包装与质地特写",
    "日常通勤随身带",
)
_CAPTIONS_COMPARE = (
    "左：刚涂抹 / 右：户外 2 小时后",
    "左：宣传图 / 右：实际到手",
    "左：使用前 / 右：使用后",
    "左：白天 / 右：晚上灯光下",
)

# 没有 CJK 字体时的英文替代文本。必须与中文列表一一对应地保持同样的
# 「有内容可讲」程度 —— 退回英文仍然要能测出 description 与 pain_hints。
_EN_HEADLINES = (
    "Daily record",
    "Third repurchase",
    "Bad buy log",
    "Unboxing test",
    "Before and after",
)
_EN_CAPTIONS_PLAIN = (
    "shot in natural indoor light",
    "after one month of use",
    "packaging and texture close-up",
    "carried around every day",
)
_EN_CAPTIONS_COMPARE = (
    "left: just applied / right: after 2 hours outdoors",
    "left: promo photo / right: what arrived",
    "left: before / right: after",
    "left: daylight / right: indoor lamp",
)


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """一张合成图的确定性规格。

    同样的 ``spec`` 永远生成同样的字节 —— 这是缓存与去重测试能成立的前提。
    """

    note_id: str
    index: int
    headline: str
    """图片上的主标题（模拟笔记封面上的大字）。"""

    caption: str = ""
    """图片上的说明文字（模拟图片内的标注）。"""

    variant: str = "plain"
    """版式：``plain`` 单图 / ``compare`` 左右对比图。"""


def require_pillow(purpose: str) -> tuple[Any, Any, Any]:
    """导入 Pillow 的三件套 ``(Image, ImageDraw, ImageFont)``。

    放在本模块是为了让 :mod:`xhs_pain_miner.pipeline.vlm` 复用同一份实现 ——
    两处各写一遍的话，其中一处迟早会漏掉 ``[vlm]`` 的修复指引。

    Args:
        purpose: 用途描述，会写进错误消息（用户据此判断自己是否真的需要它）。

    Returns:
        ``(Image, ImageDraw, ImageFont)`` 三个模块对象。

    Raises:
        MissingDependencyError: 未安装 Pillow。
    """
    try:
        return (
            require("PIL.Image", purpose=purpose),
            require("PIL.ImageDraw", purpose=purpose),
            require("PIL.ImageFont", purpose=purpose),
        )
    except MissingDependencyError as exc:
        raise MissingDependencyError(
            f"{purpose}需要可选依赖 Pillow，但它未安装。\n安装命令：{VLM_EXTRA}"
        ) from exc


def parse_synthetic_url(url: str) -> tuple[str, int] | None:
    """解析 ``synthetic://<note_id>/<index>``。

    Args:
        url: 待解析地址。

    Returns:
        ``(note_id, index)``；不是合成图地址则返回 ``None``。
    """
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if not candidate.startswith(SYNTHETIC_SCHEME):
        return None

    note_id, sep, index_text = candidate[len(SYNTHETIC_SCHEME) :].rpartition("/")
    if not sep:
        return None
    note_id = note_id.strip()
    # note_id 里再出现 "/" 说明地址有更多层级（``synthetic://a/0/1``），按非法处理 ——
    # 留一个"多余部分被静默忽略"的口子，会让打错的地址悄悄命中错误的图片。
    if not note_id or "/" in note_id:
        return None

    index_text = index_text.strip()
    # 用 isdigit 而不是 try/int：``int()`` 会接受 "  12 "、"＋12" 这类写法，
    # 而地址是从语料里读来的，宽容解析只会掩盖上游的数据错误。
    if not index_text.isdigit():
        return None
    return note_id, int(index_text)


def _digest(*parts: object) -> bytes:
    """把若干字段拼成稳定摘要。

    用 ``hashlib`` 而不是内置 ``hash()``：后者对 str 加了每进程随机盐
    （PYTHONHASHSEED），换个进程就得到不同的图 —— 缓存与去重会全部失效，
    而且这种失效只在跨进程时才出现，最难排查。
    """
    joined = "\x00".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).digest()


def build_spec(note_id: str, index: int, *, seed_text: str = "") -> SyntheticSpec:
    """按 ``note_id`` 与 ``index`` 派生一个稳定的规格。

    同一个 ``(note_id, index)`` 必须永远得到同一个 ``headline`` / ``caption`` ——
    用哈希派生，不用随机数。

    Args:
        note_id: 笔记 ID。
        index: 图片在该笔记中的序号。
        seed_text: 参与派生的文本（通常是笔记标题），让图片内容与笔记相关。

    Returns:
        确定性规格。
    """
    digest = _digest("spec", note_id, index, seed_text)

    # 一半图片是对比图：只画单图的话，VISION_PROMPT 要求的 pain_hints
    # 整条链路都测不到（纯色块里没有"可抱怨的点"）。
    variant = "compare" if digest[0] % 2 == 0 else "plain"

    subject = seed_text.strip()[:10] or _DEFAULT_SUBJECT
    suffix = _HEADLINE_SUFFIXES[digest[1] % len(_HEADLINE_SUFFIXES)]
    captions = _CAPTIONS_COMPARE if variant == "compare" else _CAPTIONS_PLAIN

    return SyntheticSpec(
        note_id=note_id,
        index=index,
        headline=f"{subject}｜{suffix}",
        caption=captions[digest[2] % len(captions)],
        variant=variant,
    )


def _has_cjk(text: str) -> bool:
    """文本里是否含中文（含扩展区与常用标点区附近的范围）。"""
    return any("\u3000" <= ch <= "\u9fff" for ch in text)


def _resolve_font(size: int) -> tuple[Any, bool]:
    """按 :data:`_FONT_CANDIDATES` 顺序找可用字体。

    Returns:
        ``(字体对象, 是否支持中文)``。找不到中文字体时返回内置字体并置 ``False`` ——
        调用方据此改用英文文本，而不是硬画一排方框。
    """
    _, _, pil_font = require_pillow("生成合成图片")
    for candidate in _FONT_CANDIDATES:
        if not Path(candidate).is_file():
            continue
        try:
            return pil_font.truetype(candidate, size), True
        except OSError:  # 文件在但读不了（权限 / 损坏）—— 换下一个候选
            continue
    try:
        return pil_font.load_default(size=size), False
    except TypeError:  # Pillow < 10.1 的 load_default 不接受 size
        return pil_font.load_default(), False


def _localized_text(spec: SyntheticSpec, has_cjk: bool) -> tuple[str, str]:
    """按字体能力选择图片上的文字。

    有中文字体就用 spec 里的原文；没有则换一组等价的英文文本。
    """
    if has_cjk or not _has_cjk(spec.headline + spec.caption):
        return spec.headline, spec.caption

    digest = _digest("en", spec.note_id, spec.index, spec.variant)
    captions = _EN_CAPTIONS_COMPARE if spec.variant == "compare" else _EN_CAPTIONS_PLAIN
    return (
        _EN_HEADLINES[digest[0] % len(_EN_HEADLINES)],
        captions[digest[1] % len(captions)],
    )


def _draw_text(
    draw: Any,
    xy: tuple[float, float],
    text: str,
    *,
    font: Any,
    fill: tuple[int, int, int],
    anchor: str = "mm",
) -> None:
    """写一行字（``anchor="mm"`` 表示以给定点为中心）。"""
    if text:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _fit_font(draw: Any, text: str, sizes: tuple[int, ...], max_width: float) -> tuple[Any, bool]:
    """选一个能让 ``text`` 放进 ``max_width`` 的最大字号。"""
    font: Any = None
    has_cjk = True
    for size in sizes:
        font, has_cjk = _resolve_font(size)
        if draw.textlength(text, font=font) <= max_width:
            break
    return font, has_cjk


def render_image(spec: SyntheticSpec) -> bytes:
    """把规格渲染成 JPEG 字节。

    版式要能让 VLM 分析出**有意义的内容**：``plain`` 画一个带标题与说明的色块；
    ``compare`` 画左右两块，左边偏暗、右边偏亮，模拟"使用前 / 使用后"对比图 ——
    这样 :data:`~xhs_pain_miner.pipeline.vlm.VISION_PROMPT` 要求的
    ``pain_hints`` 才有东西可提取，否则整条 VLM 链路测了等于没测。

    Args:
        spec: 图片规格。

    Returns:
        JPEG 字节。

    Raises:
        xhs_pain_miner.pipeline.deps.MissingDependencyError: 缺少 Pillow。
    """
    pil_image, pil_draw, _ = require_pillow("生成合成图片")

    width, _ = _IMAGE_SIZE
    image = pil_image.new("RGB", _IMAGE_SIZE, _BG)
    draw = pil_draw.Draw(image)

    headline, caption = _localized_text(spec, _resolve_font(_HEADLINE_SIZES[0])[1])
    headline_font, _ = _fit_font(draw, headline, _HEADLINE_SIZES, width - 2 * _MARGIN)
    caption_font, _ = _fit_font(draw, caption, (_CAPTION_SIZE, 16, 13), width - 2 * _MARGIN)
    _draw_text(draw, (width / 2, _HEADER_TOP + 30), headline, font=headline_font, fill=_INK)

    # 主体色由哈希派生：不同图片一眼能看出区别（也省得所有图长得一模一样）。
    digest = _digest("color", spec.note_id, spec.index, spec.variant)
    base = (90 + digest[3] % 110, 90 + digest[4] % 110, 90 + digest[5] % 110)

    if spec.variant == "compare":
        _draw_compare(draw, base)
        left_text, sep, right_text = caption.partition(" / ")
        if sep:
            _draw_text(
                draw,
                ((_LEFT_BLOCK[0] + _LEFT_BLOCK[1]) / 2, _FOOTER_TOP + 30),
                left_text,
                font=caption_font,
                fill=_INK,
            )
            _draw_text(
                draw,
                ((_RIGHT_BLOCK[0] + _RIGHT_BLOCK[1]) / 2, _FOOTER_TOP + 30),
                right_text,
                font=caption_font,
                fill=_INK,
            )
        else:
            _draw_text(draw, (width / 2, _FOOTER_TOP + 30), caption, font=caption_font, fill=_INK)
    else:
        _draw_plain(draw, base)
        _draw_text(draw, (width / 2, _FOOTER_TOP + 30), caption, font=caption_font, fill=_INK)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _draw_plain(draw: Any, base: tuple[int, int, int]) -> None:
    """单图版式：一整块主色 + 居中的"产品"矩形。"""
    width, _ = _IMAGE_SIZE
    draw.rectangle((_MARGIN, _BLOCK_TOP, width - _MARGIN, _BLOCK_BOTTOM), fill=base)
    # 居中且左右对称：单图版式的左右两半亮度应当接近，
    # 不然「对比图左暗右亮」的断言就失去了区分度。
    draw.rectangle((width / 2 - 90, 230, width / 2 + 90, 400), fill=_COMPARE_LIGHT)


def _draw_compare(draw: Any, base: tuple[int, int, int]) -> None:
    """对比图版式：左暗右亮两块 + 中缝分隔线。"""
    draw.rectangle((_LEFT_BLOCK[0], _BLOCK_TOP, _LEFT_BLOCK[1], _BLOCK_BOTTOM), fill=_COMPARE_DARK)
    draw.rectangle(
        (_RIGHT_BLOCK[0], _BLOCK_TOP, _RIGHT_BLOCK[1], _BLOCK_BOTTOM), fill=_COMPARE_LIGHT
    )
    draw.line((_DIVIDER_X, _BLOCK_TOP, _DIVIDER_X, _BLOCK_BOTTOM), fill=base, width=3)


def synthesize(url: str, *, seed_text: str = "") -> bytes:
    """``fetch_image`` 处理 ``synthetic://`` 时调用的入口。

    Args:
        url: 合成图地址。
        seed_text: 参与内容派生的文本。

    Returns:
        JPEG 字节。

    Raises:
        ValueError: 不是合法的合成图地址。
    """
    parsed = parse_synthetic_url(url)
    if parsed is None:
        raise ValueError(
            f"不是合法的合成图地址: {url!r}（应形如 {SYNTHETIC_SCHEME}<note_id>/<index>）"
        )
    note_id, index = parsed
    return render_image(build_spec(note_id, index, seed_text=seed_text))
