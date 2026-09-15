"""合成图生成器测试 —— 确定性、版式可断言、字体回退不渲染方框。

合成图存在的唯一理由是让 VLM 链路可以在**离线且确定性**的前提下被测：
去重要有"同一张图"、缓存要有"同一个哈希"、压缩要有"真的超大图"。
因此本模块重点验证两件事：

1. 同样的 ``(note_id, index)`` 永远得到同样的字节（含跨进程 —— 用内置 ``hash()``
   派生会让它在换进程后全部失效，而那种缺陷只在缓存命中的场景下暴露）；
2. ``compare`` 版式真的画出了左右明暗对比，``pain_hints`` 才有东西可提取。
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from xhs_pain_miner import synthetic
from xhs_pain_miner.pipeline.deps import MissingDependencyError
from xhs_pain_miner.synthetic import (
    SYNTHETIC_SCHEME,
    SyntheticSpec,
    build_spec,
    parse_synthetic_url,
    render_image,
    synthesize,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 采样区：落在版式定义的色块内部（见 synthetic 模块顶部的版式常量），
# 避开标题与说明文字所在的行，这样断言的是"块"而不是"字"。
_LEFT_SAMPLE = (48, 200, 216, 400)
_RIGHT_SAMPLE = (296, 200, 464, 400)


@pytest.fixture
def pillow():
    """需要 Pillow 的测试用它显式跳过，而不是伪装通过。"""
    return pytest.importorskip("PIL.Image")


def _luma(data: bytes, box: tuple[int, int, int, int]) -> float:
    """取一块区域的平均亮度。"""
    from PIL import Image, ImageStat

    with Image.open(io.BytesIO(data)) as image:
        return ImageStat.Stat(image.convert("L").crop(box)).mean[0]


def _size_of(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.size


# --------------------------------------------------------------------------- #
# 地址解析
# --------------------------------------------------------------------------- #


class TestParseSyntheticUrl:
    """``synthetic://<note_id>/<index>`` 的正反例。"""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("synthetic://note-42/0", ("note-42", 0)),
            ("synthetic://note-42/7", ("note-42", 7)),
            ("synthetic://6a1b2c3d0001/12", ("6a1b2c3d0001", 12)),
            ("synthetic://note_42_x/3", ("note_42_x", 3)),
            ("  synthetic://padded/1  ", ("padded", 1)),
        ],
    )
    def test_valid_urls(self, url, expected):
        assert parse_synthetic_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "synthetic://",
            "synthetic://note-42",  # 缺序号
            "synthetic://note-42/",  # 序号为空
            "synthetic:///0",  # note_id 为空
            "synthetic://note-42/abc",  # 序号不是数字
            "synthetic://note-42/-1",  # 负序号
            "synthetic://note-42/1.5",
            "synthetic://note-42/1/2",  # 多了一层
            "https://cdn.example.invalid/x.jpg",
            "/tmp/local.jpg",
            "synthetic:/note-42/0",  # 少一个斜杠
        ],
    )
    def test_invalid_urls(self, url):
        assert parse_synthetic_url(url) is None

    def test_non_string_is_rejected(self):
        assert parse_synthetic_url(None) is None  # type: ignore[arg-type]

    def test_scheme_constant_matches_parser(self):
        assert parse_synthetic_url(f"{SYNTHETIC_SCHEME}x/0") == ("x", 0)


# --------------------------------------------------------------------------- #
# 规格派生
# --------------------------------------------------------------------------- #


class TestBuildSpec:
    """规格必须由 ``(note_id, index, seed_text)`` 决定，与调用次数、进程无关。"""

    def test_same_input_same_spec(self):
        assert build_spec("note-1", 0) == build_spec("note-1", 0)
        assert build_spec("note-1", 0, seed_text="防晒霜") == build_spec(
            "note-1", 0, seed_text="防晒霜"
        )

    def test_index_changes_the_spec(self):
        assert build_spec("note-1", 0) != build_spec("note-1", 1)

    def test_note_id_changes_the_spec(self):
        assert build_spec("note-1", 0) != build_spec("note-2", 0)

    def test_seed_text_changes_the_spec(self):
        assert build_spec("note-1", 0, seed_text="防晒霜") != build_spec("note-1", 0, seed_text="")

    def test_seed_text_lands_in_the_headline(self):
        spec = build_spec("note-1", 0, seed_text="防晒霜")
        assert "防晒霜" in spec.headline

    def test_variant_is_derived_not_random(self):
        """两种版式都要出现 —— 否则"对比图"的分支永远测不到。"""
        variants = {build_spec(f"note-{i}", 0).variant for i in range(12)}
        assert variants == {"plain", "compare"}

    def test_spec_is_immutable(self):
        spec = build_spec("note-1", 0)
        with pytest.raises(FrozenInstanceError):
            spec.headline = "改一下"  # type: ignore[misc]

    def test_caption_is_never_empty(self):
        assert build_spec("note-1", 0).caption


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


class TestRenderImage:
    """渲染出的字节要能当"图片"用，版式要能当"素材"用。"""

    def test_renders_a_jpeg(self, pillow):
        data = render_image(build_spec("note-1", 0))
        assert data[:2] == b"\xff\xd8"
        assert _size_of(data) == synthetic._IMAGE_SIZE

    def test_rendering_is_deterministic(self, pillow):
        spec = build_spec("note-1", 2, seed_text="防晒霜")
        assert render_image(spec) == render_image(spec)

    def test_same_url_renders_the_same_bytes(self, pillow):
        """缓存与去重测试的前提。"""
        assert synthesize("synthetic://note-1/0") == synthesize("synthetic://note-1/0")

    def test_different_index_renders_different_bytes(self, pillow):
        assert synthesize("synthetic://note-1/0") != synthesize("synthetic://note-1/1")

    def test_compare_layout_is_dark_left_bright_right(self, pillow):
        """``compare`` 的左右明暗是对比图的全部信息量所在。

        若两侧一样亮，VISION_PROMPT 要求的 ``pain_hints`` 就没有东西可提取 ——
        整条 VLM 链路会"测了等于没测"，而调用次数照样在烧钱。
        """
        spec = SyntheticSpec(
            note_id="note-1",
            index=0,
            headline="前后对比",
            caption="左：宣传图 / 右：实际到手",
            variant="compare",
        )
        data = render_image(spec)
        left = _luma(data, _LEFT_SAMPLE)
        right = _luma(data, _RIGHT_SAMPLE)
        assert left < right - 60, f"左 {left:.0f} / 右 {right:.0f} 没有形成对比"

    def test_plain_layout_is_symmetric(self, pillow):
        """单图版式左右应接近 —— 否则它也能"通过"上面那条对比断言。"""
        spec = SyntheticSpec(
            note_id="note-1", index=1, headline="日常记录", caption="随手拍", variant="plain"
        )
        data = render_image(spec)
        left = _luma(data, _LEFT_SAMPLE)
        right = _luma(data, _RIGHT_SAMPLE)
        assert abs(left - right) < 30, f"单图版式左右亮度差 {abs(left - right):.0f}，异常"

    def test_variants_differ(self, pillow):
        plain = SyntheticSpec(note_id="n", index=0, headline="h", caption="c", variant="plain")
        compare = SyntheticSpec(note_id="n", index=0, headline="h", caption="c", variant="compare")
        assert render_image(plain) != render_image(compare)

    def test_unknown_variant_falls_back_to_plain(self, pillow):
        """版式字符串来自派生结果，写错也不该直接崩掉整条 fixture 生成。"""
        spec = SyntheticSpec(note_id="n", index=0, headline="h", caption="c", variant="weird")
        assert render_image(spec)[:2] == b"\xff\xd8"


# --------------------------------------------------------------------------- #
# 字体回退
# --------------------------------------------------------------------------- #


class TestFontFallback:
    """找不到中文字体时改用英文 —— 内置字体没有 CJK 字形，硬写就是一排方框。"""

    def test_falls_back_when_no_font_file_exists(self, pillow, monkeypatch):
        monkeypatch.setattr(synthetic, "_FONT_CANDIDATES", ())
        _, has_cjk = synthetic._resolve_font(24)
        assert has_cjk is False

    def test_found_font_is_reported_as_cjk_capable(self, pillow):
        if not synthetic._resolve_font(24)[1]:
            pytest.skip("本机没有任何候选中文字体")
        assert synthetic._resolve_font(24)[1] is True

    def test_cjk_text_is_replaced_when_font_is_missing(self, pillow):
        spec = build_spec("note-1", 0, seed_text="防晒霜")
        headline, caption = synthetic._localized_text(spec, has_cjk=False)
        assert headline.isascii() and caption.isascii()
        assert headline and caption

    def test_cjk_text_is_kept_when_font_exists(self, pillow):
        spec = build_spec("note-1", 0, seed_text="防晒霜")
        assert synthetic._localized_text(spec, has_cjk=True) == (spec.headline, spec.caption)

    def test_ascii_text_is_never_replaced(self, pillow):
        spec = SyntheticSpec(note_id="n", index=0, headline="Hello", caption="world")
        assert synthetic._localized_text(spec, has_cjk=False) == ("Hello", "world")

    def test_rendering_without_cjk_font_still_produces_an_image(self, pillow, monkeypatch):
        monkeypatch.setattr(synthetic, "_FONT_CANDIDATES", ())
        data = render_image(build_spec("note-1", 0, seed_text="防晒霜"))
        assert data[:2] == b"\xff\xd8"
        assert _size_of(data) == synthetic._IMAGE_SIZE

    def test_fallback_actually_changes_the_pixels(self, pillow, monkeypatch):
        """证明"换文本"不是空转：有字体与无字体的渲染结果必须不同。"""
        if not synthetic._resolve_font(24)[1]:
            pytest.skip("本机没有任何候选中文字体")
        spec = build_spec("note-1", 0, seed_text="防晒霜")
        with_font = render_image(spec)
        monkeypatch.setattr(synthetic, "_FONT_CANDIDATES", ())
        without_font = render_image(spec)
        assert with_font != without_font

    def test_long_headline_is_shrunk_to_fit(self, pillow):
        """标题过长时必须缩字号 —— 溢出画布的文字 VLM 读不到，等于白画。"""
        spec = SyntheticSpec(
            note_id="n", index=0, headline="超长标题" * 8, caption="c", variant="plain"
        )
        assert render_image(spec)[:2] == b"\xff\xd8"


# --------------------------------------------------------------------------- #
# 入口与依赖
# --------------------------------------------------------------------------- #


class TestSynthesize:
    """``synthesize`` 是 ``fetch_image`` 处理 ``synthetic://`` 的入口。"""

    def test_matches_build_spec_plus_render(self, pillow):
        url = "synthetic://note-7/2"
        assert synthesize(url) == render_image(build_spec("note-7", 2))

    def test_invalid_url_raises_value_error(self):
        with pytest.raises(ValueError, match="不是合法的合成图地址"):
            synthesize("synthetic://broken")

    def test_seed_text_is_honoured(self, pillow):
        assert synthesize("synthetic://note-7/0", seed_text="防晒霜") != synthesize(
            "synthetic://note-7/0"
        )


class TestMissingPillow:
    """缺 Pillow 时必须给出 ``[vlm]`` 的修复命令，而不是裸 ImportError。"""

    def test_render_raises_actionable_error(self, monkeypatch):
        # sys.modules 里放 None 会让 importlib 抛 ImportError —— 无需真的卸载 Pillow
        monkeypatch.setitem(sys.modules, "PIL.Image", None)
        with pytest.raises(MissingDependencyError, match=r"\[vlm\]"):
            render_image(SyntheticSpec(note_id="n", index=0, headline="h"))


# --------------------------------------------------------------------------- #
# 跨进程确定性
# --------------------------------------------------------------------------- #


class TestCrossProcessDeterminism:
    """同一份规格在不同进程里必须渲染出同样的字节。

    这条测试专门盯住"用内置 ``hash()`` 派生"这个错误：它对 str 加了每进程随机盐，
    因此单进程内的确定性测试**照样通过**，只有在跨进程时才会暴露 ——
    而那时表现为"缓存全部失效、去重率归零"，是纯粹的账单损失。
    """

    _CODE = (
        "import hashlib, sys;"
        "sys.path.insert(0, 'src');"
        "from xhs_pain_miner.synthetic import build_spec, render_image;"
        "data = render_image(build_spec('note-1', 0, seed_text='防晒霜'));"
        "print(hashlib.sha256(data).hexdigest())"
    )

    def test_render_is_stable_across_processes(self, pillow):
        digests = set()
        for seed in ("0", "1"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            completed = subprocess.run(
                [sys.executable, "-c", self._CODE],
                capture_output=True,
                text=True,
                env=env,
                cwd=_REPO_ROOT,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            assert completed.stdout.strip()
            digests.add(completed.stdout.strip())

        assert len(digests) == 1, f"不同 PYTHONHASHSEED 渲染出了不同的图: {digests}"

    def test_hash_is_not_used_for_derivation(self, pillow):
        """直接比对同进程内两次渲染的哈希，作为上面那条的快速失败版本。"""
        spec = build_spec("note-1", 0, seed_text="防晒霜")
        assert (
            hashlib.sha256(render_image(spec)).hexdigest()
            == hashlib.sha256(render_image(spec)).hexdigest()
        )
