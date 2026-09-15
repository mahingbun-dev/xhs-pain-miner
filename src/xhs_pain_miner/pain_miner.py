"""XHS Pain Miner 门面 —— 面向使用者的统一入口。

M0 里程碑的状态：

* :meth:`PainMiner.collect` —— **已实现**，可真实采集（受所选采集后端约束）。
* :meth:`PainMiner.mine` —— 分析流水线（清洗 → 聚类 → 命名 → 竞品调研 → 机会分 →
  渲染）将在 M1 里程碑实现，当前调用会抛出 :class:`PipelineNotAvailableError`。

命令行用户可以先跑 ``xhs-pain-miner collect`` 验证采集层与配置是否正确。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xhs_pain_miner.collectors.base import CollectorBackend
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.config import Settings, load_settings
from xhs_pain_miner.models import MiningResult, RawCorpus

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

PIPELINE_PENDING_MESSAGE = (
    "分析流水线将在 M1 里程碑实现（清洗 → 向量化聚类 → LLM 命名 → 竞品调研 → 机会分 → 渲染）。\n"
    "当前可以先用 `xhs-pain-miner collect` 验证采集层与配置。"
)

# 不可见字符：从网页复制关键词时很常见，肉眼看不见但 strip() 不会去掉，
# 会让「关键词是否为空」的判断失效。必须写成转义序列 —— 直接写字面字符会让
# 这一行在代码审查时完全看不出来，也容易在编辑中被误删。
#
# 覆盖：软连字符 / 蒙古文元音分隔符 / 零宽字符族 / 双向文本控制符 /
#       不可见运算符 / 双向隔离符 / 谚文填充符 / BOM
_INVISIBLE_CHARS = (
    "\u00ad"  # SOFT HYPHEN
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u200b\u200c\u200d"  # ZWSP / ZWNJ / ZWJ
    "\u200e\u200f"  # LRM / RLM
    "\u202a\u202b\u202c\u202d\u202e"  # 双向文本嵌入与覆盖
    "\u2060\u2061\u2062\u2063\u2064"  # WORD JOINER / 不可见运算符
    "\u2066\u2067\u2068\u2069"  # 双向隔离
    "\u3164"  # HANGUL FILLER
    "\ufeff"  # BOM / ZWNBSP
)
_REMOVE_INVISIBLE = str.maketrans("", "", _INVISIBLE_CHARS)


def normalize_keyword(keyword: str) -> str:
    """规范化品类关键词。

    去掉首尾空白与零宽字符；若结果为空则抛 ``ValueError``。

    为什么需要它：采集后端在关键词为空时可能回退到自己的默认值
    （内置样例后端就是如此），使 ``corpus.keyword`` 与调用方传入的值不一致 ——
    这是一个没人会注意到的静默数据错误。**CLI 与 Python API 必须走同一个校验**，
    否则两层契约会不一致。

    Args:
        keyword: 用户传入的品类关键词。

    Returns:
        规范化后的关键词。

    Raises:
        ValueError: 关键词不是字符串，或去掉空白后为空。
    """
    if not isinstance(keyword, str):
        raise ValueError(f"关键词必须是字符串，收到 {type(keyword).__name__}")
    cleaned = keyword.translate(_REMOVE_INVISIBLE).strip()
    if not cleaned:
        raise ValueError("关键词不能为空")
    return cleaned


class PipelineNotAvailableError(RuntimeError):
    """分析流水线尚未实现。"""


class PainMiner:
    """痛点挖掘的主入口。

    Usage::

        from xhs_pain_miner import PainMiner

        miner = PainMiner()
        corpus = miner.collect("防晒霜", limit=100)
        print(corpus.summary())

    Args:
        settings: 全局配置。为 ``None`` 时从环境变量 / ``.env`` 加载。
        api_key: 便捷参数，覆盖 ``settings.llm_api_key``（方便脚本里临时指定）。
        model: 便捷参数，覆盖 ``settings.llm_model``。
        collector: 自定义采集后端。为 ``None`` 时按配置构造。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        collector: CollectorBackend | None = None,
    ) -> None:
        self.settings = settings or load_settings(llm_api_key=api_key, llm_model=model)
        if settings is not None and (api_key or model):
            # settings 由调用方传入时，便捷参数仍然生效
            self.settings = settings.model_copy(
                update={k: v for k, v in {"llm_api_key": api_key, "llm_model": model}.items() if v}
            )
        self._collector = collector

    # ------------------------------------------------------------------ 采集 --
    @property
    def collector(self) -> CollectorBackend:
        """当前采集后端（惰性构造）。"""
        if self._collector is None:
            self._collector = build_collector(self.settings)
        return self._collector

    def collect(
        self,
        keyword: str,
        *,
        limit: int | None = None,
        max_comments_per_note: int | None = None,
    ) -> RawCorpus:
        """采集一个品类的笔记与评论。

        Args:
            keyword: 品类关键词，如 ``"防晒霜"``。
            limit: 最多采集的笔记数。默认取 ``settings.max_notes``。
            max_comments_per_note: 每篇笔记最多采集的评论数。

        Returns:
            采集结果。

        Raises:
            ValueError: 关键词为空或不是字符串。
            CollectorError: 采集失败。
        """
        return self.collector.collect(
            normalize_keyword(keyword),
            limit=limit if limit is not None else self.settings.max_notes,
            max_comments_per_note=(
                max_comments_per_note
                if max_comments_per_note is not None
                else self.settings.max_comments_per_note
            ),
        )

    # ------------------------------------------------------------------ 分析 --
    def mine(
        self,
        keyword: str,
        *,
        notes_count: int | None = None,
        deep: bool = False,
        progress: Callable[[str, float], None] | None = None,
    ) -> MiningResult:
        """采集并分析一个品类，产出机会卡片。

        Args:
            keyword: 品类关键词。
            notes_count: 采集笔记数，默认取 ``settings.max_notes``。
            deep: 是否开启 VLM 图片分析（成本与耗时显著上升）。
            progress: 进度回调 ``(阶段名, 完成比例)``。

        Returns:
            包含机会卡片与成本统计的分析结果。

        Raises:
            PipelineNotAvailableError: 流水线尚未实现（M1 里程碑）。
            CollectorError: 采集失败。
            LLMError: LLM 调用失败。
        """
        raise PipelineNotAvailableError(PIPELINE_PENDING_MESSAGE)

    def close(self) -> None:
        """释放资源（当前无需要显式释放的持有对象）。"""

    def __enter__(self) -> PainMiner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
