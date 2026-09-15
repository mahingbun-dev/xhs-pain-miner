"""文本清洗 —— 把原始语料切成可分析的文本单元。

这一层解决的是**信噪比**问题。小红书语料里混着大量对痛点挖掘毫无价值的文本：
广告引流、纯表情、无意义短句、"求链接"式互动。它们会直接污染聚类 —— embedding
空间里会聚出一个"求链接"的簇，而它显然不是产品机会。

边界（重要）
------------
本模块**不改写文本内容**，只做「保留 / 丢弃」判断与轻量规范化（空白、零宽字符、
全半角）。任何改写都会破坏证据链 —— 机会卡片上的每句话都必须能逐字点回原文，
这是本产品对"免费 LLM 摘要"的正面防守。凡是想在这里做摘要 / 纠错的改动，都先
读一遍 :mod:`~xhs_pain_miner.models` 里 ``Evidence`` 的说明。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from xhs_pain_miner.models import RawComment, RawCorpus, TextUnit, hash_id

MIN_UNIT_CHARS = 6
"""短于该长度的文本会被丢弃。

中文里 6 个字以下几乎无法承载一个完整痛点（"我也是" / "求链接" / "蹲一个"）。
调小会让聚类里塞满互动噪声，调大会丢掉"闷痘"这类两三个字就说清的强痛点 ——
所以对极短文本的取舍要按「能否独立表达一个可行动的抱怨」来判断。
"""

MAX_UNIT_CHARS = 500
"""超长文本的截断长度。

保留前 500 字足以覆盖痛点表达。不截断的话，少数长文会在 embedding 里主导
一整条向量，把同簇的短证据挤出去。
"""

_MIN_WEIGHT = 1e-3
"""证据权重的下界。

:class:`~xhs_pain_miner.models.TextUnit` 把 ``weight`` 的值域声明为 ``(0, 1]``，
而纯 log 归一化会让 0 赞的单元得到精确的 0.0 —— 那等于断言"这条证据不存在"。
0 赞评论依然是真实证据，只是信号弱，所以压到一个小正数而不是抹掉。
"""

# 广告与引流话术。命中即整条丢弃 —— 这类文本不仅无价值，还会因为措辞高度相似
# 而形成巨大的假簇。使用正则而非子串匹配，是因为平台用户会用插入符号规避检测。
_AD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(加|扣)\s*(v|V|微|威)\s*[:：]?\s*[a-zA-Z0-9_-]{4,}"),
    re.compile(r"(vx|wx|v信|微信|威信|weixin)\s*[:：]?\s*[a-zA-Z0-9_-]{4,}", re.IGNORECASE),
    re.compile(r"(私信|滴滴|戳)我(拿|要|领|发)?(链接|同款|资源|资料|教程)?"),
    re.compile(r"关注我.*(领取|获取|送)"),
    re.compile(r"(点击|戳)?(主页|下方)?链接.*(购买|下单|优惠)"),
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"(https?://|www\.)\S+"),
)

# 纯互动 / 无信息量的固定短语。「求链接」本身是需求信号，但它不描述痛点，
# 也不够具体到能支撑一个产品决策 —— 它的正确位置是需求热度统计，而不是痛点簇。
_LOW_VALUE_PHRASES: frozenset[str] = frozenset(
    {
        "蹲一个",
        "求链接",
        "求同款",
        "我也是",
        "同求",
        "打卡",
        "已收藏",
        "马住",
        "马克",
        "谢谢分享",
        "感谢分享",
        "学到了",
        "哈哈",
        "呵呵",
        "哈哈哈哈",
        "笑死",
        "第一",
        "前排",
    }
)

# 只有表情 / 标点 / 空白，没有任何文字内容
_NO_TEXT = re.compile(r"^[^\w\u4e00-\u9fff]+$")

# 零宽与方向控制字符。
# 必须写成 \uXXXX 转义而不是字面量：这些字符在编辑器里完全不可见，
# 直接写字面量会让这一行在 code review 时看不出任何内容，也极易被误删。
_INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")

# C0/C1 控制字符。\t \n \v \f \r 不在此列 —— 它们是空白，交给下面的空白折叠统一处理，
# 若在这里直接删除，"两句话"会被粘成一句，反而改动了语义。
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 空白（含全角空格 U+3000、不换行空格 U+00A0 —— Python 的 \s 按 Unicode 定义匹配）
_WHITESPACE = re.compile(r"\s+")

_IDEOGRAPHIC_SPACE = "\u3000"

# 去掉空白 / 标点 / 表情后剩下的"实义文本"。
# \W 在 str 模式下按 Unicode 匹配，中文属于 \w；下划线是 \w 但明显不算实义字符，
# 所以要显式排除，否则"____"会被当成 4 个字。
_NON_CORE = re.compile(r"[\W_]+")

_LOW_VALUE_BY_LENGTH: tuple[str, ...] = tuple(sorted(_LOW_VALUE_PHRASES, key=len, reverse=True))


def normalize_text(raw: str) -> str:
    """轻量规范化文本，**不改变语义**。

    做四件事：剥离控制字符与零宽字符、统一全角空白为半角、折叠连续空白与换行、
    去掉首尾空白。

    Args:
        raw: 原始文本。

    Returns:
        规范化后的文本。调用方仍然可以逐字回溯到原文 —— 规范化只动空白与不可见
        字符，不动任何可见字符。
    """
    text = _INVISIBLE.sub("", raw)
    text = _CONTROL.sub("", text)
    text = text.replace(_IDEOGRAPHIC_SPACE, " ")
    return _WHITESPACE.sub(" ", text).strip()


def _core_text(text: str) -> str:
    """取出文本的「实义部分」：去掉空白、标点与表情后剩下的字符。

    长度判定必须基于实义部分而不是原串，否则"😭😭😭😭😭😭"（6 个字符）会绕过
    长度下限混进分析流程 —— 它一个字的有效信息都没有。
    """
    return _NON_CORE.sub("", text)


def _is_low_value(core: str) -> bool:
    """实义文本是否由低价值短语拼成。

    不能只做等值比较：平台上的互动话术会连用（"蹲一个同款" / "我也是求同款"），
    等值比较会漏掉一半。这里用贪心切分，只有当整串能被短语表**恰好切完**时才判定
    为低价值 —— 真痛点句子里出现"我也是"（"我也是敏感肌一用就泛红"）不会被误伤。
    """
    if not core:
        return True
    if core in _LOW_VALUE_PHRASES:
        return True

    remaining = core
    while remaining:
        for phrase in _LOW_VALUE_BY_LENGTH:
            if remaining.startswith(phrase):
                remaining = remaining[len(phrase) :]
                break
        else:
            return False
    return True


def _is_noise(text: str, *, min_chars: int) -> bool:
    """:func:`is_noise` 的实现体，长度下限可调。

    单独抽出来是因为 :func:`build_units` 允许调用方覆盖 ``min_chars``：
    若让它去调公共的 :func:`is_noise`（内部写死 :data:`MIN_UNIT_CHARS`），
    传 ``min_chars=3`` 就永远留不住 3 个字的短句，参数的语义会被悄悄架空。
    """
    if not text or not text.strip():
        return True
    if _NO_TEXT.match(text):
        return True
    for pattern in _AD_PATTERNS:
        if pattern.search(text):
            return True

    core = _core_text(text)
    if _is_low_value(core):
        return True
    return len(core) < min_chars


def is_noise(text: str) -> bool:
    """判断文本是否为无分析价值的噪声。

    判定为噪声的情形：空文本、纯表情/标点、命中广告话术、属于低价值固定短语、
    去重后仍短于 :data:`MIN_UNIT_CHARS`。

    Args:
        text: 已规范化的文本。

    Returns:
        ``True`` 表示应丢弃。
    """
    return _is_noise(text, min_chars=MIN_UNIT_CHARS)


def compute_weights(likes: Sequence[int]) -> list[float]:
    """把一组点赞数压成 ``(0, 1]`` 的证据权重。

    用 ``log1p`` 压缩后再按最大值归一。直接用原始点赞数会让热门笔记碾压一切，
    使「痛点强度」因子退化成「哪篇笔记最火」；完全不用点赞数又会丢掉
    "高赞评论 = 更多人有同感"这一真实信号。

    Args:
        likes: 每条文本的点赞数。

    Returns:
        与输入等长的权重列表。全部为 0 时返回全 1.0（无信号时一视同仁，
        而不是全 0 —— 全 0 会让「痛点强度」因子整体塌陷）。
    """
    # 负数不是合法的点赞数，但上游（插件 / 手工构造的语料）可能给出 -1 之类的哨兵值。
    # math.log1p(-1) 会直接抛 ValueError 把整条流水线打断，夹到 0 更合理。
    compressed = [math.log1p(max(int(value), 0)) for value in likes]
    if not compressed:
        return []

    peak = max(compressed)
    if peak <= 0:
        # 全是 0 赞：没有任何区分信号，一视同仁
        return [1.0] * len(compressed)

    return [max(value / peak, _MIN_WEIGHT) for value in compressed]


def _truth_label(extra: dict[str, object]) -> str:
    """从 ``extra`` 里取出 fixture 语料写入的真实痛点标签。

    非字符串一律当作"没有标注"：这个字段是验收脚本的输入，一个类型不对的值
    （如 ``None`` / 数字）如果被 ``str()`` 转成 ``"None"``，会在聚类质量评估里
    凭空多出一个叫 "None" 的痛点类别。
    """
    value = extra.get("truth_label", "")
    return value if isinstance(value, str) else ""


def _note_text(title: str, desc: str) -> str:
    """把标题与正文拼成一条单元。

    标题往往是痛点的浓缩表达（"假白到像糊了面粉"），正文才是展开，两者必须同时
    进入 embedding；但也不能拆成两条单元 —— 那会让同一篇笔记在簇里被计两次。
    """
    parts = [part for part in (normalize_text(title), normalize_text(desc)) if part]
    return " ".join(parts)


def _clip(text: str, *, max_chars: int, min_chars: int) -> str | None:
    """噪声判定 + 超长截断。返回 ``None`` 表示这条文本应当丢弃。

    先判噪声再截断：广告话术常常挂在长文末尾，先截断会把广告标记切掉，
    让一条纯广告长文混进分析流程。
    """
    if _is_noise(text, min_chars=min_chars):
        return None
    clipped = text[:max_chars] if max_chars > 0 else ""
    return clipped or None


def build_units(
    corpus: RawCorpus,
    *,
    min_chars: int = MIN_UNIT_CHARS,
    max_chars: int = MAX_UNIT_CHARS,
    max_comments_per_note: int | None = None,
) -> list[TextUnit]:
    """把原始语料切成待分析的文本单元。

    笔记正文与评论都会被切出来，同等地进入后续阶段。笔记的标题与正文会拼接成
    一条单元（标题往往就是痛点的浓缩表达），图片地址挂在笔记单元上供 VLM 使用。

    Args:
        corpus: 采集到的原始语料。
        min_chars: 短于该长度的文本丢弃。
        max_chars: 超长文本截断到该长度。
        max_comments_per_note: 每篇笔记最多取多少条评论。``None`` 表示不限制。

    Returns:
        文本单元列表，保持稳定顺序（笔记按原顺序，每篇笔记的评论紧随其后）。
        **稳定顺序很重要**：聚类结果要靠下标与单元对应，顺序抖动会让证据挂错簇。

    Note:
        fixture 语料里的 ``truth_label`` 必须原样传递到 :class:`TextUnit` ——
        它是验收门③「频次误差」能自动计算的前提。
    """
    # 只按 note_id 归并一次评论，避免笔记数 × 评论数 的嵌套扫描
    comments_by_note: dict[str, list[RawComment]] = {}
    for comment in corpus.comments:
        comments_by_note.setdefault(comment.note_id, []).append(comment)

    units: list[TextUnit] = []
    likes: list[int] = []

    for note in corpus.notes:
        note_hash = hash_id(note.note_id)

        note_text = _clip(
            _note_text(note.title, note.desc), max_chars=max_chars, min_chars=min_chars
        )
        if note_text is not None:
            units.append(
                TextUnit(
                    text=note_text,
                    source="note",
                    likes=note.likes,
                    note_id=note.note_id,
                    note_hash=note_hash,
                    created_at=note.publish_time,
                    images=list(note.images),
                    truth_label=_truth_label(note.extra),
                )
            )
            likes.append(note.likes)

        kept = 0
        for comment in comments_by_note.get(note.note_id, ()):
            if max_comments_per_note is not None and kept >= max_comments_per_note:
                break
            comment_text = _clip(
                normalize_text(comment.content), max_chars=max_chars, min_chars=min_chars
            )
            if comment_text is None:
                continue
            # 上限作用于**保留下来的**评论数而不是扫描过的评论数：
            # 这个参数的用途是限制下游 embedding / LLM 的处理量，
            # 按扫描计数会让一篇噪声评论特别多的笔记只产出零星几条单元。
            kept += 1
            units.append(
                TextUnit(
                    text=comment_text,
                    source="comment",
                    likes=comment.likes,
                    note_id=note.note_id,
                    note_hash=note_hash,
                    created_at=comment.created_at,
                    images=[],
                    truth_label=_truth_label(comment.extra),
                )
            )
            likes.append(comment.likes)

    # 权重在一次调用里统一归一，让"哪条证据更有分量"在整份语料内可比。
    # 逐条调用 compute_weights 只会得到恒等于 1.0 的权重，点赞数这个信号就丢了。
    for unit, weight in zip(units, compute_weights(likes), strict=True):
        unit.weight = weight
    return units
