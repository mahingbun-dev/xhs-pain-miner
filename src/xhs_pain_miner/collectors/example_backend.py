"""采集后端插件模板 —— 复制这个文件，改成你自己的采集器。

使用步骤
--------
1. 复制本文件到你的工作目录，重命名为 ``my_xhs_backend.py``。
2. 把 :meth:`MyCollectorBackend.collect` 里的 TODO 换成调用**你自己合法持有**的采集器。
3. 按下面的映射表把平台字段填进 :class:`RawNote` / :class:`RawComment`。
4. 运行::

       export XHS_COLLECTOR_PLUGIN=/绝对路径/my_xhs_backend.py
       xhs-pain-miner collect -k 防晒霜 --backend plugin

字段映射表
----------
===============  ==========================  ==========================================
RawNote 字段      常见平台字段                 说明
===============  ==========================  ==========================================
``note_id``      note_id / id / noteId       **必填**，用于去重
``title``        title / display_title       笔记标题
``desc``         desc / content / note_text  正文。VLM 之外的主要分析对象
``url``          note_url / share_url        笔记链接（写进机会卡片的证据链）
``images``       image_list / images         **图片地址列表**，``--deep`` 时送给 VLM
``likes``        liked_count / likes         点赞数，用于证据权重
``collects``     collected_count / collects  收藏数
``comments_count`` comment_count             评论总数
``publish_time`` time / create_time          发布时间，用于趋势分析
``author_hash``  ——                           **必须哈希**，不要直接写 UID
===============  ==========================  ==========================================

================  =============================  ========================================
RawComment 字段    常见平台字段                   说明
================  =============================  ========================================
``comment_id``    comment_id / id                **必填**
``content``       content / text                 评论正文
``likes``         like_count / likes             点赞数
``parent_id``     parent_comment_id              二级评论的父评论 ID，一级评论填 None
``note_id``       所属笔记 ID                    用于把评论挂回笔记
``created_at``    create_time / time             评论时间
``user_hash``     ——                             **必须哈希**
================  =============================  ========================================

合规红线（请不要绕过）
----------------------
1. **个人信息最小化**：UID / 昵称 / 头像一律哈希或不采集。原文与个人信息是
   《个人信息保护法》与刑法 253 条之一的风险来源，而它们对你的分析毫无必要。
2. **不要绕过技术保护措施**：不要实现签名逆向、不要用 IP 池 / 账号池规避风控。
   这是《反不正当竞争法》2025 修订第 13 条第 3 款的构成要件。
3. **不要对外提供数据**：本工具的定位是本地分析，采集结果不要上传或转售。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from xhs_pain_miner.models import RawComment, RawCorpus, RawNote

# 换成你自己的采集器：
# from my_xhs_crawler import XhsClient


def _hash(value: str | None) -> str:
    """把平台侧标识转成不可逆哈希。

    直接复用库内的实现，保证与其它模块的哈希口径一致。
    """
    from xhs_pain_miner.models import hash_id

    return hash_id(value) if value else ""


class MyCollectorBackend:
    """你的采集后端 —— 实现 CollectorBackend 协议即可。

    协议只要求两个方法：:meth:`collect` 与 :meth:`available`，外加一个 ``name`` 属性。
    """

    name = "my-collector（自备采集器）"

    def __init__(self, cookie: str | None = None) -> None:
        self.cookie = cookie
        # TODO: 初始化你自己的客户端
        # self.client = XhsClient(cookie=cookie)

    def collect(
        self,
        keyword: str,
        *,
        limit: int,
        max_comments_per_note: int = 20,
    ) -> RawCorpus:
        """采集一个品类。

        约定：**失败时抛异常，不要返回空语料** —— 空语料会被下游误判为
        「这个品类没有痛点」，而实际上只是采集失败。

        .. note::
           本模板在 TODO 完成前会返回**空语料**。这不是该约定的例外，而是
           「尚未实现」的状态 —— 配套的 :meth:`available` 返回 ``False``，
           ``doctor`` 会据此提示后端尚未就绪。
        """
        notes: list[RawNote] = []
        comments: list[RawComment] = []

        # TODO: 换成真实调用
        # raw_notes = self.client.search(keyword, limit=limit)
        raw_notes: list[dict] = []

        for item in raw_notes:
            note_id = str(item.get("note_id", ""))
            notes.append(
                RawNote(
                    note_id=note_id,
                    title=item.get("title", ""),
                    desc=item.get("desc", ""),
                    url=item.get("note_url", ""),
                    images=list(item.get("image_list", [])),
                    likes=int(item.get("liked_count", 0)),
                    collects=int(item.get("collected_count", 0)),
                    comments_count=int(item.get("comment_count", 0)),
                    publish_time=_parse_time(item.get("time")),
                    author_hash=_hash(str(item.get("user_id", ""))),
                )
            )
            comments.extend(self._fetch_comments(note_id, max_comments_per_note))

        return RawCorpus(
            keyword=keyword,
            notes=notes,
            comments=comments,
            backend=self.name,
        )

    def _fetch_comments(self, note_id: str, limit: int) -> Iterator[RawComment]:
        """采集一篇笔记的评论（含二级评论）。"""
        # TODO: 换成真实调用
        raw_comments: list[dict] = []
        for item in raw_comments[:limit]:
            yield RawComment(
                comment_id=str(item.get("comment_id", "")),
                content=item.get("content", ""),
                likes=int(item.get("like_count", 0)),
                parent_id=item.get("parent_comment_id"),
                note_id=note_id,
                created_at=_parse_time(item.get("create_time")),
                user_hash=_hash(str(item.get("user_id", ""))),
            )

    def available(self) -> bool:
        """报告后端当前是否可用（供 ``doctor`` 调用，不应产生副作用）。"""
        # TODO: 换成真实的连通性检查，例如 self.client.ping()
        return False


def _parse_time(value: object) -> datetime | None:
    """把平台返回的时间戳 / 字符串解析成 datetime。

    常见形态有三种：秒级时间戳、毫秒级时间戳、ISO 字符串。按需扩展。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# 插件入口：模块级的 BACKEND 变量（也支持小写 backend 或 create_backend() 工厂）
BACKEND = MyCollectorBackend()
