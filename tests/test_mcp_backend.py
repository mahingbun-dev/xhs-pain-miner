"""MCP 采集后端测试（对接 ``xiaohongshu-mcp``）。

**不联网**：所有 HTTP 请求都走注入的 ``httpx.MockTransport``（假传输层），
沿用 ``tests/test_appstore.py`` 的同一套约定（autouse 的 ``no_real_network``
让任何漏装的真实请求直接失败，``install_transport`` 负责注入）。

假响应**照抄上游**（``xpzouying/xiaohongshu-mcp``，Apache-2.0，main =
``aad2a3d2``），不写"看起来像"的形状 —— 本模块的全部价值就是猜对上游的形状，
拿一份自己编的假响应去测它，等于把上游契约换成自己的臆想：

* 成功走 ``respondSuccess`` → ``{success, data, message}``，状态码**恒 200**；
* 失败走 ``respondError`` → ``{error, code, details}``，状态码显式指定 4xx/5xx，
  **错误体里没有 ``success`` 键**；两者没有共同的键，所以"怎么判失败"只有
  状态码一条路（见 ``_Client.request`` 的 Note）；
* 详情是**双层 data**：``SuccessResponse.data`` 装着
  ``FeedDetailResponse{feed_id, data:{note, comments}}``。

几处量纲的依据（同样来自上游源码，不是推测）：

* 计数是**字符串**（``InteractInfo.LikedCount string``），值是站点展示文案；
* ``note.time`` / ``comment.createTime`` 是裸 ``int64``
  （``FeedDetail.Time`` / ``Comment.CreateTime``），Go 侧原样透传、不换算，
  所以"单位是毫秒"只在 ``docs/API.md`` 里有依据；
* ``comments.list`` / ``note.imageList`` / ``subComments`` 都是无 ``omitempty``
  的切片字段 —— **键恒存在、值可能是 ``null``**。
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from xhs_pain_miner.cli import main as cli_main
from xhs_pain_miner.collectors import mcp
from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.collectors.mcp import MCPBackend
from xhs_pain_miner.config import Settings
from xhs_pain_miner.models import RawCorpus, hash_id

Handler = Callable[[httpx.Request], httpx.Response]

SEARCH_PATH = "/api/v1/feeds/search"
DETAIL_PATH = "/api/v1/feeds/detail"
LOGIN_PATH = "/api/v1/login/status"
HEALTH_PATH = "/health"

_UNSET = object()
"""区分"没传这个参数"与"显式传了 ``None``（即键存在、值为 null）"。"""


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class Recorder:
    """HTTP 处理器 + 请求记录（与 ``test_appstore.py`` 同名同类）。"""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def body(self, index: int = 0) -> Any:
        return json.loads(self.requests[index].read())


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """兜底：没有显式装上的请求一律在这里炸掉，宁可红也不要静默打真实网络。"""

    def forbid(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"测试不得发起真实网络请求：{request.url}")

    monkeypatch.setattr(mcp, "_transport", httpx.MockTransport(forbid))


@pytest.fixture
def install_transport(monkeypatch: pytest.MonkeyPatch):
    """把假传输层装进模块的注入点。"""

    def install(handler: Handler) -> Recorder:
        recorder = Recorder(handler)
        monkeypatch.setattr(mcp, "_transport", httpx.MockTransport(recorder))
        return recorder

    return install


def route(mapping: Mapping[str, Handler]) -> Handler:
    """按 URL 路径分发请求；未登记的路径直接断言失败。

    比"所有路径回同一份响应"严格：本模块有多个端点，若适配器把详情请求打到了
    搜索路径上，一份通用的假响应会让测试静默通过。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        target = mapping.get(request.url.path)
        if target is None:
            raise AssertionError(f"未预期的请求：{request.method} {request.url.path}")
        return target(request)

    return handler


# --------------------------------------------------------------------------- #
# 上游形状的假响应
# --------------------------------------------------------------------------- #


def success(data: object, message: str = "成功") -> httpx.Response:
    """上游 ``respondSuccess`` 的形状。"""
    return httpx.Response(200, json={"success": True, "data": data, "message": message})


def failure(status: int, *, code: str = "BOOM", error: str = "服务端出错了") -> httpx.Response:
    """上游 ``respondError`` 的形状 —— 注意**没有** ``success`` 键。"""
    return httpx.Response(status, json={"error": error, "code": code, "details": error})


def person(
    user_id: str = "U-1",
    *,
    nickname: str = "昵称",
    nick_name: str = "驼峰昵称",
    avatar: str = "https://cdn.example/avatar.jpg",
) -> dict[str, Any]:
    """上游 ``User`` 结构体的 JSON 形状。"""
    return {"userId": user_id, "nickname": nickname, "nickName": nick_name, "avatar": avatar}


def interact(
    liked: Any = "80",
    *,
    comment: Any = "12",
    collected: Any = "3",
) -> dict[str, Any]:
    """上游 ``InteractInfo``：三个计数都是**字符串**。"""
    return {
        "liked": False,
        "likedCount": liked,
        "sharedCount": "0",
        "commentCount": comment,
        "collectedCount": collected,
        "collected": False,
    }


def card(
    feed_id: str | None = "N1",
    *,
    xsec_token: str | None = "XSEC-N1",
    title: str = "列表页标题",
    model_type: str | None = "note",
    note_card: Any = None,
    user: Any = None,
    interact_info: Any = None,
) -> dict[str, Any]:
    """一条搜索卡片（上游 ``Feed`` 的 JSON 形状）。

    ``note_card=None`` 表示"这条没有 noteCard"（直播卡 / 热词就是这个形态）；
    ``xsec_token=None`` / ``model_type=None`` 表示**键缺失**，不是空值。
    """
    if note_card is None:
        note_card = {
            "type": "normal",
            "displayTitle": title,
            "user": user if user is not None else person(),
            "interactInfo": interact_info if interact_info is not None else interact(),
        }
    item: dict[str, Any] = {"id": feed_id, "modelType": model_type, "noteCard": note_card}
    if xsec_token is not None:
        item["xsecToken"] = xsec_token
    return item


def note(
    note_id: str = "N1",
    *,
    title: str = "详情标题",
    desc: Any = "详情正文",
    time: Any = 1700000000000,
    interact_info: Any = _UNSET,
    image_list: Any = _UNSET,
    user: Any = None,
    omit: Sequence[str] = (),
    **extra: Any,
) -> dict[str, Any]:
    """一篇详情笔记（上游 ``FeedDetail`` 的 JSON 形状）。

    ``omit`` 用来删键（模拟上游结构变更），与"值为空"是两件事 ——
    ``_require_keys`` 查的正是"键在不在"。``image_list=None`` 表示**键在、值为
    ``null``**（nil slice 的序列化结果），不是"用默认图片"。
    """
    payload: dict[str, Any] = {
        "noteId": note_id,
        "xsecToken": "XSEC-N1",
        "title": title,
        "desc": desc,
        "type": "normal",
        "time": time,
        "ipLocation": "上海",
        "user": user if user is not None else person(),
        "interactInfo": interact() if interact_info is _UNSET else interact_info,
        "imageList": [
            {
                "width": 1080,
                "height": 1440,
                "urlDefault": "https://img.example/a.jpg",
                "urlPre": "https://img.example/a-pre.jpg",
            }
        ]
        if image_list is _UNSET
        else image_list,
    }
    payload.update(extra)
    for key in omit:
        payload.pop(key, None)
    return payload


def comment(
    comment_id: str | None = "C1",
    *,
    content: Any = "评论正文",
    like: Any = "5",
    create_time: Any = 1700000001000,
    subs: Any = None,
    note_id: str | None = "N1",
    user: Any = None,
) -> dict[str, Any]:
    """一条评论（上游 ``Comment`` 的 JSON 形状）。"""
    payload: dict[str, Any] = {
        "id": comment_id,
        "noteId": note_id,
        "content": content,
        "likeCount": like,
        "createTime": create_time,
        "ipLocation": "上海",
        "liked": False,
        "userInfo": user if user is not None else person(),
        "subCommentCount": "0",
        "subComments": subs,
        "showTags": [],
    }
    if comment_id is None:
        payload.pop("id")
    return payload


def comments(*items: Any, list_value: Any = ...) -> Any:
    """评论容器的 JSON 形状；``list_value=...`` 时用 ``items``。"""
    return {
        "list": list(items) if list_value is ... else list_value,
        "cursor": "",
        "hasMore": False,
    }


def detail(
    note_obj: Any = None,
    comments_obj: Any = None,
    *,
    feed_id: str = "N1",
) -> httpx.Response:
    """详情响应：外层信封 + ``FeedDetailResponse{feed_id, data}``（**双层 data**）。"""
    return success(
        {
            "feed_id": feed_id,
            "data": {
                "note": note(note_id=feed_id) if note_obj is None else note_obj,
                "comments": comments_obj,
            },
        },
        "获取Feed详情成功",
    )


def healthy() -> httpx.Response:
    """``/health`` 的真实形状：公开端点 + 成功信封（``success`` 恒为 true）。"""
    return success({"status": "healthy", "service": "xiaohongshu-mcp"}, "服务正常")


def login_status(logged_in: Any = True) -> httpx.Response:
    """``/api/v1/login/status``：``data.is_logged_in``（snake_case）。"""
    return success({"is_logged_in": logged_in}, "检查登录状态成功")


def backend(**kwargs: Any) -> MCPBackend:
    return MCPBackend(**kwargs)


def detail_router(
    feeds: Sequence[Any],
    *,
    fail_ids: Sequence[str] = (),
    comments_obj: Any = None,
) -> Handler:
    """搜索返回 ``feeds``；详情对 ``fail_ids`` 里的笔记回 500，其余正常。

    带一个 ``login/status``：详情 500 会触发「是不是掉登录了」的判据（见
    ``_http_error_message``），漏了它这个路由会直接断言失败 —— 那正是我们要的
    效果（未预期的请求不许静默通过）。
    """

    def on_detail(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if body["feed_id"] in fail_ids:
            return failure(500, code="GET_FEED_DETAIL_FAILED", error="获取Feed详情失败")
        return detail(
            note_obj=note(str(body["feed_id"]), **{"xsecToken": body["xsec_token"]}),
            comments_obj=comments() if comments_obj is None else comments_obj,
            feed_id=str(body["feed_id"]),
        )

    return route(
        {
            SEARCH_PATH: lambda request: success({"feeds": list(feeds), "count": len(feeds)}),
            DETAIL_PATH: on_detail,
            LOGIN_PATH: lambda request: login_status(False),
        }
    )


# --------------------------------------------------------------------------- #
# 搜索 → 卡片筛选
# --------------------------------------------------------------------------- #


class TestSearchSelection:
    """搜索响应 → 待取详情的卡片列表。"""

    def test_searches_with_keyword_param(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        backend().collect("防晒霜", limit=5)

        request = recorder.requests[0]
        assert request.method == "GET"
        assert request.url.path == SEARCH_PATH
        assert request.url.params["keyword"] == "防晒霜"

    def test_non_note_cards_are_dropped(self, install_transport):
        """直播卡（live_v2）与热词（hot_query）没有 noteCard，混进来会占掉 limit 名额。"""
        recorder = install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {
                            "feeds": [
                                card("N1"),
                                card("LIVE-1", model_type="live_v2", note_card=None),
                                card("HOT-1", model_type="hot_query", note_card=None),
                                card("N2"),
                            ],
                            "count": 4,
                        }
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note(str(json.loads(request.read())["feed_id"])),
                        comments_obj=comments(),
                        feed_id=str(json.loads(request.read())["feed_id"]),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=2)

        assert [n.note_id for n in corpus.notes] == ["N1", "N2"]
        assert recorder.paths == [SEARCH_PATH, DETAIL_PATH, DETAIL_PATH]

    def test_card_with_non_object_note_card_is_dropped(self, install_transport):
        """``noteCard`` 不是对象时连展示标题都拿不到 —— 当噪音丢掉，不占名额。"""
        recorder = install_transport(detail_router([card("N1", note_card="不是对象"), card("N2")]))
        corpus = backend().collect("防晒霜", limit=5)

        assert [n.note_id for n in corpus.notes] == ["N2"]
        assert recorder.paths == [SEARCH_PATH, DETAIL_PATH]

    def test_missing_model_type_is_tolerated(self, install_transport):
        """上游若改了过滤口径（整个 ``modelType`` 键消失），不能把这一页笔记全丢掉。"""
        install_transport(detail_router([card("N1", model_type=None)]))
        assert len(backend().collect("防晒霜", limit=5).notes) == 1


# --------------------------------------------------------------------------- #
# 详情 → 笔记
# --------------------------------------------------------------------------- #


class TestNoteMapping:
    """详情响应 → ``RawNote``。"""

    def test_maps_detail_fields(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note(
                            "N1",
                            title="防晒搓泥怎么办",
                            desc="上脸假白，搓泥严重",
                            time=1700000000000,
                            interact_info=interact("1.2万", comment="34", collected="7"),
                            image_list=[
                                {"urlDefault": "https://img.example/1.jpg", "urlPre": "u-p-1"},
                                {"urlPre": "https://img.example/2p.jpg"},
                                {"urlDefault": "", "urlPre": ""},
                            ],
                            user=person("U-9"),
                        ),
                        comments_obj=comments(),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert len(corpus.notes) == 1
        item = corpus.notes[0]
        assert item.note_id == "N1"
        assert item.title == "防晒搓泥怎么办"
        assert item.desc == "上脸假白，搓泥严重"
        assert item.likes == 12000
        assert item.comments_count == 34
        assert item.collects == 7
        assert item.publish_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        assert item.author_hash == hash_id("U-9")
        # 两张有效图：urlDefault 优先，缺了回退 urlPre，两者都空的那条丢掉。
        assert item.images == ["https://img.example/1.jpg", "https://img.example/2p.jpg"]
        assert item.extra == {}

    def test_url_carries_xsec_token(self, install_transport):
        """xsec_token 是详情页能不能打开的前提，不是装饰参数。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", xsec_token="TOKEN-A")], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", **{"xsecToken": "TOKEN-A"}), comments_obj=comments()
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes[0].url == (
            "https://www.xiaohongshu.com/explore/N1?xsec_token=TOKEN-A&xsec_source=pc_feed"
        )

    def test_title_falls_back_to_card_when_detail_is_empty(self, install_transport):
        """详情标题为空时用卡片的 ``displayTitle`` 兜底（列表页的截断标题也比空好）。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", title="列表页截断标题")], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", title="", desc="有正文"), comments_obj=comments()
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].title == "列表页截断标题"

    def test_card_interact_info_is_used_when_detail_lacks_it(self, install_transport):
        """详情缺 interactInfo 时回退到搜索卡片上的那个，而不是把计数归零。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", interact_info=interact("99"))], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj={
                            "noteId": "N1",
                            "title": "t",
                            "desc": "d",
                            "interactInfo": "不是对象",
                            "imageList": [],
                        },
                        comments_obj=comments(),
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].likes == 99

    def test_note_id_falls_back_to_card_id(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj={
                            "noteId": "",
                            "title": "t",
                            "desc": "d",
                            "interactInfo": {},
                            "imageList": [],
                        },
                        comments_obj=comments(),
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].note_id == "N1"

    def test_image_list_prefers_url_default(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note(
                            "N1",
                            image_list=[
                                {"urlDefault": "D", "urlPre": "P"},
                                {"urlPreview": "别的键"},
                                "不是对象",
                            ],
                        ),
                        comments_obj=comments(),
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].images == ["D"]

    def test_no_images_yields_empty_list(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", image_list=[]), comments_obj=comments()
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes[0].images == []
        assert corpus.total_images == 0


# --------------------------------------------------------------------------- #
# 评论树 → RawComment
# --------------------------------------------------------------------------- #


class TestCommentMapping:
    """详情里的评论树摊平成 ``RawComment`` 列表。"""

    def test_flattens_two_levels_in_order(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment(
                                "C1",
                                content="一级甲",
                                create_time=1700000001000,
                                like="7",
                                subs=[comment("C1-1", content="二级甲", note_id="N1")],
                            ),
                            comment("C2", content="一级乙"),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["C1", "C1-1", "C2"]
        assert [c.content for c in corpus.comments] == ["一级甲", "二级甲", "一级乙"]

    def test_parent_id_only_on_second_level(self, install_transport):
        """上游没有 ``parentId`` 字段，父子关系只能由嵌套推断。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", subs=[comment("C1-1"), comment("C1-2")]),
                            comment("C2"),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        by_id = {c.comment_id: c for c in corpus.comments}
        assert by_id["C1"].parent_id is None
        assert by_id["C2"].parent_id is None
        assert by_id["C1-1"].parent_id == "C1"
        assert by_id["C1-2"].parent_id == "C1"

    def test_note_id_attribution(self, install_transport):
        """一二级评论都要归到同一篇笔记 —— clean.py 按 ``note_id`` 归并评论。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", subs=[comment("C1-1", subs=[comment("C1-1-1")])])
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert {c.note_id for c in corpus.comments} == {corpus.notes[0].note_id} == {"N1"}

    def test_empty_content_comments_are_dropped(self, install_transport):
        """空正文（图片评论）对分析无价值；有正文的那条必须留下。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", content="   "), comment("C2", content="有内容")
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["C2"]

    def test_dropping_a_parent_keeps_its_replies(self, install_transport):
        """父评论因正文为空被丢时，它的**子回复仍然要进语料**。

        图片式一级评论（正文为空）底下常挂着真正的痛点发言。此前实现用的是
        ``if parent is None: continue``，会把整棵子树一起吞掉，且不出声。
        """
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment(None, content="", subs=[comment("S1", content="子回复有内容")]),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.content for c in corpus.comments] == ["子回复有内容"]
        # parent_id 指向那个**没进语料**的合成 id —— 这是刻意的：父子关系不能因为
        # 父被丢掉就假装它不存在（渲染层据此能标出"这条是对某条已过滤评论的回复"）。
        assert corpus.comments[0].parent_id == "N1#c0"
        assert corpus.comments[0].parent_id not in {c.comment_id for c in corpus.comments}

    def test_dropping_a_parent_keeps_replies_for_every_parent(self, install_transport):
        """多个父都空正文时，各自的子回复都要留下，且 parent_id 各归各的。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment(None, content="", subs=[comment(None, content="甲回复")]),
                            comment(None, content="", subs=[comment(None, content="乙回复")]),
                            comment(
                                "C2", content="父有正文", subs=[comment(None, content="丙回复")]
                            ),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.content for c in corpus.comments] == ["甲回复", "乙回复", "父有正文", "丙回复"]
        assert [c.parent_id for c in corpus.comments] == ["N1#c0", "N1#c1", None, "C2"]

    def test_missing_comments_object_is_empty(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(note_obj=note("N1"), comments_obj=None),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.comments == []
        assert len(corpus.notes) == 1

    def test_comment_counts_are_parsed(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", like="1.5k", create_time=1700000001000)
                        ),
                    ),
                }
            )
        )
        item = backend().collect("防晒霜", limit=5, max_comments_per_note=20).comments[0]

        assert item.likes == 1500
        assert item.created_at == datetime(2023, 11, 14, 22, 13, 21, tzinfo=timezone.utc)


class TestCommentLimit:
    """``max_comments_per_note`` 的截断语义。"""

    @pytest.mark.parametrize(
        ("limit", "expected"),
        [(1, 1), (3, 3), (5, 5), (11, 11), (20, 12)],
    )
    def test_total_is_capped(self, install_transport, limit: int, expected: int):
        """上限按**摊平后**的总数算：一条带 10 条回复的评论不能把单篇顶穿。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", subs=[comment(f"C1-{i}") for i in range(10)]),
                            comment("C2"),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=limit)

        assert len(corpus.comments) == expected
        assert corpus.comments[0].comment_id == "C1"

    def test_zero_means_no_comments(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(comment("C1"))
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=0)

        assert corpus.comments == []
        assert len(corpus.notes) == 1

    def test_order_is_parent_then_its_children(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("A", subs=[comment("A1"), comment("A2")]),
                            comment("B", subs=[comment("B1")]),
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["A", "A1", "A2", "B", "B1"]

    def test_truncation_can_cut_between_a_parent_and_its_child(self, install_transport):
        """截断的边界：限额按摊平后的总数算，所以**保留的父**的子回复也可能被切掉。

        这不是"丢父连子树"那个缺陷（那条说的是父被丢弃的情形），而是限额本身的
        语义 —— 记下来是为了让这条边界是"已知且被固定的"，不是随时会变的巧合。
        """
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(comment("A", subs=[comment("A1")])),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=1)

        assert [c.comment_id for c in corpus.comments] == ["A"]


# --------------------------------------------------------------------------- #
# 计数字符串
# --------------------------------------------------------------------------- #


class TestCounts:
    """计数是站点展示文案，不是数字。

    每个期望值的依据：``_COUNT_MULTIPLIERS`` 的倍率表 + ``_parse_count`` 的
    "解析不出来一律 0，不抛异常"约定。``"1,234"`` 与 ``"10万+"`` 先去掉千分位与
    后缀**再**匹配（见 ``text.replace``），所以``_COUNT_RE`` 里没有 ``,`` 与 ``+``。
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("80", 80),  # 纯数字文案
            ("1.2万", 12000),  # 中文万
            ("3.4w", 34000),  # 小写 w
            ("1.5k", 1500),  # 千
            ("10万+", 100000),  # 后缀 + 先被去掉
            ("1,234", 1234),  # 千分位先被去掉
            ("1.2亿", 120000000),  # 亿在倍率表里
            ("0.5万", 5000),  # 小数倍率走浮点乘，不是整数截断
            ("0", 0),
            ("", 0),  # 空串
            ("   ", 0),  # 只有空白
            (None, 0),  # 键在、值为空
            ("赞", 0),  # 文案里没有数字
            ("-5", 0),  # 负数字符串不匹配正则 → 0，不会变成 -5
            (True, 0),  # bool 是 int 子类，但在这里一定是数据错误
            (False, 0),
            (3, 3),  # 上游若给数字也认
            (3.7, 3),  # 浮点截断
            (-5, 0),  # 负数钳到 0
            ({"a": 1}, 0),  # 不是标量
            (["1"], 0),
        ],
    )
    def test_value_table(self, raw: Any, expected: int):
        assert mcp._parse_count(raw) == expected

    @pytest.mark.parametrize("raw", [float("inf"), float("-inf"), float("nan")])
    def test_infinite_and_nan_return_zero(self, raw: float):
        """ "解析不出来的一律返回 0，不抛异常"要真的成立：``int(inf)``/``int(nan)`` 会抛。

        Go 的 ``json.Marshal`` 拒绝 Inf/NaN，所以这三个输入**经真实上游不可达**；
        但这条契约不该依赖调用方那边的序列化器来兜（数值分支的 try/except 就是
        为此加的）。
        """
        assert mcp._parse_count(raw) == 0

    def test_wan_reaches_the_note(self, install_transport):
        """表驱动之外再从公开路径验一次：解析结果确实进了 ``RawNote``。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note(
                            "N1", interact_info=interact("1.2万", comment="赞", collected="")
                        ),
                        comments_obj=comments(),
                    ),
                }
            )
        )
        item = backend().collect("防晒霜", limit=5).notes[0]

        assert item.likes == 12000
        assert item.comments_count == 0
        assert item.collects == 0


# --------------------------------------------------------------------------- #
# 时间戳
# --------------------------------------------------------------------------- #


class TestTimestamps:
    """时间戳解析：只按"秒还是毫秒"分支，产出**恒为 aware UTC**。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (1700000000000, datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)),  # 毫秒
            (1700000000, datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)),  # 秒级兜底
            (0, None),  # 上游用 0 表示"没有这个时间"
            (None, None),
            (-1, None),
            ("1700000000000", None),  # 字符串一律不认
            ("昨天", None),
            (True, None),
            (10**20, None),  # 超出 datetime 范围 → None，不抛
        ],
    )
    def test_value_table(self, raw: Any, expected: datetime | None):
        assert mcp._parse_timestamp(raw) == expected

    @pytest.mark.parametrize("raw", [1700000000000, 1700000000, 1700000000123])
    def test_result_is_timezone_aware(self, raw: int):
        """naive 与 aware 混在同一个列表里做 min/max 会抛 TypeError（趋势因子）。"""
        parsed = mcp._parse_timestamp(raw)

        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_note_time_is_aware(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", time=1700000000000), comments_obj=comments()
                    ),
                }
            )
        )
        item = backend().collect("防晒霜", limit=5).notes[0]

        assert item.publish_time is not None
        assert item.publish_time.tzinfo is not None

    def test_zero_time_becomes_none_not_1970(self, install_transport):
        """把 0 当成 1970 会让趋势因子把几十条证据全算成"很久以前"。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", time=0),
                        comments_obj=comments(comment("C1", create_time=0)),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert corpus.notes[0].publish_time is None
        assert corpus.comments[0].created_at is None


# --------------------------------------------------------------------------- #
# 结构不符 / null
# --------------------------------------------------------------------------- #


class TestStructureErrors:
    """结构不符必须报错，不能静默产出缩水语料。"""

    def test_missing_desc_key_raises(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", omit=("desc",)), comments_obj=comments()
                    ),
                }
            )
        )
        with pytest.raises(CollectorError, match="desc"):
            backend().collect("防晒霜", limit=5)

    def test_empty_desc_is_fine(self, install_transport):
        """图文笔记可以没有正文 —— 空值与"键消失"必须分开。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", desc=""), comments_obj=comments()
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].desc == ""

    @pytest.mark.parametrize("key", ["title", "desc", "interactInfo", "imageList"])
    def test_each_required_key_is_checked(self, install_transport, key: str):
        """``_NOTE_REQUIRED_KEYS`` 的每一项单独缺失都要被指名报出来。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", omit=(key,)), comments_obj=comments()
                    ),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert key in message
        assert "data.data.note" in message

    def test_missing_image_list_would_zero_the_images(self, install_transport):
        """``imageList`` 进必检名单的理由：不查的话图片**静默归零**、``--deep`` 空转。

        这条用"同一份响应、键在但值为空数组"做对照 —— 那才是合法的"这篇没有图片"。
        """
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", image_list=[]), comments_obj=comments()
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes[0].images == []
        assert corpus.total_images == 0

    @pytest.mark.parametrize("inner", [None, "不是对象", 3, [1]])
    def test_missing_inner_data_raises(self, install_transport, inner: Any):
        """``data`` 底下又套了一层 ``data``；少剥一层必须是显式错误。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: success({"feed_id": "N1", "data": inner}),
                }
            )
        )
        with pytest.raises(CollectorError, match="data.data"):
            backend().collect("防晒霜", limit=5)

    @pytest.mark.parametrize("bad", [{"0": "不是数组"}, "list", 3, True])
    def test_comments_list_must_be_array(self, install_transport, bad: Any):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(list_value=bad)
                    ),
                }
            )
        )
        with pytest.raises(CollectorError, match="comments.list"):
            backend().collect("防晒霜", limit=5)

    def test_comments_object_must_be_mapping(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=["不是对象"]
                    ),
                }
            )
        )
        with pytest.raises(CollectorError, match="data.data.comments"):
            backend().collect("防晒霜", limit=5)

    def test_missing_comments_list_key_raises(self, install_transport):
        """``list`` 改名时不查就会**静默返回 0 条评论**（看起来像"这篇没评论"）。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj={"cursor": "", "hasMore": False}
                    ),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        assert "list" in str(exc_info.value)
        assert "data.data.comments" in str(exc_info.value)

    @pytest.mark.parametrize("level", ["parent", "sub"])
    def test_missing_comment_content_key_raises(self, install_transport, level: str):
        """``content`` 改名时不查就会**整批评论静默消失**（空正文评论被全部丢掉）。

        后果不是"少几条"，是语料看起来像"这篇笔记压根没有评论" —— 而下游所有
        「提及次数」都建立在这批文本上，所以必须报错而不是当成空正文。
        """
        broken = comment("C1")
        broken.pop("content")
        if level == "parent":
            payload = comments(broken)
            expected_path = "comments.list[0]"
        else:
            payload = comments(comment("C1", subs=[broken]))
            expected_path = "comments.list[0].subComments[0]"
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(note_obj=note("N1"), comments_obj=payload),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        message = str(exc_info.value)
        assert "content" in message
        assert expected_path in message

    def test_empty_comment_content_is_still_dropped_not_an_error(self, install_transport):
        """与上一条互为对照：键**在**、值为空串是正常数据（图片式评论），不是结构变更。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(
                            comment("C1", content=""), comment("C2", content="有")
                        ),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["C2"]

    def test_missing_sub_comments_key_raises(self, install_transport):
        """``subComments`` 改名时不查就会**静默丢掉全部二级评论**。"""
        broken = comment("C1")
        broken.pop("subComments")
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(broken, comment("C2"))
                    ),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        message = str(exc_info.value)
        assert "subComments" in message
        assert "comments.list[0]" in message

    def test_sub_comments_key_is_required_on_every_parent(self, install_transport):
        """逐条一级评论都要有 —— 不能只看第一条。"""
        broken = comment("C2")
        broken.pop("subComments")
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(comment("C1"), broken)
                    ),
                }
            )
        )
        with pytest.raises(CollectorError, match=r"comments\.list\[1\]"):
            backend().collect("防晒霜", limit=5, max_comments_per_note=20)

    def test_junk_entries_are_skipped_not_fatal(self, install_transport):
        """列表里混进非对象条目仍按"跳过"处理 —— 新增的键校验不该把噪音升级成致命错误。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments("不是对象", comment("C1", subs=["也不是对象"])),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["C1"]

    def test_sub_comments_must_be_array(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"),
                        comments_obj=comments(comment("C1", subs="不是数组")),
                    ),
                }
            )
        )
        with pytest.raises(CollectorError, match="subComments"):
            backend().collect("防晒霜", limit=5, max_comments_per_note=20)

    @pytest.mark.parametrize("bad", [{"feeds": "不是数组"}, {"feeds": 3}])
    def test_feeds_must_be_array(self, install_transport, bad: Any):
        install_transport(route({SEARCH_PATH: lambda request: success(bad)}))
        with pytest.raises(CollectorError, match="data.feeds"):
            backend().collect("防晒霜", limit=5)

    def test_missing_feeds_key_raises(self, install_transport):
        """``feeds`` 键整个消失 = 结构变更，必须报错。

        上游 ``onlyNotes()`` 用 ``make([]Feed, 0, ...)`` 构造，`feeds` 至少是 ``[]``
        而不会是 null（``FeedsListResponse.Feeds`` 的 json tag 也没有 omitempty），
        所以"键不见了"只可能是契约变了 —— 若当成"没有结果"，用户会拿着一个
        **适配器缺陷**去反复换关键词（见 ``_search`` 里那段注释）。
        """
        install_transport(route({SEARCH_PATH: lambda request: success({"count": 0})}))
        with pytest.raises(CollectorError, match="feeds"):
            backend().collect("防晒霜", limit=5)

    def test_null_feeds_is_still_an_empty_page(self, install_transport):
        """与上一条互为对照：键**在**、值为 null 是"没有内容"，不是结构变更。"""
        install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": None, "count": 0})})
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes == []
        assert len(corpus.warnings) == 1
        assert "换关键词" in corpus.warnings[0]

    @pytest.mark.parametrize("body", ["<html>oops</html>", ""])
    def test_non_json_body_raises(self, install_transport, body: str):
        install_transport(route({SEARCH_PATH: lambda request: httpx.Response(200, text=body)}))
        with pytest.raises(CollectorError, match="不是合法 JSON"):
            backend().collect("防晒霜", limit=5)

    @pytest.mark.parametrize("payload", [[1, 2, 3], "字符串", 3])
    def test_non_object_json_raises(self, install_transport, payload: Any):
        install_transport(route({SEARCH_PATH: lambda request: httpx.Response(200, json=payload)}))
        with pytest.raises(CollectorError, match="JSON 对象"):
            backend().collect("防晒霜", limit=5)

    def test_null_body_raises(self, install_transport):
        """字面量 ``null`` 是合法 JSON，但不是对象 —— 两个分支都不能漏。"""
        install_transport(route({SEARCH_PATH: lambda request: httpx.Response(200, text="null")}))
        with pytest.raises(CollectorError, match="JSON 对象"):
            backend().collect("防晒霜", limit=5)

    def test_200_without_success_flag_raises(self, install_transport):
        """200 但没有 ``success: true`` 说明地址指到了别的服务上。"""
        install_transport(
            route({SEARCH_PATH: lambda request: httpx.Response(200, json={"result": "别的服务"})})
        )
        with pytest.raises(CollectorError, match="XHS_MCP_URL"):
            backend().collect("防晒霜", limit=5)

    def test_missing_xsec_token_raises_per_note(self, install_transport):
        """没有 xsecToken 就打不开详情页 —— 报错要说清楚是"这篇缺 token"。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", xsec_token=None)], "count": 1}
                    )
                }
            )
        )
        with pytest.raises(CollectorError, match="xsecToken"):
            backend().collect("防晒霜", limit=5)


class TestNulls:
    """``null`` 是"没有内容"，不是"结构变了"。"""

    def test_null_comment_list(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(list_value=None)
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).comments == []

    def test_null_image_list(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", image_list=None), comments_obj=comments()
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].images == []

    def test_null_sub_comments(self, install_transport):
        """``subComments`` 键恒存在、值可能是 ``null``（nil slice 序列化成 null）。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(comment("C1", subs=None))
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["C1"]
        assert corpus.comments[0].parent_id is None

    def test_null_feeds(self, install_transport):
        install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": None, "count": 0})})
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes == []


# --------------------------------------------------------------------------- #
# 失败语义
# --------------------------------------------------------------------------- #


class TestFailureSemantics:
    """单篇失败跳过并计数；过半失败必须中止整次采集。"""

    def test_single_failure_is_skipped_and_reported(self, install_transport):
        # limit 取 3（= 搜索结果条数）：否则还会多一条"要的比拿到的多"的告警，
        # 这条用例只关心失败那一条。
        install_transport(detail_router([card("N1"), card("N2"), card("N3")], fail_ids=["N2"]))
        corpus = backend().collect("防晒霜", limit=3)

        assert [n.note_id for n in corpus.notes] == ["N1", "N3"]
        assert len(corpus.warnings) == 1
        assert "1/3" in corpus.warnings[0]
        assert "N2" in corpus.warnings[0]

    def test_exactly_half_does_not_abort(self, install_transport):
        """边界：``len(failures) * 2 > len(selected)`` —— 恰好一半不抛。"""
        install_transport(detail_router([card("N1"), card("N2")], fail_ids=["N2"]))
        corpus = backend().collect("防晒霜", limit=2)

        assert [n.note_id for n in corpus.notes] == ["N1"]
        assert "1/2" in corpus.warnings[0]

    def test_two_thirds_aborts(self, install_transport):
        """边界：3 篇里 2 篇失败 = 66% > 50%，抛。"""
        install_transport(
            detail_router([card("N1"), card("N2"), card("N3")], fail_ids=["N2", "N3"])
        )
        with pytest.raises(CollectorError, match="2 篇取详情失败"):
            backend().collect("防晒霜", limit=5)

    def test_all_failed_aborts(self, install_transport):
        install_transport(detail_router([card("N1"), card("N2")], fail_ids=["N1", "N2"]))
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "2 篇取详情失败" in message
        assert "第一条失败的原因" in message
        assert "N1" in message

    def test_failed_note_produces_no_partial_note(self, install_transport):
        """失败的那篇不能留下半条笔记（否则正文与评论会各说各话）。"""
        install_transport(detail_router([card("N1"), card("N2")], fail_ids=["N1"]))
        corpus = backend().collect("防晒霜", limit=5)

        assert [n.note_id for n in corpus.notes] == ["N2"]

    def test_failure_ratio_counts_notes_not_requests(self, install_transport):
        """分母是**选中的笔记数**，不是发出的请求数 —— 这里两者相同，故用 limit 验证。"""
        install_transport(detail_router([card(f"N{i}") for i in range(4)], fail_ids=["N3"]))
        corpus = backend().collect("防晒霜", limit=4)

        assert len(corpus.notes) == 3
        assert "1/4" in corpus.warnings[0]

    def test_empty_result_is_not_a_failure(self, install_transport):
        """搜不到结果是正常情况（不是失败），但要在 warnings 里说明。"""
        install_transport(route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})}))
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes == []
        assert len(corpus.warnings) == 1
        assert "0 篇" in corpus.warnings[0]

    @pytest.mark.parametrize("keyword", ["", "   ", "\n\t"])
    def test_empty_keyword_raises_before_any_request(self, install_transport, keyword: str):
        recorder = install_transport(route({SEARCH_PATH: lambda request: success({"feeds": []})}))
        with pytest.raises(CollectorError, match="搜索词为空"):
            backend().collect(keyword, limit=5)

        assert recorder.requests == []


class TestHttpErrors:
    """连接、超时、鉴权、非 200。"""

    def test_connect_error_says_start_the_service(self, install_transport):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(boom)
        with pytest.raises(CollectorError, match="无法连接"):
            backend().collect("防晒霜", limit=5)

    def test_timeout_names_the_knob(self, install_transport):
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("超时")

        install_transport(slow)
        with pytest.raises(CollectorError, match="XHS_MCP_TIMEOUT"):
            backend().collect("防晒霜", limit=5)

    def test_401_points_at_the_token(self, install_transport):
        install_transport(
            route({SEARCH_PATH: lambda request: failure(401, code="UNAUTHORIZED", error="未授权")})
        )
        with pytest.raises(CollectorError, match="XHS_MCP_TOKEN"):
            backend().collect("防晒霜", limit=5)

    @pytest.mark.parametrize("status", [400, 403, 404, 405, 502])
    def test_non_200_raises(self, install_transport, status: int):
        # 5xx 会去问一次 login/status（判"是不是掉登录了"），路由必须给出这一条。
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(status),
                    LOGIN_PATH: lambda request: login_status(True),
                }
            )
        )
        with pytest.raises(CollectorError, match=str(status)):
            backend().collect("防晒霜", limit=5)

    def test_plain_text_404_is_reported_with_its_body(self, install_transport):
        """路径打错时上游返回 ``text/plain`` 的 404（gin 没注册 NoRoute）。"""
        install_transport(
            route({SEARCH_PATH: lambda request: httpx.Response(404, text="404 page not found")})
        )
        with pytest.raises(CollectorError, match="404 page not found"):
            backend().collect("防晒霜", limit=5)

    def test_plain_text_405_is_reported(self, install_transport):
        install_transport(
            route({SEARCH_PATH: lambda request: httpx.Response(405, text="Method Not Allowed")})
        )
        with pytest.raises(CollectorError, match="Method Not Allowed"):
            backend().collect("防晒霜", limit=5)

    def test_bearer_token_is_sent_when_configured(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        backend(token="secret").collect("防晒霜", limit=5)

        assert recorder.requests[0].headers["Authorization"] == "Bearer secret"

    def test_no_authorization_header_without_token(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        backend().collect("防晒霜", limit=5)

        assert "Authorization" not in recorder.requests[0].headers

    def test_base_url_trailing_slash_is_normalized(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        collector = backend(base_url="http://127.0.0.1:18060/")
        collector.collect("防晒霜", limit=5)

        assert recorder.requests[0].url.host == "127.0.0.1"
        assert collector.base_url == "http://127.0.0.1:18060"

    def test_timeout_reaches_the_transport(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        backend(timeout=7.5).collect("防晒霜", limit=5)

        assert recorder.requests[0].extensions["timeout"]["read"] == 7.5


class TestFiveHundredBranches:
    """5xx 的两种分岔：掉登录 vs 服务/风控 —— 指引方向相反，文案必须不同。

    上游的 ``SearchFeeds`` / ``GetFeedDetailWithConfig`` 都没有登录前置检查，
    掉登录时页面取不到数据，同样变成 HTTP 500。所以 5xx 时唯一的判据是
    ``login/status`` 的 ``is_logged_in``。
    """

    def test_not_logged_in_points_at_rescan(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500),
                    LOGIN_PATH: lambda request: login_status(False),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "账号掉登录" in message
        assert "扫码" in message

    def test_logged_in_points_at_page_or_risk_control(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500),
                    LOGIN_PATH: lambda request: login_status(True),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "风控" in message
        assert "页面加载失败" in message

    def test_the_two_branches_say_different_things(self, install_transport):
        """合并这两条文案会让用户朝反方向排查。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500),
                    LOGIN_PATH: lambda request: login_status(False),
                }
            )
        )
        with pytest.raises(CollectorError) as logged_out:
            backend().collect("防晒霜", limit=5)

        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500),
                    LOGIN_PATH: lambda request: login_status(True),
                }
            )
        )
        with pytest.raises(CollectorError) as logged_in:
            backend().collect("防晒霜", limit=5)

        assert str(logged_out.value) != str(logged_in.value)
        assert "扫码" in str(logged_out.value)
        assert "风控" in str(logged_in.value)

    def test_unreachable_login_check_does_not_claim_rescan(self, install_transport):
        """问不出登录状态时（``None``）不能说"账号掉登录" —— 那是没有依据的断言。"""

        def login_boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(
            route({SEARCH_PATH: lambda request: failure(500), LOGIN_PATH: login_boom})
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "无法确认登录状态" in message
        assert "账号掉登录" not in message

    def test_missing_is_logged_in_key_is_not_reported_as_logged_out(self, install_transport):
        """``is_logged_in`` 键消失（改名）算"问不出来"，**不能**断言"账号掉登录"。

        键**在不在**是"结构对不对"，值是什么才是"登没登" —— 合并这两件事会把
        结构变更伪装成掉登录，把用户支去重新扫码，而真正该做的是更新适配器。
        """
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500),
                    LOGIN_PATH: lambda request: success({"别的字段": True}),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "is_logged_in" in message
        assert "账号掉登录" not in message
        assert "扫码" not in message

    def test_4xx_does_not_ask_for_login_state(self, install_transport):
        """4xx 是调用方的问题，多打一次 login/status 只是白开一个浏览器页面。"""
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: failure(400, code="INVALID_REQUEST")})
        )
        with pytest.raises(CollectorError):
            backend().collect("防晒霜", limit=5)

        assert recorder.paths == [SEARCH_PATH]

    def test_detail_failure_also_gets_the_branch(self, install_transport):
        """分岔不止用于搜索：详情 500 时给出的指引必须同样可操作。"""
        install_transport(detail_router([card("N1")], fail_ids=["N1"]))
        with pytest.raises(CollectorError, match="2 篇取详情失败|1 篇取详情失败"):
            backend().collect("防晒霜", limit=5)


# --------------------------------------------------------------------------- #
# available()
# --------------------------------------------------------------------------- #


class TestAvailable:
    """``doctor`` 诊断用；报错必须指出**是哪一步**不对。"""

    def test_unreachable_service(self, install_transport):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(boom)
        collector = backend()

        assert collector.available() is False
        assert collector.last_error is not None
        assert "无法连接" in collector.last_error
        assert "服务已启动" in collector.last_error

    def test_running_but_not_logged_in(self, install_transport):
        install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: login_status(False),
                }
            )
        )
        collector = backend()

        assert collector.available() is False
        assert collector.last_error is not None
        assert "扫码" in collector.last_error

    def test_running_and_logged_in(self, install_transport):
        install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: login_status(True),
                }
            )
        )
        collector = backend()

        assert collector.available() is True
        assert collector.last_error is None

    def test_login_check_unreachable_is_surfaced_not_swallowed(self, install_transport):
        """登录状态问不出来时，原因要原样透出（``state=None``），不能压成一句"不可用"。"""

        def login_boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        install_transport(route({HEALTH_PATH: lambda request: healthy(), LOGIN_PATH: login_boom}))
        collector = backend()

        assert collector.available() is False
        assert collector.last_error is not None
        assert "无法确认登录状态" in collector.last_error

    def test_login_status_missing_field_points_at_structure_not_at_rescan(self, install_transport):
        """``is_logged_in`` 键消失时，``doctor`` 要给"适配器该更新"，不是"去扫码"。"""
        install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: success({"别的字段": True}),
                }
            )
        )
        collector = backend()

        assert collector.available() is False
        assert collector.last_error is not None
        assert "is_logged_in" in collector.last_error
        assert "扫码" not in collector.last_error

    def test_login_status_500_does_not_recurse(self, install_transport):
        """``/api/v1/login/status`` 自己回 5xx：报"问不出来"，**不许**回头再问一次登录状态。

        这条曾经是无限递归（``_http_error_message`` → ``login_state`` → ``request``
        → ``_http_error_message``），最终以 ``RecursionError`` 崩在 ``doctor``/
        ``available()`` 上。守卫是 ``_http_error_message`` 里按 ``path`` 的提前返回。
        """
        install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: failure(500, code="STATUS_CHECK_FAILED"),
                }
            )
        )
        collector = backend()

        assert collector.available() is False
        assert collector.last_error is not None
        assert "无法确认登录状态" in collector.last_error
        assert "登录状态本身查不出来" in collector.last_error

    def test_login_status_500_is_asked_once(self, install_transport):
        """非递归的可观测形式：登录检查只发**一次**请求（递归会发很多次直到崩）。"""
        recorder = install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: failure(500, code="STATUS_CHECK_FAILED"),
                }
            )
        )
        backend().available()

        assert recorder.paths == [HEALTH_PATH, LOGIN_PATH]

    def test_search_500_with_login_status_500_does_not_recurse(self, install_transport):
        """递归的另一半入口：搜索 500 → 问登录状态 → 那里也 500 → 曾经无限递归。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: failure(500, code="SEARCH_FEEDS_FAILED"),
                    LOGIN_PATH: lambda request: failure(500, code="STATUS_CHECK_FAILED"),
                }
            )
        )
        with pytest.raises(CollectorError) as exc_info:
            backend().collect("防晒霜", limit=5)

        message = str(exc_info.value)
        assert "无法确认登录状态" in message
        assert "登录状态本身查不出来" in message

    def test_only_two_endpoints_are_touched(self, install_transport):
        """``available()`` 不该顺手去搜索 / 取详情 —— 那会在服务端开浏览器页面。"""
        recorder = install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: login_status(True),
                }
            )
        )
        backend().available()

        assert recorder.paths == [HEALTH_PATH, LOGIN_PATH]

    def test_health_is_public_so_a_wrong_token_still_reports_service_up(self, install_transport):
        """``/health`` 在鉴权中间件之外：token 配错时仍要能回答"服务在不在"。"""
        recorder = install_transport(
            route(
                {
                    HEALTH_PATH: lambda request: healthy(),
                    LOGIN_PATH: lambda request: failure(401, code="UNAUTHORIZED", error="未授权"),
                }
            )
        )
        collector = backend(token="wrong-token")

        assert collector.available() is False
        assert collector.last_error is not None
        assert collector.base_url == "http://127.0.0.1:18060"
        assert LOGIN_PATH in recorder.paths

    def test_stops_after_health_failure(self, install_transport):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连接被拒绝")

        recorder = install_transport(boom)
        backend().available()

        assert recorder.paths == [HEALTH_PATH]


# --------------------------------------------------------------------------- #
# 请求体
# --------------------------------------------------------------------------- #


class TestRequestBodies:
    """详情请求的 body 必须逐字段对得上上游 ``FeedDetailRequest``。"""

    def test_fast_path_body(self, install_transport):
        """``max_comments_per_note`` ≤ 10：``load_all_comments=false`` 且不带 comment_config。"""
        recorder = install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", xsec_token="TOKEN-A")], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments()
                    ),
                }
            )
        )
        backend().collect("防晒霜", limit=5, max_comments_per_note=10)

        assert recorder.requests[1].method == "POST"
        assert recorder.body(1) == {
            "feed_id": "N1",
            "xsec_token": "TOKEN-A",
            "load_all_comments": False,
        }

    @pytest.mark.parametrize("limit", [0, 1, 10])
    def test_fast_path_boundary(self, install_transport, limit: int):
        recorder = install_transport(detail_router([card("N1")]))
        backend().collect("防晒霜", limit=5, max_comments_per_note=limit)

        body = recorder.body(1)
        assert body["load_all_comments"] is False
        assert "comment_config" not in body

    @pytest.mark.parametrize("limit", [11, 20, 200])
    def test_slow_path_body(self, install_transport, limit: int):
        recorder = install_transport(detail_router([card("N1")]))
        backend().collect("防晒霜", limit=5, max_comments_per_note=limit)

        body = recorder.body(1)
        assert body["load_all_comments"] is True
        assert body["comment_config"]["max_comment_items"] == limit
        assert body["comment_config"]["click_more_replies"] is False
        assert body["comment_config"]["scroll_speed"] == "normal"

    def test_feed_id_and_token_come_from_the_card(self, install_transport):
        """``feed_id`` 取 ``data.feeds[].id``，``xsec_token`` 取 ``data.feeds[].xsecToken``。"""
        recorder = install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("CARD-ID", xsec_token="CARD-TOKEN")], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("CARD-ID", **{"xsecToken": "DETAIL-TOKEN"}),
                        comments_obj=comments(),
                        feed_id="CARD-ID",
                    ),
                }
            )
        )
        backend().collect("防晒霜", limit=5)

        body = recorder.body(1)
        assert body["feed_id"] == "CARD-ID"
        assert body["xsec_token"] == "CARD-TOKEN"

    def test_snake_case_token_alias_is_accepted(self, install_transport):
        """卡片上若给的是 ``xsec_token``（另一版上游 / 另一个页面），也要能用。"""
        item = card("N1", xsec_token=None)
        item["xsec_token"] = "SNAKE-TOKEN"
        recorder = install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [item], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments()
                    ),
                }
            )
        )
        backend().collect("防晒霜", limit=5)

        assert recorder.body(1)["xsec_token"] == "SNAKE-TOKEN"

    def test_one_request_per_note_plus_one_search(self, install_transport):
        """每篇笔记一次详情请求：N 篇 = N+1 次浏览器会话，这是耗时的来源。"""
        recorder = install_transport(detail_router([card(f"N{i}") for i in range(3)]))
        backend().collect("防晒霜", limit=5)

        assert recorder.paths == [SEARCH_PATH] + [DETAIL_PATH] * 3


# --------------------------------------------------------------------------- #
# limit 语义
# --------------------------------------------------------------------------- #


class TestLimitSemantics:
    """``limit`` 只能截断，不能扩张（上游一次搜索只返回一页）。"""

    @pytest.mark.parametrize("limit", [0, -1, -100])
    def test_non_positive_limit_makes_no_request(self, install_transport, limit: int):
        recorder = install_transport(route({SEARCH_PATH: lambda request: success({"feeds": []})}))
        corpus = backend().collect("防晒霜", limit=limit)

        assert recorder.requests == []
        assert corpus.notes == []
        assert corpus.comments == []
        assert corpus.warnings == []
        assert corpus.backend == "xiaohongshu-mcp"
        assert corpus.keyword == "防晒霜"

    def test_truncates_to_limit(self, install_transport):
        recorder = install_transport(detail_router([card(f"N{i}") for i in range(5)]))
        corpus = backend().collect("防晒霜", limit=2)

        assert [n.note_id for n in corpus.notes] == ["N0", "N1"]
        assert recorder.paths == [SEARCH_PATH, DETAIL_PATH, DETAIL_PATH]
        # 要的比拿到的少：这不是"不完整"，不该有告警。
        assert corpus.warnings == []

    def test_more_requested_than_available_warns(self, install_transport):
        install_transport(detail_router([card("N0"), card("N1")]))
        corpus = backend().collect("防晒霜", limit=10)

        assert len(corpus.notes) == 2
        assert len(corpus.warnings) == 1
        assert "2 篇" in corpus.warnings[0]
        assert "10 篇" in corpus.warnings[0]
        assert "一页" in corpus.warnings[0]
        assert "防晒霜" in corpus.warnings[0]

    def test_keyword_is_stripped(self, install_transport):
        recorder = install_transport(
            route({SEARCH_PATH: lambda request: success({"feeds": [], "count": 0})})
        )
        corpus = backend().collect("  防晒霜 \n", limit=5)

        assert recorder.requests[0].url.params["keyword"] == "防晒霜"
        assert corpus.keyword == "防晒霜"

    def test_summary_and_collected_at_are_filled(self, install_transport):
        install_transport(detail_router([card("N0")]))
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.summary() == "1 篇笔记 / 0 条评论 / 1 张图片 (来源: xiaohongshu-mcp)"
        assert corpus.collected_at.tzinfo is not None


# --------------------------------------------------------------------------- #
# 合成 id
# --------------------------------------------------------------------------- #


class TestSyntheticIds:
    """``comment_id`` 是去重键：必须**稳定**（跨运行同值）且**唯一**（不撞车）。"""

    @staticmethod
    def router(payload: Any = None) -> Handler:
        if payload is None:
            payload = comments(
                comment(None, content="一级缺 id", subs=[comment(None, content="二级缺 id")]),
                comment("C2", content="有一级 id", subs=[comment(None, content="另一个二级缺 id")]),
            )
        return route(
            {
                SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                DETAIL_PATH: lambda request: detail(note_obj=note("N1"), comments_obj=payload),
            }
        )

    def test_same_input_gives_same_ids(self, install_transport):
        """两次采集同一份数据必须得到同一组 id，否则提及次数会随机抖动。"""
        install_transport(self.router())
        first = backend().collect("防晒霜", limit=5, max_comments_per_note=20)
        second = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in first.comments] == [c.comment_id for c in second.comments]

    def test_synthetic_id_shape(self, install_transport):
        """父序号进合成键（``c{index}`` / ``c{index}s{sub}``）—— 两个父下的首个子回复不再同 id。"""
        install_transport(self.router())
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == [
            "N1#c0",
            "N1#c0s0",
            "C2",
            "N1#c1s0",
        ]

    def test_real_id_wins_over_synthetic(self, install_transport):
        install_transport(self.router())
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert "C2" in [c.comment_id for c in corpus.comments]

    def test_comment_id_alias_is_accepted(self, install_transport):
        """上游驼峰是 ``id``；另一版若给 ``commentId`` 也要认。"""
        item = comment(None, content="只有 commentId")
        item["commentId"] = "ALIAS-1"
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1"), comments_obj=comments(item)
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["ALIAS-1"]

    def test_synthetic_ids_are_unique(self, install_transport):
        install_transport(self.router())
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)
        ids = [c.comment_id for c in corpus.comments]

        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("parents", [1, 2, 5])
    def test_uniqueness_holds_for_many_parents(self, install_transport, parents: int):
        """尽力证伪：N 个父、每个父都有若干**缺 id 的子评论**（每父的序号都从 0 起）。

        曾经这里会撞车（``s0`` 只带子序号）。这条用例是"唯一性"的真正判据 ——
        只测两个父容易被巧合满足。
        """
        payload = comments(
            *[
                comment(
                    None,
                    content=f"父{i}",
                    subs=[comment(None, content=f"子{i}"), comment(None, content=f"子{i}b")],
                )
                for i in range(parents)
            ]
        )
        install_transport(self.router(payload))
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=100)

        ids = [c.comment_id for c in corpus.comments]
        assert len(ids) == len(set(ids)) == parents * 3

    def test_real_id_shaped_like_ours_still_wins(self, install_transport):
        """上游给的真实 id 与"我们会合成的形状"相同：**真实 id 优先**，且不额外撞车。

        合成键的唯一性是"合成值之间"的性质（``_comment_id`` 的 docstring 也只这么
        主张）。构造性的反例是存在的 —— 让**真实** id 恰好等于**另一个父的子评论**
        的合成键（如父 1 的真实 id 是 ``N1#c0s0``，而父 0 的缺 id 子评论合成出同一个
        字符串）就会撞。但那需要平台下发一个**我们私有格式**的 id（含本适配器拼的
        ``#`` 与父/子序号），不可达，也没有修法（换格式只是换一个私有前缀）。记在
        这里是为了把边界写清楚，不是留个待办。
        """
        payload = comments(
            comment(None, content="父缺 id", subs=[comment("N1#c0s0", content="真实 id 撞了")]),
        )
        install_transport(self.router(payload))
        corpus = backend().collect("防晒霜", limit=5, max_comments_per_note=20)

        assert [c.comment_id for c in corpus.comments] == ["N1#c0", "N1#c0s0"]
        assert len({c.comment_id for c in corpus.comments}) == 2


# --------------------------------------------------------------------------- #
# 个人信息
# --------------------------------------------------------------------------- #


PII_VALUES = (
    "PII-USER-RAW",
    "PII-昵称-明文",
    "PII-驼峰昵称-明文",
    "https://cdn.example/PII-AVATAR.jpg",
    "PII-IP-上海",
)


def pii_person() -> dict[str, Any]:
    """带标识性字符串的 ``User`` —— 只要有一个进了语料，断言就能抓到。"""
    return {
        "userId": "PII-USER-RAW",
        "nickname": "PII-昵称-明文",
        "nickName": "PII-驼峰昵称-明文",
        "avatar": "https://cdn.example/PII-AVATAR.jpg",
        "ipLocation": "PII-IP-上海",
    }


class TestPiiGuard:
    """上游响应里带昵称 / 头像 / 原始 userId / ipLocation —— 一个都不许进语料。

    模型是 ``slots=True`` 的，没有 ``__dict__``，所以用 :func:`dataclasses.asdict`
    序列化整个语料再找这些值 —— 这比逐字段列举更难被"新增字段"绕过。
    """

    @staticmethod
    def collect_with_pii(install_transport) -> RawCorpus:
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success(
                        {"feeds": [card("N1", user=pii_person())], "count": 1}
                    ),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", user=pii_person()),
                        comments_obj=comments(
                            comment("C1", user=pii_person()),
                            comment(
                                "C2", user=pii_person(), subs=[comment("C2-1", user=pii_person())]
                            ),
                        ),
                    ),
                }
            )
        )
        return backend().collect("防晒霜", limit=5, max_comments_per_note=20)

    def test_serialized_corpus_has_no_pii(self, install_transport):
        corpus = self.collect_with_pii(install_transport)
        blob = json.dumps(dataclasses.asdict(corpus), ensure_ascii=False, default=str)

        for value in PII_VALUES:
            assert value not in blob, f"语料里出现了个人信息：{value}"

    def test_raw_user_id_is_hashed_not_kept(self, install_transport):
        corpus = self.collect_with_pii(install_transport)

        assert corpus.notes[0].author_hash == hash_id("PII-USER-RAW")
        assert all(c.user_hash == hash_id("PII-USER-RAW") for c in corpus.comments)
        assert corpus.notes[0].author_hash != "PII-USER-RAW"

    def test_hash_shape(self, install_transport):
        """16 位十六进制（``hash_id`` 的约定），不是"砍掉几个字符"。"""
        corpus = self.collect_with_pii(install_transport)
        digest = corpus.notes[0].author_hash

        assert len(digest) == 16
        assert all(char in "0123456789abcdef" for char in digest)

    def test_snake_case_user_id_alias_is_hashed(self, install_transport):
        """上游的站点态里 ``userId`` 与 ``user_id`` 两种写法都存在（login.go 里同款兜底）。"""
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", user={"user_id": "SNAKE-USER"}),
                        comments_obj=comments(),
                    ),
                }
            )
        )
        corpus = backend().collect("防晒霜", limit=5)

        assert corpus.notes[0].author_hash == hash_id("SNAKE-USER")

    def test_missing_user_is_empty_hash(self, install_transport):
        install_transport(
            route(
                {
                    SEARCH_PATH: lambda request: success({"feeds": [card("N1")], "count": 1}),
                    DETAIL_PATH: lambda request: detail(
                        note_obj=note("N1", user="不是对象"), comments_obj=comments()
                    ),
                }
            )
        )
        assert backend().collect("防晒霜", limit=5).notes[0].author_hash == ""


# --------------------------------------------------------------------------- #
# warnings 的去向
# --------------------------------------------------------------------------- #


class TestWarningsPropagation:
    """``RawCorpus.warnings`` 必须走到用户看得到的地方。"""

    def test_cli_collect_prints_warnings(self, install_transport):
        """``collect`` 命令是用户验证采集层的地方，语料不完整必须当场说。"""
        install_transport(detail_router([card("N0"), card("N1")]))
        result = CliRunner().invoke(
            cli_main, ["collect", "-k", "防晒霜", "-n", "10", "--backend", "mcp"]
        )

        assert result.exit_code == 0, result.output
        assert "2 篇" in result.output
        assert "换关键词" in result.output

    def test_mine_puts_warnings_into_result_notes(self):
        """单元级：采集告警要进 ``MiningResult.notes``，不能只躺在语料对象上。

        只构造一个"采集到但没东西可分析"的语料走 ``_empty_result`` 分支，
        不跑整条 mine（那需要 LLM 与 embedding）。
        """
        from xhs_pain_miner.pain_miner import PainMiner

        corpus = RawCorpus(
            keyword="防晒霜",
            backend="xiaohongshu-mcp",
            warnings=["有 3/10 篇笔记取详情失败、未计入语料。"],
        )
        miner = PainMiner(settings=Settings(_env_file=None))
        try:
            result = miner.mine("防晒霜", corpus=corpus)
        finally:
            miner.close()

        assert any("3/10" in message for message in result.notes)

    def test_warnings_default_to_empty_list(self):
        """默认必须是空列表，不能是 ``None`` —— 调用方会直接 for 它。"""
        corpus = RawCorpus(keyword="防晒霜")

        assert corpus.warnings == []
        assert list(corpus.warnings) == []


# --------------------------------------------------------------------------- #
# 配置层
# --------------------------------------------------------------------------- #


class TestSettings:
    """三个新配置项与一个被删掉的死配置。"""

    def test_defaults(self):
        settings = Settings(_env_file=None)

        assert settings.xhs_mcp_url == "http://127.0.0.1:18060"
        assert settings.xhs_mcp_timeout == 120.0
        assert settings.xhs_mcp_token is None

    def test_url_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XHS_MCP_URL", "http://127.0.0.1:19999")
        assert Settings(_env_file=None).xhs_mcp_url == "http://127.0.0.1:19999"

    @pytest.mark.parametrize("name", ["XHS_MCP_TOKEN", "XHS_MCP_AUTH_TOKEN"])
    def test_token_accepts_both_names(self, monkeypatch: pytest.MonkeyPatch, name: str):
        monkeypatch.delenv("XHS_MCP_TOKEN", raising=False)
        monkeypatch.delenv("XHS_MCP_AUTH_TOKEN", raising=False)
        monkeypatch.setenv(name, "sk-local")

        assert Settings(_env_file=None).xhs_mcp_token == "sk-local"

    def test_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XHS_MCP_TIMEOUT", "42.5")
        assert Settings(_env_file=None).xhs_mcp_timeout == 42.5

    def test_dead_cookie_setting_is_gone(self):
        """登录态由被对接的服务保管，本程序不接触也不该有 cookie 配置项。"""
        assert not hasattr(Settings(_env_file=None), "xhs_cookie")

    def test_default_timeout_matches_the_backend_constant(self):
        """两处不一致会让"直接构造后端"与"走配置构造"得到不同行为。"""
        assert Settings(_env_file=None).xhs_mcp_timeout == mcp.DEFAULT_TIMEOUT

    @pytest.mark.parametrize("name", ["xhs_mcp_url", "xhs_mcp_token", "xhs_mcp_timeout"])
    def test_default_documented_in_env_example(self, name: str):
        """``.env.example`` 是用户配置的唯一入口，键名必须对得上。"""
        from pathlib import Path

        example = Path(__file__).resolve().parent.parent / ".env.example"
        text = example.read_text(encoding="utf-8")

        # 注释掉的示例行形如 ``# XHS_MCP_URL=http://...``
        assert name.upper() in text
        if name != "xhs_mcp_token":
            assert f"# {name.upper()}=" in text

    def test_factory_builds_mcp_backend_from_settings(self):
        from xhs_pain_miner.collectors.factory import build_collector

        settings = Settings(
            _env_file=None,
            collector_backend="mcp",
            xhs_mcp_url="http://127.0.0.1:19998",
            xhs_mcp_timeout=9.0,
        )
        collector = build_collector(settings)

        assert isinstance(collector, MCPBackend)
        assert collector.base_url == "http://127.0.0.1:19998"
        assert collector.name == "xiaohongshu-mcp"
