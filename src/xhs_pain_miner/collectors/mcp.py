"""MCP 采集后端 —— 对接 ``xiaohongshu-mcp`` 服务。

上游许可，以及「为什么这个适配器可以在仓库里」
------------------------------------------
本仓库的规则是**不携带任何第三方采集实现**（见 :mod:`~xhs_pain_miner.collectors.base`
与 ``docs/collector-plugin.md``）。本模块是这条规则的一个**有边界的例外**，
边界就是许可证：

* ``xiaohongshu-mcp`` 是 **Apache-2.0** —— 可商用、可再分发、无附加限制；
* MediaCrawler 一类是「非商业学习许可」，只能由用户自备（``plugin`` 后端）。

即便如此，本模块里也**没有一行采集代码**：采集由用户在本机运行的
``xiaohongshu-mcp`` 服务完成，本仓库只调用它的 HTTP 接口并做字段映射。

走 REST 而不是 MCP 协议
----------------------
``xiaohongshu-mcp`` 同时提供两套接口，同一个进程、同一个端口、同一份数据：
REST 的 ``/api/v1/*``（该仓库 ``docs/API.md`` 有完整文档）与 MCP 协议的 ``/mcp``
（``routes.go`` 把两者挂在同一个 ``AppServer`` 上）。本适配器走 **REST**：

1. **契约强度**：MCP 那条路把同一份 JSON 塞进 ``content[].text`` 字符串里，
   调用方还要再 ``json.loads`` 一次 —— 同一份数据、更弱的形状约定；
2. **依赖与并发模型**：官方 MCP SDK 是 async 优先，而本仓库全程同步；为一个
   HTTP 接口在同步函数里跑事件循环，换不来任何东西；
3. **可测**：REST 能直接用 ``httpx.MockTransport`` 注入，与
   :mod:`xhs_pain_miner.research.appstore` 是同一套约定（见 ``_transport``）。

后端名仍叫 ``mcp``，因为它对接的是 **xiaohongshu-mcp 这个服务**，
而不是"某种传输协议"。

用之前请先读：这个后端的三个硬约束
--------------------------------
1. **一次搜索只有一页。** 上游的 ``SearchAction.Search`` 是"导航到搜索页 →
   读页面 ``__INITIAL_STATE__`` → 返回"，**不滚动、没有 cursor**。所以
   ``limit`` 只能**截断**，不能**扩张** —— 想覆盖更大的语料只能换关键词分多次
   采集。``collect`` 会在"要的比拿到的多"时把这件事写进
   :attr:`~xhs_pain_miner.models.RawCorpus.warnings`。
2. **每篇笔记要再单独请求一次**才有正文：搜索结果的卡片里只有标题与封面，
   没有 ``desc``。而服务端每处理一次请求就开一个浏览器页面，所以 N 篇笔记
   ≈ N+1 次浏览器会话，这是耗时的主要来源，也是风控暴露的主要来源。
3. **平台把计数当显示文案下发**（``"80"``、``"1.2万"``），不是数字，
   见 :func:`_parse_count`。

个人信息
--------
上游响应里带 ``userInfo.nickname`` / ``avatar`` / ``userId``（详情里连
``ipLocation`` 都有）。本模块**只取 ``userId`` 并立即哈希**，昵称与头像在
:func:`_to_note` / :func:`_to_comments` 里就丢掉了，不进入 ``RawNote.extra``、
也不进入任何其它字段。``ipLocation`` 同样不取。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx

from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.models import RawComment, RawCorpus, RawNote, hash_id

DEFAULT_BASE_URL = "http://127.0.0.1:18060"
"""``xiaohongshu-mcp`` 的默认监听地址（该服务只监听本机，不要指向公网地址）。"""

DEFAULT_TIMEOUT = 120.0
"""单次请求超时（秒）。

比 :mod:`~xhs_pain_miner.research.appstore` 的 15 秒大一个量级，因为服务端每次
请求都要开一个浏览器页面去小红书页面上取数据；开了 ``load_all_comments`` 之后
还要滚动加载，几十秒是常态。超时定小了会把"慢"误报成"失败"。

**但它仍远小于上游给自己留的余地**：上游详情接口的超时是 10 分钟，导航与点击
前后还各带一段拟人延时（导航后 0.6–6 秒、点击前 80 毫秒–1 秒、点击后 150 毫秒–2 秒）。
所以 120 秒是一个**"宁可失败也不无限等"的选择**，不是服务能力的上限 —— 网络慢
或评论特别多时，一次合法但很慢的请求会在这里被掐断，表现为"这篇取详情失败"。
单篇失败会被跳过并计数（见 :data:`MAX_FAILURE_RATIO` 与
:attr:`~xhs_pain_miner.models.RawCorpus.warnings`），所以后果是可见的；
确实要等它跑完就调大 ``XHS_MCP_TIMEOUT``。
"""

FAST_PATH_COMMENT_LIMIT = 10
"""``--comments`` 不超过这个数时走"快路径"（``load_all_comments=false``）。

**10 这个数只来自上游 README 的一句说明**（"默认 false 仅返回前 10 条一级评论"），
在上游源码里**找不到任何对应的常量或截断逻辑** —— 快路径既不滚动也不切片，返回的
就是页面初始状态里恰好有的那批。也就是说：

* 真实条数**可能少于 10**，而且是常态（页面首屏就几条时就是几条）；
* 快路径因此**可能比慢路径少拿到评论**，且这是**静默的** —— ``_warnings`` 目前
  只覆盖"详情失败"与"取到的篇数少于 limit"，不覆盖"评论少于请求数"。

所以它只用来**决定走哪条路径**，不用来截断（真正的截断按调用方给的
``max_comments_per_note`` 做）。默认值 20 落在慢路径上，这是有意的：默认行为应当
满足"要够用的证据"，而不是"要最快"。只有调用方**显式**把 ``--comments`` 调到 10
以内，才会换来"快、但可能少"。
"""

MAX_FAILURE_RATIO = 0.5
"""详情请求的失败率上限；**超过**就中止整次采集（恰好一半不中止）。

少数几篇取详情失败（单篇超时）不该让整轮白跑，所以默认跳过并计数。但**过半失败
通常不是个别笔记的问题** —— 这时返回一份缩水一半的语料，下游会把"提及次数偏少"
读成"这个品类没什么人提"，那是比直接报错更糟的结论。

**"通常是服务异常或风控"是大概率归因，不是全部可能。** 至少还有三种可达成因，
而且它们各自需要完全不同的处置，所以异常信息里会带上"第一条失败的原因"供分辨：

* ``XHS_MCP_TIMEOUT`` 调小了 —— 一次合法但很慢的请求被自己这边掐断
  （见 :data:`DEFAULT_TIMEOUT`），与服务端无关；
* 响应结构变更 —— 例如搜索卡片里的 ``xsecToken`` 被改名，**每一篇**都会在
  :meth:`MCPBackend._fetch_detail` 抛错，失败率 100%，而服务完全正常；
* 笔记被删或转为私密 —— 上游的页面可访问性检查会明确拒绝这类笔记，属于内容
  不可得，不是故障。
"""

_transport: httpx.BaseTransport | None = None
"""测试注入点：非 ``None`` 时所有请求都走这个传输层（``httpx.MockTransport``）。

与 :mod:`xhs_pain_miner.research.github` / ``appstore`` 上的同名变量是同一套约定：
生产路径恒为 ``None``；测试注入而不是 monkeypatch ``httpx`` 内部实现，后者会把
测试绑死在 httpx 的私有结构上，一次依赖升级就能让测试静默失效。
"""

_COUNT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([万亿wWkK]?)$")
"""平台侧计数的形态：``80`` / ``1.2万`` / ``3.4w`` / ``1.5k`` / ``10万+``。"""

_COUNT_MULTIPLIERS = {
    "": 1,
    "w": 10_000,
    "W": 10_000,
    "万": 10_000,
    "k": 1_000,
    "K": 1_000,
    "亿": 100_000_000,
}

_NOTE_REQUIRED_KEYS = ("title", "desc", "interactInfo", "imageList")
"""详情响应里 ``note`` 必须具备的键。

查的是**键在不在**，不是值真不真 —— 图文笔记可以没有正文（``desc`` 为 ``""``），
那是正常数据；而**键消失**意味着上游或小红书页面结构变了，那是适配器需要更新的
信号。把这两件事混在一起（例如"desc 为空就报错"）会让一条空正文把整次采集打断。

``imageList`` 也在名单里：它改名时后果是**图片静默归零**，``--deep`` 于是什么都不做
却看起来一切正常 —— 那比报错难查得多。

**这份名单的判据**：该字段在上游 Go 结构体里的 json tag **没有 ``omitempty``**，
因此只要响应是这个服务给的，键就恒存在（值可能是 ``null``）。有 ``omitempty``
就不能这么判 —— 键缺失可能只是"这个字段没有值"。下面评论那几个键同理，
它们的 tag 同样不带 ``omitempty``。
"""

_COMMENT_REQUIRED_KEYS = ("content",)
"""评论对象必须具备的键（**判据同** :data:`_NOTE_REQUIRED_KEYS`：json tag 无 ``omitempty``）。

**这个键尤其不能漏检**：它改名时，每一条评论的正文都会变成空串，而空正文评论会被
:func:`_comment_from` 丢掉 —— 结果是**整批评论静默消失**，语料看上去像"这篇笔记
压根没有评论"，而下游所有的「提及次数」都建立在这批文本上。

``likeCount`` / ``createTime`` **不列进来**，用的是**另一套判据**：它们有定义好的
降级路径（解析不出来取 ``0`` / ``None``），所以键缺失按降级处理，不当结构变更报错。

⚠️ 那套判据回答的是"**该不该报错**"，**不是"影不影响结论"—— 它们会影响。**
实测（同一簇）：

* ``likeCount`` 缺失（取 ``0``）→ 「痛点强度」因子 **0.90 → 0.45**；
* ``createTime`` 缺失（取 ``None``）→ 证据里只剩 1/6 带时间戳，「增长趋势」
  **从 0.1667（明显在落）退回中性 0.5**，卡片措辞与分数都会变。

也就是说这几个键缺失时，适配器选的是"安静地按保守值算"而不是"报错"。这是有代价的
取舍，理由与量级写在 :func:`_parse_count` 的 docstring 里（``likes`` 是同一回事）。
"""

_SECONDS_UPPER_BOUND = 1e11
"""判断时间戳是秒还是毫秒的分界。

当前 epoch 秒约 1.7e9、毫秒约 1.7e12，相差三个数量级，1e11 把两者干净地分开：
比它小的是秒（对应 1973 年以前不可能出现的毫秒值），比它大的是毫秒。
"""

LOGIN_STATUS_PATH = "/api/v1/login/status"
"""登录状态查询路径。

单独提出来是因为它在 ``_http_error_message`` 里要参与判断：**这个路径失败时不能
再去查登录状态**，否则会自我递归（见那里的说明）。
"""


def _parse_count(value: object) -> int:
    """把平台下发的计数解析成整数。

    小红书把这些数字当**展示文案**下发：``"80"`` 是字符串，热门内容会是
    ``"1.2万"``，有时带千分位或 ``+`` 后缀。直接 ``int()`` 会在第一条热门笔记上
    抛 ``ValueError``。

    **解析不出来的一律返回 ``0``**，不抛异常。空值、``None``、``"赞"`` 这类无意义的
    字符串因此都归为 ``0`` —— 它是"未知"的保守取值，不是"零赞"的断言。

    ⚠️ **这个值不是无害的展示字段，别把它当可以随便归零的东西。** 实测它经
    ``Evidence.likes`` → ``_evidence_weight`` →
    :func:`~xhs_pain_miner.scoring.opportunity.pain_strength` 直接进「痛点强度」
    **因子**，而那是机会分的一项（权重 0.25）。同一簇只改 likes（实测取
    ``sentiment = -0.8`` 的簇）：``0`` 时 ``pain_strength = 0.45``，``12000`` 时
    ``0.90``。

    要记住的是**比值而不是那两个绝对数** —— 后者随簇的情感值变（中性情感下是
    ``0.25`` / ``0.50``），但 ``likes`` 从 0 到上万**恒定把这一项翻倍**。也就是说
    **这个解析一失败，卡片分数可能腰斩**。返回 ``0`` 是"宁可低估也不编造"，
    但它是**有代价**的降级，不是无所谓的兜底。

    （同一批字段里 ``collects`` 与 ``comments_count`` 才是纯展示：前者全仓库没有
    消费者，后者只出现在 ``collect`` 命令的表格里。）

    Args:
        value: 平台返回的原始值（字符串 / 数字 / ``None``）。

    Returns:
        解析出的非负整数；无法解析时为 ``0``。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，但它在这里一定是数据错误
        return 0
    if isinstance(value, (int, float)):
        # ±inf 与 nan 都会让 int() 抛异常（OverflowError / ValueError）。
        # 经真实上游不可达（Go 的 json.Marshal 拒绝这三个值），但"不抛异常"是这个
        # 函数的契约，不该依赖调用方那边的序列化器来兜。
        try:
            return max(0, int(value))
        except (ValueError, OverflowError):
            return 0
    if not isinstance(value, str):
        return 0

    text = value.strip().replace(",", "").replace("+", "")
    if not text:
        return 0

    match = _COUNT_RE.match(text)
    if match is None:
        return 0

    number, unit = match.groups()
    try:
        return max(0, int(float(number) * _COUNT_MULTIPLIERS[unit]))
    except (ValueError, OverflowError):  # pragma: no cover - 正则已排除绝大多数情况
        return 0


def _parse_timestamp(value: object) -> datetime | None:
    """把平台下发的时间戳解析成**带时区的 UTC datetime**。

    ``0``、``None``、非数字一律返回 ``None`` —— 上游对"没有这个时间"的表示就是
    ``0``，把它当成 1970-01-01 会让趋势因子把几十条证据全算成"很久以前"。

    **单位是毫秒这件事，依据是上游的 ``docs/API.md``（"``note.time``: 笔记发布时间戳
    （毫秒）"），不是它能被源码证明的事实** —— Go 侧把这两个字段声明成裸 ``int64``
    原样透传，从不解析也不换算，单位完全取决于小红书页面给什么。真按秒下发时
    下面的兼容分支能兜住（结果仍然正确），但它兜不住"既不是秒也不是毫秒"的单位。

    时区选择沿用仓库内既有的两条约定：内置样例语料用的是带偏移量的 ISO 字符串
    （即 **aware**），``RawCorpus.collected_at`` 用的是 ``timezone.utc``。这里同样
    产出 aware UTC。**不要**改成 ``datetime.fromtimestamp(v)``（naive 本地时间）：
    那会让同一份数据在不同时区的机器上解析出不同的值，而本项目对"可复现"的要求
    不低；而且 aware 与 naive 混在同一个列表里做 ``min`` / ``max`` 会直接抛
    ``TypeError``（趋势因子的算法就是这么比较的）。

    Args:
        value: 毫秒时间戳；也容忍秒级时间戳（见 :data:`_SECONDS_UPPER_BOUND`）。

    Returns:
        带 UTC 时区的 datetime；无法解析时为 ``None``。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None

    seconds = value / 1000 if value > _SECONDS_UPPER_BOUND else value
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], path: str) -> None:
    """校验响应对象里存在预期的键。

    Raises:
        CollectorError: 有键缺失 —— 这通常意味着上游或小红书页面结构已经变更。
    """
    missing = [key for key in keys if key not in obj]
    if missing:
        raise CollectorError(
            f"xiaohongshu-mcp 的响应结构与预期不符：{path} 里缺少 {missing}。"
            "适配器是按该服务的 docs/API.md 与源码中的结构写的，出现这个错误通常"
            "意味着上游或小红书页面结构已变更，需要更新适配器（而不是重试）。"
        )


def _as_mapping(value: object, path: str) -> Mapping[str, Any]:
    """确认 ``value`` 是一个 JSON 对象。

    Raises:
        CollectorError: 不是对象。
    """
    if not isinstance(value, Mapping):
        raise CollectorError(
            f"xiaohongshu-mcp 的响应结构与预期不符：{path} 应当是 JSON 对象，"
            f"实际是 {type(value).__name__}。"
        )
    return value


def _as_list(value: object, path: str) -> list[Any]:
    """确认 ``value`` 是一个 JSON 数组（``null`` 视作空数组）。

    ``null`` 之所以不算错：笔记关闭评论区、或搜索结果为空时，上游会把数组序列化成
    ``null``。它表示"没有内容"，不是"结构变了"。
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise CollectorError(
            f"xiaohongshu-mcp 的响应结构与预期不符：{path} 应当是 JSON 数组，"
            f"实际是 {type(value).__name__}。"
        )
    return value


def _author_hash(user: object) -> str:
    """从上游的用户对象里取出 ``userId`` 并**立即**哈希。

    只读 ``userId``：``nickname`` / ``nickName`` / ``avatar`` 属于个人信息，
    对本项目的分析毫无必要，因此在映射层就丢掉，不进入任何下游字段。
    兼容驼峰与下划线两种写法（详情页与搜索结果来自不同页面，字段名不完全一致）。

    Args:
        user: 上游的用户对象（``user`` 或 ``userInfo``）。

    Returns:
        16 位十六进制哈希；取不到 ``userId`` 时为空串。
    """
    if not isinstance(user, Mapping):
        return ""
    raw = user.get("userId") or user.get("user_id") or ""
    return hash_id(str(raw)) if raw else ""


def _note_url(note_id: str, xsec_token: str) -> str:
    """拼出笔记的网页地址（写进机会卡片的证据链）。

    ``xsec_token`` 是访问该笔记所必需的：不带它打开详情页会跳登录或直接 404。
    这不是"额外的装饰参数"，是链接能不能点开的前提。
    """
    if not note_id:
        return ""
    if not xsec_token:
        return f"https://www.xiaohongshu.com/explore/{note_id}"
    return (
        f"https://www.xiaohongshu.com/explore/{note_id}?xsec_token={xsec_token}&xsec_source=pc_feed"
    )


class _Client:
    """``xiaohongshu-mcp`` REST 接口的薄封装 —— 只管传输与错误翻译。

    与业务无关：它不认识笔记、评论或痛点，只保证"要么返回一个 ``success: true``
    的 JSON 对象，要么抛一个说清楚发生了什么的 :class:`CollectorError`"。
    """

    def __init__(self, base_url: str, token: str | None, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """服务地址（错误信息与 ``doctor`` 展示用）。"""
        return self._base_url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """发起一次请求并解开通用响应信封。

        Args:
            method: HTTP 方法。
            path: 以 ``/`` 开头的路径，如 ``"/api/v1/feeds/search"``。
            json_body: 请求体。
            params: 查询参数。

        Returns:
            响应体 ``data`` 字段之外的完整对象 —— 即 ``{"success", "data", "message"}`。
            调用方通常直接取 ``result["data"]``。

        Raises:
            CollectorError: 连接失败、超时、鉴权失败、HTTP 非 200、响应不是 JSON，
                或 200 响应体里没有 ``success: true``。

        Note:
            **判据是 HTTP 状态码，不是响应体。** 上游的成功与错误用的是两个不同的
            结构体：成功走 ``SuccessResponse{success, data, message}`` 且状态码固定
            200；错误走 ``ErrorResponse{error, code, details}`` —— **错误体里根本
            没有 ``success`` 字段**，状态码则是显式指定的 4xx/5xx。所以
            ``success: false`` 这种形态在上游不存在，不能作为错误判据。
        """
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(transport=_transport, timeout=self._timeout) as client:
                response = client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=self._headers(),
                )
        except httpx.ConnectError as exc:
            raise CollectorError(
                f"无法连接 xiaohongshu-mcp 服务（{self._base_url}）：{exc}\n"
                "请确认服务已在运行。首次使用需要先用该服务自带的登录工具扫码登录，"
                "再启动服务本身；详见 docs/collector-mcp.md。"
            ) from exc
        except httpx.TimeoutException as exc:
            raise CollectorError(
                f"xiaohongshu-mcp 请求超时（{path}，超过 {self._timeout:.0f}s）。"
                "该服务用浏览器自动化取数据，加载全部评论时耗时会明显变长 —— "
                "可以调大 XHS_MCP_TIMEOUT 后重试。"
            ) from exc
        except httpx.HTTPError as exc:
            raise CollectorError(
                f"xiaohongshu-mcp 请求失败（{path}）：{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code == 401:
            raise CollectorError(
                "xiaohongshu-mcp 拒绝了本次请求（HTTP 401）：服务端启用了鉴权，"
                "而本程序提供的 token 缺失或不正确。请把与服务端 AUTH_TOKEN 相同的值"
                "写进 XHS_MCP_TOKEN。"
            )
        if response.status_code != 200:
            raise CollectorError(self._http_error_message(path, response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise CollectorError(
                f"xiaohongshu-mcp 返回的不是合法 JSON（{path}）。"
                "最常见的原因是地址指到了别的服务上（例如端口被占用），"
                "请确认 XHS_MCP_URL 指向的是 xiaohongshu-mcp。"
            ) from exc

        if not isinstance(payload, Mapping):
            raise CollectorError(
                f"xiaohongshu-mcp 的响应不是 JSON 对象（{path}），实际是 {type(payload).__name__}。"
            )

        if payload.get("success") is not True:
            # 上游 200 恒为业务成功，所以走到这里说明响应不是它给的（端口被占用、
            # 前面挂了别的服务……）。这句话要说得像"地址不对"，而不是"采集失败"。
            raise CollectorError(
                f"xiaohongshu-mcp 返回了 HTTP 200 但响应体不是它的格式（{path}）："
                f"{str(payload)[:200]}\n请确认 XHS_MCP_URL 指向的是 xiaohongshu-mcp。"
            )

        return payload

    def _http_error_message(self, path: str, response: httpx.Response) -> str:
        """为一次非 200 响应组织错误信息；5xx 时额外分辨"是不是掉登录了"。

        这一步存在的理由：上游的 ``SearchFeeds`` 与 ``GetFeedDetailWithConfig``
        **都没有登录前置检查** —— 账号掉登录时页面取不到数据，直接变成
        **HTTP 500**，而不是任何形式的 401。于是"服务坏了"和"该重新扫码了"这两种
        需要完全相反处理的情况，在原始报错里长得一模一样。
        """
        detail = response.text[:200].strip()
        message = f"xiaohongshu-mcp 返回 HTTP {response.status_code}（{path}）：{detail}"

        if response.status_code < 500:
            return message

        if path == LOGIN_STATUS_PATH:
            # 失败的**就是**登录状态查询本身。这里绝不能再问一次登录状态：
            # login_state() → request() → 本方法 → login_state() …… 会一路自我
            # 调用到 RecursionError，而 RecursionError 不是 CollectorError，
            # login_state 里的 except 拦不住，最终以崩栈的形式冒到用户面前。
            #
            # 可达性不低：上游的登录状态检查出错时回的正是 500 STATUS_CHECK_FAILED
            # —— "服务活着、只是这一次登录检查失败了"就会走到这里。
            return f"{message}\n登录状态本身查不出来，因此无法判断是不是掉登录了。"

        state, reason = self.login_state()
        if state is False:
            # 5xx 里最常见的一种不是"服务坏了"。说清楚，省一次来回。
            return f"{message}\n这是账号掉登录的典型表现：{reason}"
        if state is True:
            return (
                f"{message}\n该服务报告账号仍处于登录状态，因此更像是页面加载失败或被"
                "平台风控。可以稍后重试；持续出现请查看服务端日志。"
            )
        return f"{message}\n{reason}"

    def healthy(self) -> bool:
        """``GET /health`` 是否可达。

        ``/health`` 在服务端是**公开端点**（``routes.go`` 把它注册在鉴权中间件
        之外），所以哪怕 token 配错了，这里也能给出"服务在不在"这个独立的信息 ——
        这两件事混在一个错误里，用户就得靠猜。
        """
        try:
            self.request("GET", "/health")
        except CollectorError:
            return False
        return True

    def login_state(self) -> tuple[bool | None, str]:
        """账号当前的登录状态。

        Returns:
            ``(状态, 说明)``，状态有三种取值：

            * ``True`` —— 已登录（``available()`` 与 5xx 报错文案都会用到）；
            * ``False`` —— **确认**未登录（服务明确回了 ``is_logged_in: false``），
              说明串给出可操作的下一步；
            * ``None`` —— **问不出来**（服务不可达 / 响应不是合法结构 / 响应里
              没有 ``is_logged_in`` 这个键），说明串给出原因。

            区分 ``False`` 与 ``None`` 是必要的：这两者决定了"去重新扫码"还是
            "去查服务"，合并成 ``False`` 会给出方向相反的指引。特别是**键缺失**
            必须算 ``None`` 而不是 ``False`` —— 那是响应结构变了的信号，把它读成
            "账号未登录"会把用户支去重新扫码，而真正该做的是更新适配器。

        Note:
            服务端实现这个检查时会开一个浏览器页面（``CheckLoginStatus`` 里
            ``newBrowser()`` + ``NewPage()``），因此它有几秒开销，但**不产生任何
            数据副作用**。``doctor`` 调用它是为了让用户在看到"后端可用"之前就
            知道账号状态，而不是跑完采集才发现没登录。
        """
        try:
            data = _as_mapping(self.request("GET", LOGIN_STATUS_PATH).get("data"), "data")
        except CollectorError as exc:
            return None, f"无法确认登录状态：{exc}"

        # 用 `in` 而不是 `.get(...) is True`：键**在不在**是"结构对不对"的问题，
        # 值是什么才是"登没登"的问题。两者混在一起会让结构变更伪装成掉登录。
        if "is_logged_in" not in data:
            return None, (
                f"登录状态响应里没有 is_logged_in 字段（{LOGIN_STATUS_PATH}），"
                "因此无法判断账号状态 —— 这通常意味着上游响应结构已变更。"
            )

        if data["is_logged_in"] is True:
            return True, ""
        return False, (
            "服务在运行，但小红书账号未登录。请先运行 xiaohongshu-mcp 自带的登录工具"
            "扫码登录（登录态由该服务自己保存在本机），再重试。"
        )


class MCPBackend:
    """通过本机运行的 ``xiaohongshu-mcp`` 服务采集。

    满足 :class:`~xhs_pain_miner.collectors.base.CollectorBackend` 协议。

    Attributes:
        name: 后端名称，出现在 CLI 与 ``doctor`` 输出里。
        last_error: 最近一次 :meth:`available` 失败的原因，供 ``doctor`` 展示。
            与插件后端的同名属性是同一个约定 —— 没有它，"cookie 失效"
            这类可操作信息会被压成一句无用的"当前不可用"。
    """

    name = "xiaohongshu-mcp"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Args:
            base_url: 服务地址，默认 ``http://127.0.0.1:18060``。
            token: 服务端启用了鉴权时所需的 Bearer token；未启用时为 ``None``。
            timeout: 单次请求超时（秒）。
        """
        self._client = _Client(base_url, token, timeout)
        self.last_error: str | None = None

    @property
    def base_url(self) -> str:
        """正在对接的服务地址。

        暴露出来是因为"连不上"的排查第一步就是确认**连的是哪个地址** —— 端口
        写成默认值之外的用户尤其需要看到这个实际生效的值。
        """
        return self._client.base_url

    # ---------------------------------------------------------------- 采集 --

    def collect(
        self,
        keyword: str,
        *,
        limit: int,
        max_comments_per_note: int = 20,
    ) -> RawCorpus:
        """采集一个品类的笔记与评论。

        Args:
            keyword: 品类关键词，如 ``"防晒霜"``。
            limit: 最多采集多少篇笔记。**注意它只能截断**：上游一次搜索只返回一页，
                要的比拿到的多时会在语料的 ``warnings`` 里说明。
            max_comments_per_note: 每篇笔记最多保留多少条评论。

        Returns:
            采集结果。``notes`` 为空表示确实没有搜索结果（不是失败）。

        Raises:
            CollectorError: 搜索失败、详情失败率超过 :data:`MAX_FAILURE_RATIO`、
                或响应结构与预期不符。**失败一律抛异常，绝不返回缩水的语料** ——
                空语料会被下游误判为「这个品类没有痛点」。
        """
        term = keyword.strip() if isinstance(keyword, str) else ""
        if not term:
            raise CollectorError(
                "搜索词为空，已放弃本次采集 —— 空搜索词返回的空结果会被下游误读成"
                "「这个品类没有痛点」。"
            )

        if limit <= 0:
            # 调用方明确要求不取结果：这是"确实没有"，不是失败。
            # 与 research/appstore.py 对 limit<=0 的处理保持一致。
            return RawCorpus(keyword=term, backend=self.name)

        feeds = self._search(term)
        selected = feeds[:limit]

        notes: list[RawNote] = []
        comments: list[RawComment] = []
        failures: list[str] = []

        for feed in selected:
            feed_id = str(feed.get("id") or "")
            try:
                detail = self._fetch_detail(feed, max_comments_per_note=max_comments_per_note)
            except CollectorError as exc:
                # 单篇失败不该让整轮白跑 —— 记下来，最后一起说。
                failures.append(f"{feed_id or '(无 id)'}: {exc}")
                continue
            notes.append(_to_note(feed, detail))
            comments.extend(_to_comments(detail, feed_id, max_comments_per_note))

        if selected and len(failures) * 2 > len(selected):
            raise CollectorError(
                f"采集「{term}」时 {len(selected)} 篇笔记里有 {len(failures)} 篇取详情失败，"
                "已中止本次采集。过半失败通常不是个别笔记的问题 —— 常见的几种成因是"
                "服务异常、被平台风控、需要重新登录，也可能是超时设得太短、响应结构"
                "变了、或这批笔记大量被删/转私密（这几种的辨别方式见 "
                "docs/collector-mcp.md 的故障排查一节）。\n"
                "继续跑下去只会产出严重缩水的语料，让下游把「采不到」读成"
                "「这个品类没什么人提」。\n"
                f"第一条失败的原因：{failures[0]}"
            )

        return RawCorpus(
            keyword=term,
            notes=notes,
            comments=comments,
            backend=self.name,
            warnings=self._warnings(term, feeds, selected, failures, limit),
        )

    def available(self) -> bool:
        """后端当前是否可用（供 ``doctor`` 诊断）。

        分两步，为的是让报错能指出**是哪一步**不对：

        1. ``GET /health`` —— 服务在不在（公开端点，不受 token 配置影响）；
        2. ``GET /api/v1/login/status`` —— 账号登没登。

        第 2 步在服务端会开一个浏览器页面，有几秒开销，但不产生数据副作用。
        把登录状态也纳入判断，是因为"服务起着、账号掉线"是本后端最常见的一种
        "看起来能用、一跑就废"的状态。
        """
        if not self._client.healthy():
            self.last_error = (
                f"无法连接 {self._client.base_url}。请确认 xiaohongshu-mcp 服务已启动"
                "（首次使用需先用它自带的登录工具扫码登录）。"
            )
            return False

        state, reason = self._client.login_state()
        if state is not True:
            # state 为 None（问不出来）时 reason 已经说明了原因，原样透出即可 ——
            # 换成一句笼统的"当前不可用"会把唯一有用的信息丢掉。
            self.last_error = reason
            return False

        self.last_error = None
        return True

    # ---------------------------------------------------------------- 内部 --

    def _search(self, keyword: str) -> list[Mapping[str, Any]]:
        """搜索并返回笔记卡片（已剔除直播卡与搜索热词）。

        Raises:
            CollectorError: 请求失败或响应结构不符。
        """
        result = self._client.request("GET", "/api/v1/feeds/search", params={"keyword": keyword})
        data = _as_mapping(result.get("data"), "data")
        # ★ `feeds` 这个键**必须在**，与 note 的 desc 用同一套判据。
        #
        # 上游的 onlyNotes() 用 make([]Feed, 0, ...) 构造成员，保证它至少是 `[]`
        # 而不会是 null —— 所以"键不见了"只可能是结构变了。若把它和 null 一起当成
        # 空数组，用户看到的是"只取到 0 篇笔记…需要更大的语料请换关键词分多次采集"，
        # 于是拿着一个**契约变更**去反复换关键词。这与 note 那边"缺键就报错"的
        # 口径也不一致，同一份文件里两套判据本身就是缺陷。
        _require_keys(data, ("feeds",), "data")
        feeds = _as_list(data["feeds"], "data.feeds")

        notes: list[Mapping[str, Any]] = []
        for item in feeds:
            if not isinstance(item, Mapping):
                continue
            # 上游 service.go 的 SearchFeeds 已经用 onlyNotes 过滤过一遍；这里再过滤
            # 一次是防"上游改了过滤口径"—— 直播卡与热词没有 noteCard，放进来只会
            # 得到一条标题为空、正文为空的笔记，白白占掉 limit 的名额。
            if item.get("modelType") not in (None, "note"):
                continue
            if not isinstance(item.get("noteCard"), Mapping):
                continue
            notes.append(item)
        return notes

    def _fetch_detail(
        self,
        feed: Mapping[str, Any],
        *,
        max_comments_per_note: int,
    ) -> Mapping[str, Any]:
        """取一篇笔记的详情（正文 / 图片 / 互动数 / 评论）。

        Args:
            feed: ``_search`` 返回的卡片，需含 ``id`` 与 ``xsecToken``。
            max_comments_per_note: 本次要保留多少条评论 —— 决定走快路径还是慢路径。

        Returns:
            详情响应里的内层对象，即 ``note`` 与 ``comments`` 的父对象。

        Raises:
            CollectorError: 缺少 ``xsecToken``、请求失败或响应结构不符。
        """
        feed_id = str(feed.get("id") or "")
        xsec_token = str(feed.get("xsecToken") or feed.get("xsec_token") or "")
        if not xsec_token:
            raise CollectorError(
                f"笔记 {feed_id or '(无 id)'} 的搜索结果里没有 xsecToken —— "
                "没有它无法打开详情页。这通常意味着上游响应结构已变更。"
            )

        load_all = max_comments_per_note > FAST_PATH_COMMENT_LIMIT
        payload: dict[str, Any] = {
            "feed_id": feed_id,
            "xsec_token": xsec_token,
            "load_all_comments": load_all,
        }
        # 只在走慢路径时才带 comment_config：该服务的 comment_config 会覆盖掉
        # 一批默认值，默认路径下不传更接近上游的默认行为。
        if load_all:
            payload["comment_config"] = {
                "max_comment_items": max_comments_per_note,
                # 不展开二级回复的"更多"按钮：收益（多几条长尾回复）远小于代价
                # （每篇笔记多一次点击 + 等待），而痛点证据主要由一级评论提供。
                "click_more_replies": False,
                "scroll_speed": "normal",
            }

        result = self._client.request("POST", "/api/v1/feeds/detail", json_body=payload)
        outer = _as_mapping(result.get("data"), "data")
        # 注意这里是**两层** data：外层信封一次，FeedDetailResponse 自己又一次。
        return _as_mapping(outer.get("data"), "data.data")

    def _warnings(
        self,
        keyword: str,
        feeds: Sequence[Mapping[str, Any]],
        selected: Sequence[Mapping[str, Any]],
        failures: Sequence[str],
        limit: int,
    ) -> list[str]:
        """汇总"采集成功了，但结果不达预期"的情况。

        Returns:
            给用户看的提示；没有异常情况时为空列表。

        Note:
            **已知未覆盖的一种情形**：走快路径时，拿回来的评论数可能少于
            ``max_comments_per_note``（快路径不滚动，页面首屏有几条就是几条，
            见 :data:`FAST_PATH_COMMENT_LIMIT`）。这不是错误 —— 那个参数的含义是
            "最多" —— 但它是**静默**的，这里不提示。要确保拿满就走慢路径
            （``--comments`` 调到 10 以上）。
        """

        def excerpt(text: str, width: int = 200) -> str:
            return text if len(text) <= width else f"{text[:width]}…"

        messages: list[str] = []

        if failures:
            messages.append(
                f"有 {len(failures)}/{len(selected)} 篇笔记取详情失败、未计入语料。"
                "**本次的「提及次数」可能因此偏小** —— 它不等于「这个品类没人提」，"
                f"只是这几篇没采到。第一条原因：{excerpt(failures[0])}"
            )

        if len(feeds) < limit:
            messages.append(
                f"本次只取到 {len(feeds)} 篇笔记，少于要求的 {limit} 篇："
                f"xiaohongshu-mcp 的搜索接口一次只返回一页（搜索词「{keyword}」），"
                "上游不滚动、也没有翻页参数，所以重试不会变多。"
                "需要更大的语料请换关键词分多次采集。"
            )

        return messages


def _to_note(feed: Mapping[str, Any], detail: Mapping[str, Any]) -> RawNote:
    """把搜索卡片 + 详情映射成 :class:`RawNote`。

    两处来源的分工：``title`` / ``images`` / ``likes`` 等以**详情**为准（它是笔记页
    上的真实数据），搜索卡片只在详情缺字段时兜底 —— 搜索卡片上的
    ``displayTitle`` 是列表页的截断标题，且没有正文。
    """
    note = _as_mapping(detail.get("note"), "data.data.note")
    _require_keys(note, _NOTE_REQUIRED_KEYS, "data.data.note")

    card = _as_mapping(feed.get("noteCard"), "noteCard")
    card_interact = card.get("interactInfo")
    interact = note.get("interactInfo")
    if not isinstance(interact, Mapping):
        interact = card_interact if isinstance(card_interact, Mapping) else {}

    note_id = str(note.get("noteId") or feed.get("id") or "")
    xsec_token = str(note.get("xsecToken") or feed.get("xsecToken") or "")

    return RawNote(
        note_id=note_id,
        title=str(note.get("title") or card.get("displayTitle") or ""),
        desc=str(note.get("desc") or ""),
        url=_note_url(note_id, xsec_token),
        images=_detail_images(note),
        likes=_parse_count(interact.get("likedCount")),
        collects=_parse_count(interact.get("collectedCount")),
        comments_count=_parse_count(interact.get("commentCount")),
        publish_time=_parse_timestamp(note.get("time")),
        author_hash=_author_hash(note.get("user")),
    )


def _detail_images(note: Mapping[str, Any]) -> list[str]:
    """从详情的 ``imageList`` 里取图片地址。

    优先 ``urlDefault``（小红书 CDN 的默认档位，稳定），回退 ``urlPre``。
    ``fileId`` / ``urlPre`` 之外的字段不取：VLM 只需要一个能下载的地址。
    """
    images: list[str] = []
    for item in _as_list(note.get("imageList"), "data.data.note.imageList"):
        if not isinstance(item, Mapping):
            continue
        url = item.get("urlDefault") or item.get("url_default") or item.get("urlPre") or ""
        if url:
            images.append(str(url))
    return images


def _to_comments(
    detail: Mapping[str, Any],
    note_id: str,
    max_comments_per_note: int,
) -> list[RawComment]:
    """把详情的评论树摊平成 :class:`RawComment` 列表。

    二级评论在响应里是**嵌套**的（``subComments`` 是同一个 ``Comment`` 类型），
    没有独立的父评论字段，所以 ``parent_id`` 只能由嵌套关系推断。摊平后顺序是
    "一级评论 → 它的二级评论 → 下一条一级评论"，便于人工核对。

    Args:
        detail: 详情响应内层对象。
        note_id: 所属笔记 id。
        max_comments_per_note: 上限。**一级评论与二级评论一起计数**，理由见实现。

    Returns:
        评论列表。

    Note:
        **丢掉一条父评论不会连带丢掉它的子树。** 正文为空的一级评论（例如图片式
        评论）自己不进语料，但它的 ``subComments`` 仍然会被映射，``parent_id``
        指向那个没进语料的 id。把这些回复一起吞掉才是错的 —— 它们常常正是痛点
        证据，而丢掉是不出声的。
    """
    comments_obj = detail.get("comments")
    if comments_obj is None:
        return []

    comments_map = _as_mapping(comments_obj, "data.data.comments")
    # 与 note 的 desc / 搜索的 feeds 同一套判据：键恒存在（上游 List 的 json tag
    # 不带 omitempty），所以"键不见了"只可能是结构变更。不查的话，`list` 一改名
    # 就会静默返回 0 条评论 —— 看起来像"这篇笔记没有评论"。
    _require_keys(comments_map, ("list",), "data.data.comments")
    raw_list = _as_list(comments_map["list"], "comments.list")

    limit = max(0, max_comments_per_note)
    if limit == 0:
        return []

    collected: list[RawComment] = []
    for index, item in enumerate(raw_list):
        if not isinstance(item, Mapping):
            continue

        # 先算 id、再映射内容：父评论可能因正文为空被丢掉，但它**仍然是子评论的父**。
        parent_id = _comment_id(item, note_id=note_id, fallback=f"c{index}")
        parent = _comment_from(
            item,
            note_id=note_id,
            parent_id=None,
            comment_id=parent_id,
            path=f"comments.list[{index}]",
        )
        if parent is not None:
            collected.append(parent)

        _require_keys(item, ("subComments",), f"comments.list[{index}]")
        for sub_index, sub in enumerate(
            _as_list(item["subComments"], f"comments.list[{index}].subComments")
        ):
            if not isinstance(sub, Mapping):
                continue
            child = _comment_from(
                sub,
                note_id=note_id,
                parent_id=parent_id,
                # 带父序号：只用 `s{sub_index}` 会让不同父评论下同序号的子评论撞成
                # 同一个 id，而 id 是去重键 —— 撞了就等于把两条不同的发言并成一条。
                comment_id=_comment_id(sub, note_id=note_id, fallback=f"c{index}s{sub_index}"),
                path=f"comments.list[{index}].subComments[{sub_index}]",
            )
            if child is not None:
                collected.append(child)

        # 上限按**摊平后**的总数算。若只按一级评论数算，一条带 50 条回复的评论就能
        # 让单篇笔记的评论数远超调用方给的限额 —— 而限额的意义正是"别让某几篇
        # 笔记把语料结构带偏"。
        if len(collected) >= limit:
            return collected[:limit]

    return collected


def _comment_id(raw: Mapping[str, Any], *, note_id: str, fallback: str) -> str:
    """取出评论的 id；上游没给就合成一个。

    ``comment_id`` 是去重键，因此合成值必须**既稳定又唯一**：

    * **稳定** —— 只依赖嵌套位置，同一份响应解析两次得到同一个值；
    * **唯一** —— ``fallback`` 由 ``_to_comments`` 按 ``父序号 + 子序号`` 给出，
      所以不同父评论下的同序号子评论不会撞车。

    只做到"稳定"是不够的：撞车会让两条不同的发言在去重时被并成一条，而
    「提及次数」是本产品的核心指标。
    """
    raw_id = str(raw.get("id") or raw.get("commentId") or "")
    return raw_id or f"{note_id}#{fallback}"


def _comment_from(
    raw: Mapping[str, Any],
    *,
    note_id: str,
    parent_id: str | None,
    comment_id: str,
    path: str,
) -> RawComment | None:
    """映射单条评论；正文为空时返回 ``None``。

    空正文的评论对分析毫无价值，还会让 ``clean.py`` 多走一遍过滤。**调用方需要
    先算好 ``comment_id`` 再调用**，因为被丢掉的评论仍可能是别人的父评论
    （见 :func:`_to_comments`）。

    Raises:
        CollectorError: 缺少 ``content`` 键 —— 见 :data:`_COMMENT_REQUIRED_KEYS`。
    """
    _require_keys(raw, _COMMENT_REQUIRED_KEYS, path)

    content = str(raw.get("content") or "").strip()
    if not content:
        return None

    return RawComment(
        comment_id=comment_id,
        content=content,
        likes=_parse_count(raw.get("likeCount")),
        parent_id=parent_id,
        note_id=str(raw.get("noteId") or note_id),
        created_at=_parse_timestamp(raw.get("createTime")),
        user_hash=_author_hash(raw.get("userInfo")),
    )
