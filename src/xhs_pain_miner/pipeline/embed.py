"""向量化 —— 把文本单元编码成向量。

两种实现，由 ``Settings.embedding_provider`` 选择：

* :class:`LocalEmbedder` —— ``sentence-transformers`` 跑 ``bge-small-zh``。**默认**。
  2000 条中文短文本在 CPU 上约 10-20 秒，成本为零。这是整个成本模型里最关键
  的一环：它把 LLM 调用量从"每条文本一次"降到"每个簇一次"（约 35 次）。
* :class:`ApiEmbedder` —— 任意 OpenAI 兼容的 ``/v1/embeddings``。适合没有本地
  算力或追求更高质量的场合。

接口约定
--------
**公共接口不出现 numpy 类型**（``encode`` 返回 ``list[list[float]]``）。原因见
:mod:`~xhs_pain_miner.pipeline` 的设计约束 2。

两条实现在这里共同保证的硬约束（改动前请先读 :doc:`m1-interfaces` 第三节）：

1. **顺序对齐**：返回值第 ``i`` 个向量必须对应 ``texts[i]``。本地实现天然满足；
   远程实现**必须按响应里的 ``index`` 重排**（部分服务不保证 ``data`` 数组顺序），
   否则证据会挂到错误的簇上 —— 结果看起来完全正常，只是每个簇里装的是别人的话。
2. **L2 归一化**：所有向量都归一化为单位长度。这样余弦相似度就是点积，
   HDBSCAN 的欧氏距离也与余弦距离单调等价（:func:`cluster.cluster_units` 用欧氏
   距离聚类，正确性依赖这一条）。
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from xhs_pain_miner.pipeline.deps import require

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.config import Settings


DEFAULT_BATCH_SIZE = 64
"""``encode`` 的默认批大小。

同时也是「调用方没有显式传 batch_size」的判定基准：构造时传入的 ``batch_size``
只在调用方不显式覆盖时生效（见 :meth:`LocalEmbedder.encode`）。
"""

DEFAULT_API_BASE_URL = "https://api.openai.com"
"""``base_url=None`` 时使用的官方端点。"""

MAX_RETRIES = 3
"""远程调用的重试次数（不含首次），即最多尝试 ``MAX_RETRIES + 1`` 次。"""

RETRY_BACKOFF_SECONDS = 0.5
"""重试退避基数，第 ``n`` 次重试前等待 ``base * 2**n`` 秒。"""

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
"""可重试的 HTTP 状态码：限流与瞬时故障。4xx 中的其余状态（401/400/404 等）
是配置或请求本身有问题，重试只会白花钱。"""


@runtime_checkable
class Embedder(Protocol):
    """文本编码器的统一接口。"""

    name: str
    """实现名，用于 CLI 展示与成本报告。"""

    dimension: int
    """向量维度。缓存的向量必须校验维度一致，否则跨模型复用缓存会静默出错。"""

    is_local: bool
    """是否本地运行。``False`` 时 ``doctor`` 会提示可能产生费用。"""

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> list[list[float]]:
        """把一批文本编码成向量。

        Args:
            texts: 待编码文本。**顺序必须与返回值严格对应** —— 聚类结果靠下标
                与文本单元对应，错位会让证据挂到错误的簇上。
            batch_size: 批大小。

        Returns:
            与 ``texts`` 等长的向量列表。

        Raises:
            ValueError: 传入空列表。
            xhs_pain_miner.pipeline.deps.MissingDependencyError: 本地依赖缺失。
            RuntimeError: 远程服务调用失败。
        """
        ...

    def close(self) -> None:
        """释放底层资源。"""
        ...


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _normalize(vector: Sequence[float]) -> list[float]:
    """把向量 L2 归一化为单位长度。

    零向量原样返回全零（而不是抛除零错误）—— 零向量在余弦口径下与任何向量都
    不相似，交由 :func:`cosine_similarity` 返回 0.0 表达这个语义。

    Args:
        vector: 待归一化向量。

    Returns:
        单位长度的 ``list[float]``（纯 Python 类型，不含 numpy 标量）。
    """
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def _to_vectors(rows: Any) -> list[list[float]]:
    """把任意序列的序列（numpy 二维数组 / list of list）转成归一化的 ``list[list[float]]``。

    Args:
        rows: 编码器产出的二维结构。

    Returns:
        归一化后的向量列表。

    Raises:
        RuntimeError: 产出的结构不是二维的数值序列。
    """
    try:
        return [_normalize(row) for row in rows]
    except TypeError as exc:  # pragma: no cover - 只有实现出错时才会走到
        raise RuntimeError(f"编码器返回了无法解析的结构: {type(rows).__name__}") from exc


def _endpoint_for(base_url: str | None) -> str:
    """由 ``base_url`` 推出 ``/embeddings`` 端点。

    用户配置的 ``base_url`` 有三种常见写法，都必须能用：

    * ``https://api.openai.com`` → ``https://api.openai.com/v1/embeddings``
    * ``https://api.siliconflow.cn/v1`` → ``.../v1/embeddings``（**不能**拼成 ``/v1/v1/``）
    * ``https://my.gateway/llm/embeddings`` → 原样使用

    Args:
        base_url: 用户配置的端点，``None`` 表示官方默认。

    Returns:
        完整的 embeddings 端点 URL。
    """
    base = (base_url or DEFAULT_API_BASE_URL).strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if re.search(r"/v\d+$", base):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


# --------------------------------------------------------------------------- #
# 本地实现
# --------------------------------------------------------------------------- #


class LocalEmbedder:
    """基于 ``sentence-transformers`` 的本地编码器。

    Args:
        model_name: 模型名，默认 ``BAAI/bge-small-zh-v1.5``。
        batch_size: 批大小。作为 :meth:`encode` 未显式传参时的默认值。
        device: 设备（``"cpu"`` / ``"mps"`` / ``"cuda"``）。``None`` 表示自动。

    Note:
        首次调用会下载模型（约 100MB）到 ``~/.cache/huggingface``。离线环境需要
        预先下载或改用 :class:`ApiEmbedder` —— 这个失败要给出可读提示，
        而不是让 transformers 抛出一大堆栈。

    Note:
        **构造本类不会加载模型、不会联网、不会读磁盘上的模型缓存** —— 模型在第一次
        :meth:`encode` 时才载入（``doctor`` 之类的诊断路径会构造它，不能让一次
        诊断触发 100MB 下载）。因此 ``dimension`` 在首次编码前不可知，
        访问会抛 ``RuntimeError``。
    """

    name = "local"
    is_local = True

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """向量维度，由模型决定。

        Raises:
            RuntimeError: 模型尚未加载（还没调用过 :meth:`encode`），维度未知。
                这里**不会**为了拿到维度而提前加载模型：那会把一次属性读取变成
                100MB 下载。
        """
        if self._dimension is None:
            raise RuntimeError(
                f"本地编码器 {self.model_name!r} 的向量维度在首次 encode 之前未知"
                "（模型是惰性加载的，构造与读取属性都不会触发下载）。\n"
                "如果你需要维度信息，请先调用一次 encode()。"
            )
        return self._dimension

    @dimension.setter
    def dimension(self, value: int) -> None:
        """赋值一律拒绝。

        存在的唯一理由是 :class:`Embedder` 协议把 ``dimension`` 声明成了可写变量，
        而 mypy 不接受「只读属性去实现可写变量」。赋一个假维度会让向量缓存的维度
        校验形同虚设，所以这里直接报错而不是静默接受。
        """
        raise AttributeError(  # pragma: no cover - 正常路径不会赋值
            f"dimension 由模型决定，不可赋值（尝试写入 {value!r}）。"
        )

    def _ensure_model(self) -> Any:
        """惰性加载模型，把各类加载失败翻译成一句可读的话。"""
        if self._model is not None:
            return self._model

        # 缺 sentence-transformers 时给出安装命令而不是裸 ImportError
        module = require("sentence_transformers", purpose="本地向量化（LocalEmbedder）")
        factory = getattr(module, "SentenceTransformer", None)
        if factory is None:  # pragma: no cover - 依赖被替换成同名假模块时才会走到
            raise RuntimeError(
                "sentence_transformers 已安装但缺少 SentenceTransformer，"
                '请重装：pip install -e ".[analysis]"'
            )

        try:
            model = factory(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(self._load_error_message(exc)) from exc

        self._model = model
        return model

    def _load_error_message(self, exc: BaseException) -> str:
        """把模型加载失败翻译成用户能照做的提示。

        最常见的失败原因是离线环境（首次使用需要下载约 100MB 权重），
        而 transformers / huggingface_hub 的原始栈把这一点埋得很深。
        """
        return (
            f"无法加载本地 embedding 模型 {self.model_name!r}。\n"
            "最常见的原因是：首次使用需要从 HuggingFace 下载模型权重（约 100MB），"
            "而当前环境访问不了外网。可选的解决办法：\n"
            "  1) 联网后重试一次完成下载，权重会缓存到 ~/.cache/huggingface；\n"
            "  2) 把 EMBEDDING_MODEL 指向本地已下载的模型目录；\n"
            "  3) 改用远程 embedding：设置 EMBEDDING_PROVIDER=api 与 "
            "EMBEDDING_API_KEY；\n"
            "  4) 若只想用采集/诊断功能，可以忽略本功能。\n"
            f"原始错误（{type(exc).__name__}）：{exc}"
        )

    def encode(
        self, texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """编码。模型加载是惰性的，第一次调用时才真正下载 / 载入。

        Args:
            texts: 待编码文本，顺序与返回值严格对应。
            batch_size: 批大小。未显式传参（即取默认的 :data:`DEFAULT_BATCH_SIZE`）时
                使用构造时设定的批大小。

        Returns:
            与 ``texts`` 等长的、已 L2 归一化的向量列表。

        Raises:
            ValueError: ``texts`` 为空。
            RuntimeError: 模型加载失败（离线 / 模型名错误），消息里带排查步骤。
        """
        # 先校验入参再加载模型：空输入不该付出一次 100MB 下载的代价
        if len(texts) == 0:
            raise ValueError(
                "encode() 收到空文本列表。向量与文本必须一一对应，空输入没有任何意义 —— "
                "如果你是想跳过向量化，请在上游就短路，不要调用 encode()。"
            )

        model = self._ensure_model()
        effective_batch = self._batch_size if batch_size == DEFAULT_BATCH_SIZE else batch_size
        if effective_batch <= 0:
            raise ValueError(f"batch_size 必须为正数，收到 {effective_batch}")

        rows = model.encode(
            list(texts),
            batch_size=effective_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = _to_vectors(rows)

        if len(vectors) != len(texts):
            # 顺序对齐是不可变式 1，宁可直接失败也不要让下标错位
            raise RuntimeError(
                f"本地模型返回了 {len(vectors)} 个向量，但输入是 {len(texts)} 条文本。"
                "向量与文本必须一一对应，继续下去会让证据挂到错误的簇上。"
            )

        dimension = len(vectors[0])
        self._dimension = dimension
        return vectors

    def close(self) -> None:
        """释放模型占用的内存。"""
        self._model = None


# --------------------------------------------------------------------------- #
# 远程实现
# --------------------------------------------------------------------------- #


class ApiEmbedder:
    """基于 OpenAI 兼容 ``/v1/embeddings`` 的远程编码器。

    Args:
        model: 模型名。
        api_key: 密钥。``None`` 时只允许配合自定义 ``base_url`` 使用（本地推理
            服务如 Ollama / vLLM 通常不需要 Key）；此时连官方端点会在发请求前
            直接报错，而不是拿到一个 401。
        base_url: 端点。``None`` 表示用官方默认。``/v1`` 与 ``/v1/embeddings``
            两种写法都能识别，不会拼出 ``/v1/v1/embeddings``。
        timeout: 单次请求超时（秒）。

    Note:
        重试次数与退避基数可在实例上覆盖（``max_retries`` / ``retry_backoff``），
        测试把它设为 0 以跳过等待。
    """

    name = "api"
    is_local = False

    max_retries = MAX_RETRIES
    """重试次数（不含首次）。"""

    retry_backoff = RETRY_BACKOFF_SECONDS
    """退避基数（秒），实际等待 ``retry_backoff * 2**n``。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._dimension: int | None = None
        self._client: httpx.Client | None = None

    @property
    def dimension(self) -> int:
        """向量维度，首次调用后才有值；调用前访问应抛 ``RuntimeError``。

        Raises:
            RuntimeError: 还没成功调用过任何一次 ``encode``。
        """
        if self._dimension is None:
            raise RuntimeError(
                f"远程编码器（{self.model}）的向量维度在首次 encode 之前未知 —— "
                "服务端返回的向量才决定维度，本地无从推算。\n"
                "请先调用一次 encode()；如果你只是想知道配置是否可用，"
                "跑一次 doctor 或对一两条文本做冒烟编码即可。"
            )
        return self._dimension

    @dimension.setter
    def dimension(self, value: int) -> None:
        """赋值一律拒绝。理由同 :attr:`LocalEmbedder.dimension`。"""
        raise AttributeError(  # pragma: no cover - 正常路径不会赋值
            f"dimension 由服务端返回的向量决定，不可赋值（尝试写入 {value!r}）。"
        )

    # ------------------------------------------------------------- HTTP 层 --
    def _build_client(self) -> httpx.Client:
        """构造 HTTP 客户端。

        独立成一个方法是为了让测试注入 ``httpx.MockTransport``（子类覆盖即可），
        而不必去 monkeypatch httpx 内部实现。
        """
        return httpx.Client(timeout=self.timeout)

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _headers(self) -> dict[str, str]:
        """组装请求头。

        Raises:
            RuntimeError: 走官方端点但没有 Key。
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.base_url is None:
            raise RuntimeError(
                "远程 embedding 需要 API Key，但 EMBEDDING_API_KEY（或 LLM_API_KEY）未设置。\n"
                "请设置环境变量或在 .env 中配置；若要连本地推理服务，"
                "请同时设置 EMBEDDING_BASE_URL（本地服务通常不需要 Key）。"
            )
        return headers

    def _post(self, batch: Sequence[str]) -> list[list[float]]:
        """发一批请求，返回**按输入顺序**排好的向量。"""
        client = self._ensure_client()
        payload: dict[str, Any] = {"model": self.model, "input": list(batch)}
        response = client.post(self._endpoint(), json=payload, headers=self._headers())

        if response.status_code >= 400:
            raise _HttpError(response.status_code, _body_snippet(response))

        return self._parse(response, expected=len(batch))

    def _endpoint(self) -> str:
        return _endpoint_for(self.base_url)

    def _parse(self, response: httpx.Response, *, expected: int) -> list[list[float]]:
        """解析响应，**按 ``index`` 字段重排**（不变式 1）。

        Args:
            response: 已确认状态码正常的响应。
            expected: 本批请求的文本条数。

        Returns:
            与输入同序的归一化向量。

        Raises:
            RuntimeError: 结构不符、条目数不符或 ``index`` 不构成 ``0..n-1``。
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"embedding 服务返回的不是合法 JSON（HTTP {response.status_code}）："
                f"{_body_snippet(response)}"
            ) from exc

        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"embedding 响应缺少 data 数组，实际内容：{_truncate(payload)}")
        if len(items) != expected:
            raise RuntimeError(
                f"embedding 响应返回 {len(items)} 条向量，但本批请求了 {expected} 条。"
                "条目数不一致说明服务端做了截断，继续下去向量会与文本错位。"
            )

        try:
            indices = [int(item["index"]) for item in items]
            embeddings = [item["embedding"] for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"embedding 响应的 data 条目缺少 index/embedding 字段：{_truncate(items)}"
            ) from exc

        # 服务端不保证顺序，甚至可能漏给条目 —— 只有 index 恰好构成 0..n-1 才能安全重排
        if sorted(indices) != list(range(expected)):
            raise RuntimeError(
                f"embedding 响应的 index 字段不是 0..{expected - 1} 的排列（实际 {indices}）。"
                "无法可靠地还原输入顺序，而顺序错位会让证据挂到错误的簇上。"
            )

        ordered = [None] * expected
        for position, embedding in zip(indices, embeddings):
            ordered[position] = embedding

        vectors = _to_vectors(ordered)
        dimension = len(vectors[0])
        if self._dimension is not None and dimension != self._dimension:
            raise RuntimeError(
                f"同一批调用里出现了两种向量维度（{self._dimension} vs {dimension}）。"
                "服务端换模型或网关串了配置，缓存的向量会与新向量混在一起静默出错。"
            )
        self._dimension = dimension
        return vectors

    # -------------------------------------------------------------- 编码 --
    def encode(
        self, texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """分批调用远程服务。

        必须处理两件事：单批失败时的重试与退避，以及**返回顺序**与输入一致
        （部分服务不保证 ``data`` 数组顺序，必须按 ``index`` 字段重排）。

        Args:
            texts: 待编码文本，顺序与返回值严格对应。
            batch_size: 批大小。

        Returns:
            与 ``texts`` 等长的、已 L2 归一化的向量列表。

        Raises:
            ValueError: ``texts`` 为空，或 ``batch_size`` 非正。
            RuntimeError: 重试后仍失败，或响应结构不可信。
        """
        if len(texts) == 0:
            raise ValueError(
                "encode() 收到空文本列表。向量与文本必须一一对应，空输入没有任何意义。"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，收到 {batch_size}")

        text_list = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            vectors.extend(self._encode_batch_with_retry(batch))
        return vectors

    def _encode_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """发送一批，失败时按指数退避重试。

        Raises:
            RuntimeError: 全部尝试都失败，或遇到不可重试的错误（如 401）。
        """
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return self._post(batch)
            except _HttpError as exc:
                if not exc.retryable:
                    raise RuntimeError(exc.message) from exc
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < attempts - 1:
                time.sleep(self.retry_backoff * (2**attempt))

        raise RuntimeError(
            f"调用 embedding 服务失败（已尝试 {attempts} 次，共 {len(batch)} 条文本）：{last_error}"
        )

    def close(self) -> None:
        """关闭 HTTP 连接。"""
        if self._client is not None:
            self._client.close()
            self._client = None


class _HttpError(Exception):
    """一次 HTTP 调用失败。``retryable`` 决定是否值得重试。"""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        self.retryable = status_code in RETRYABLE_STATUS
        hint = "" if self.retryable else _status_hint(status_code)
        super().__init__(f"embedding 服务返回 HTTP {status_code}：{body}{hint}")
        self.message = str(self)


def _status_hint(status_code: int) -> str:
    """给不可重试的错误加一句能照做的提示。"""
    if status_code in (401, 403):
        return "\n（密钥无效或没有该模型的权限：检查 EMBEDDING_API_KEY。）"
    if status_code == 404:
        return "\n（端点或模型名不对：检查 EMBEDDING_BASE_URL 与 EMBEDDING_MODEL。）"
    if status_code == 400:
        return "\n（请求被拒绝：多半是模型名不被该服务支持。）"
    return ""


def _body_snippet(response: httpx.Response) -> str:
    """截取响应体用于报错。限长，避免把整页 HTML 错误塞进日志。"""
    try:
        text = response.text
    except Exception:  # pragma: no cover - httpx 读取失败时才会走到
        return "(无法读取响应体)"
    return _truncate(text)


def _truncate(value: Any, limit: int = 300) -> str:
    """把任意值转成限长的字符串表示。"""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "...(截断)"


# --------------------------------------------------------------------------- #
# 工厂与工具
# --------------------------------------------------------------------------- #


def build_embedder(settings: Settings) -> Embedder:
    """按配置构造编码器。

    Args:
        settings: 全局配置。

    Returns:
        ``embedding_provider == "local"`` 时返回 :class:`LocalEmbedder`，
        否则返回 :class:`ApiEmbedder`。

    Note:
        构造不会触发模型下载或网络请求；失败会推迟到第一次 ``encode``，
        这样用户可以在真正花钱 / 花时间之前先看到成本预告。
    """
    if settings.embedding_provider == "local":
        return LocalEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size)
    return ApiEmbedder(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def normalize(vector: Sequence[float]) -> list[float]:
    """把向量 L2 归一化为单位长度（公共 API）。

    与 :func:`cosine_similarity` 配对：归一化后点积即余弦相似度。

    痛点分类必须走这里：**质心是若干向量的平均，不再是单位长度**。直接对未
    归一化的质心做点积，得到的是一个随样本数变化的量 —— 相似度阈值就再也
    不可比了，而阈值正是"这条文本属不属于任何一个已知痛点"的判据。

    Args:
        vector: 待归一化向量。

    Returns:
        单位长度的 ``list[float]``。零向量原样返回（全零），由调用方按
        "与任何方向都不相似"处理。
    """
    return _normalize(vector)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """两个向量的余弦相似度。

    用于**验收**而非生产路径：验证「同一簇内的文本确实语义相近」是聚类质量的
    直接证据，比只看簇大小有说服力得多。零向量返回 0.0（而不是抛除零错误）。

    Args:
        a: 向量。
        b: 向量。

    Returns:
        余弦相似度，值域 ``[-1.0, 1.0]``；任一向量为零向量时返回 ``0.0``。

    Raises:
        ValueError: 两个向量维度不一致（静默按短的那个算会得到一个看着正常的
            错误数字）。
    """
    if len(a) != len(b):
        raise ValueError(f"余弦相似度要求两个向量同维，收到 {len(a)} 与 {len(b)}")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        norm_a += fx * fx
        norm_b += fy * fy

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
