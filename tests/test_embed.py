"""向量化测试。

原则：**不联网、不下载模型**。真实网络路径用 ``httpx.MockTransport`` 注入
（httpx 官方支持的测试手段），本地路径只做「构造 / 惰性」这类不触发下载的断言。

其中 ``test_parses_vectors_in_index_order`` 是**不变式 1 的守卫测试**：
OpenAI 兼容服务不保证 ``data`` 数组顺序，不按 ``index`` 重排会让证据挂到错误的
簇上 —— 程序不崩，结果全错。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from xhs_pain_miner.config import Settings
from xhs_pain_miner.pipeline import embed as embed_module
from xhs_pain_miner.pipeline.deps import MissingDependencyError
from xhs_pain_miner.pipeline.embed import (
    ApiEmbedder,
    Embedder,
    LocalEmbedder,
    build_embedder,
    cosine_similarity,
)

Handler = Callable[[httpx.Request], httpx.Response]


class StubApiEmbedder(ApiEmbedder):
    """把 httpx 传输层换成可编程的 handler，并记录每次请求。

    覆盖 ``_build_client``（而不是去打 httpx 的内部）是 ApiEmbedder 为此预留的
    扩展点。
    """

    def __init__(
        self,
        handler: Handler,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = "sk-test",
        base_url: str | None = "https://example.test/v1",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, timeout=timeout)
        self.handler = handler
        self.requests: list[httpx.Request] = []
        # 测试里不该真的等退避
        self.retry_backoff = 0

    def _build_client(self) -> httpx.Client:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx.Client(transport=httpx.MockTransport(handle), timeout=self.timeout)


def one_hot(index: int, size: int = 4) -> list[float]:
    """第 ``index`` 维为 1 的单位向量 —— 顺序错了就必然被断言抓到。"""
    vector = [0.0] * size
    vector[index] = 1.0
    return vector


def ok_response(vectors: dict[str, list[float]], *, reverse: bool = False) -> Handler:
    """按请求里的 input 逐条返回对应向量，可选地把 data 数组倒序。"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        data = [{"index": i, "embedding": vectors[text]} for i, text in enumerate(payload["input"])]
        if reverse:
            data.reverse()
        return httpx.Response(200, json={"data": data})

    return handler


def make_api_embedder(handler: Handler, **overrides: Any) -> StubApiEmbedder:
    return StubApiEmbedder(handler, **overrides)


# --------------------------------------------------------------------------- #
# cosine_similarity
# --------------------------------------------------------------------------- #


class TestCosineSimilarity:
    """验收工具。用不上 numpy —— 它是纯 Python 的。"""

    def test_identical_vectors_are_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_are_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_ignores_magnitude(self):
        """余弦口径与向量长度无关 —— 归一化与否必须得到同一个数。"""
        assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero_instead_of_dividing_by_zero(self):
        """零向量返回 0.0：两个方向未知的向量谈不上相似。"""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        """维度不同必须报错，而不是按短的那个静默算出个看着正常的数字。"""
        with pytest.raises(ValueError, match="同维"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# LocalEmbedder
# --------------------------------------------------------------------------- #


class TestLocalEmbedder:
    """只测不触发下载的路径。"""

    def test_construction_does_not_load_or_download(self):
        """构造必须是廉价的纯内存操作（doctor 会构造它）。"""
        embedder = LocalEmbedder()
        assert embedder.name == "local"
        assert embedder.is_local is True
        assert embedder.model_name == "BAAI/bge-small-zh-v1.5"
        assert embedder._model is None
        embedder.close()

    def test_dimension_before_first_encode_raises(self):
        """维度未知时抛 RuntimeError，并说明为什么 —— 不是返回 0 蒙混过去。"""
        embedder = LocalEmbedder()
        with pytest.raises(RuntimeError, match="首次 encode"):
            _ = embedder.dimension

    def test_encode_empty_raises_value_error_without_loading_model(self):
        """空输入要在加载模型**之前**被拒绝：不值得为它下载 100MB。"""
        embedder = LocalEmbedder()
        with pytest.raises(ValueError, match="空文本列表"):
            embedder.encode([])
        assert embedder._model is None

    def test_load_failure_is_translated_into_readable_error(self, monkeypatch):
        """离线导致模型下载失败时，给出可照做的提示而不是 transformers 的原始栈。"""

        class ExplodingFactory:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise OSError("We couldn't connect to 'https://huggingface.co' (offline)")

        class FakeModule:
            SentenceTransformer = ExplodingFactory

        monkeypatch.setattr(embed_module, "require", lambda *a, **k: FakeModule)

        embedder = LocalEmbedder("BAAI/bge-small-zh-v1.5")
        with pytest.raises(RuntimeError) as excinfo:
            embedder.encode(["导入太麻烦了"])

        message = str(excinfo.value)
        assert "无法加载本地 embedding 模型" in message
        assert "EMBEDDING_PROVIDER=api" in message
        assert "huggingface" in message  # 原始错误被保留，便于排查
        assert "Traceback" not in message

    def test_missing_dependency_keeps_its_install_hint(self, monkeypatch):
        """缺 sentence-transformers 时抛的是 MissingDependencyError（带安装命令），
        不能被包装成通用的 RuntimeError —— 那样用户就看不到装什么了。"""

        def missing(*args: object, **kwargs: object) -> object:
            raise MissingDependencyError('需要可选依赖，安装命令：pip install -e ".[analysis]"')

        monkeypatch.setattr(embed_module, "require", missing)

        embedder = LocalEmbedder()
        with pytest.raises(MissingDependencyError, match="analysis"):
            embedder.encode(["导入太麻烦了"])

    def test_encode_uses_injected_fake_model(self, monkeypatch):
        """用假模型跑通编码路径：验证归一化与纯 Python 类型（不含 numpy 标量）。"""

        class FakeModel:
            def __init__(self, model_name: str, *, device: str | None = None) -> None:
                self.model_name = model_name
                self.device = device
                self.calls: list[dict[str, object]] = []

            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                self.calls.append({"texts": texts, **kwargs})
                return [[3.0, 4.0] for _ in texts]

        class FakeModule:
            SentenceTransformer = FakeModel

        monkeypatch.setattr(embed_module, "require", lambda *a, **k: FakeModule)

        embedder = LocalEmbedder(batch_size=8)
        vectors = embedder.encode(["a", "b"])

        assert vectors == [[0.6, 0.8], [0.6, 0.8]]
        assert all(type(value) is float for value in vectors[0])
        assert embedder.dimension == 2

        # 构造时给的 batch_size 就是默认批大小；显式传参则覆盖它
        calls = embedder._model.calls
        assert calls[0]["batch_size"] == 8
        assert calls[0]["normalize_embeddings"] is True
        embedder.encode(["c"], batch_size=1)
        assert calls[1]["batch_size"] == 1

        embedder.close()
        assert embedder._model is None


# --------------------------------------------------------------------------- #
# ApiEmbedder
# --------------------------------------------------------------------------- #


class TestApiEmbedder:
    """MockTransport 驱动，无网络。"""

    def test_dimension_before_first_call_raises(self):
        embedder = make_api_embedder(ok_response({}))
        with pytest.raises(RuntimeError, match="首次 encode"):
            _ = embedder.dimension

    def test_dimension_after_call(self):
        texts = {"a": one_hot(0), "b": one_hot(1)}
        embedder = make_api_embedder(ok_response(texts))
        embedder.encode(["a", "b"])
        assert embedder.dimension == 4

    def test_encode_empty_raises_value_error_and_sends_no_request(self):
        embedder = make_api_embedder(ok_response({}))
        with pytest.raises(ValueError, match="空文本列表"):
            embedder.encode([])
        assert embedder.requests == []

    def test_parses_vectors_in_index_order(self):
        """**不变式 1 守卫**：data 顺序被打乱时，必须按 index 重排。"""
        texts = ["t0", "t1", "t2", "t3"]
        vectors = {text: one_hot(i) for i, text in enumerate(texts)}
        embedder = make_api_embedder(ok_response(vectors, reverse=True))

        result = embedder.encode(texts)

        assert result == [one_hot(0), one_hot(1), one_hot(2), one_hot(3)]

    def test_parses_vectors_in_index_order_for_shuffled_payload(self):
        """同上，但打乱得更彻底（用固定的置换，避免随机性带来的不确定性）。"""
        texts = ["t0", "t1", "t2", "t3"]
        vectors = {text: one_hot(i) for i, text in enumerate(texts)}
        permutation = [2, 0, 3, 1]

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            items = [
                {"index": i, "embedding": vectors[text]} for i, text in enumerate(payload["input"])
            ]
            shuffled = [items[i] for i in permutation]
            return httpx.Response(200, json={"data": shuffled})

        result = make_api_embedder(handler).encode(texts)

        assert result == [one_hot(0), one_hot(1), one_hot(2), one_hot(3)]

    def test_normalizes_returned_vectors(self):
        """归一化是硬要求：不归一化会让 HDBSCAN 的欧氏距离与余弦不一致。"""
        embedder = make_api_embedder(ok_response({"a": [3.0, 4.0]}))
        vectors = embedder.encode(["a"])
        assert vectors == [[0.6, 0.8]]
        assert all(type(value) is float for value in vectors[0])

    def test_splits_into_batches_and_keeps_order(self):
        texts = [f"t{i}" for i in range(5)]
        vectors = {text: one_hot(i, 5) for i, text in enumerate(texts)}
        embedder = make_api_embedder(ok_response(vectors, reverse=True))

        result = embedder.encode(texts, batch_size=2)

        assert len(embedder.requests) == 3  # 2 + 2 + 1
        assert result == [one_hot(i, 5) for i in range(5)]
        sizes = [len(json.loads(r.content)["input"]) for r in embedder.requests]
        assert sizes == [2, 2, 1]

    def test_rejects_incomplete_response(self):
        """条目数少于请求数必须报错：静默少一条会让后面的向量整体错位。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 1, "embedding": [0.0, 1.0]},
                    ]
                },
            )

        embedder = make_api_embedder(handler)
        with pytest.raises(RuntimeError, match="条目数不一致|返回 2 条向量"):
            embedder.encode(["a", "b", "c"])

    def test_rejects_duplicated_index(self):
        """index 不是 0..n-1 的排列时无法可靠还原顺序，必须报错。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 0, "embedding": [0.0, 1.0]},
                    ]
                },
            )

        embedder = make_api_embedder(handler)
        with pytest.raises(RuntimeError, match="index"):
            embedder.encode(["a", "b"])

    def test_rejects_response_without_data_array(self):
        embedder = make_api_embedder(lambda request: httpx.Response(200, json={"error": "boom"}))
        with pytest.raises(RuntimeError, match="data 数组"):
            embedder.encode(["a"])

    def test_rejects_non_json_response(self):
        embedder = make_api_embedder(lambda request: httpx.Response(200, text="<html>502</html>"))
        with pytest.raises(RuntimeError, match="不是合法 JSON"):
            embedder.encode(["a"])

    def test_rejects_dimension_change_between_calls(self):
        """同一实例两次调用维度不同 = 网关串了配置，必须立刻失败。"""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            width = 2 if calls["n"] == 1 else 3
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * width}]})

        embedder = make_api_embedder(handler)
        embedder.encode(["a"])
        with pytest.raises(RuntimeError, match="维度"):
            embedder.encode(["b"])

    def test_retries_transient_failure_then_succeeds(self):
        """500 是瞬时故障，退避后重试必须成功。"""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(500, text="upstream boom")
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

        embedder = make_api_embedder(handler)
        assert embedder.encode(["a"]) == [[1.0, 0.0]]
        assert attempts["n"] == 3

    def test_gives_up_after_max_retries(self):
        """一直 503 时最终抛 RuntimeError，而不是返回空向量。"""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(503, text="overloaded")

        embedder = make_api_embedder(handler)
        embedder.max_retries = 2
        with pytest.raises(RuntimeError, match="503"):
            embedder.encode(["a"])
        assert attempts["n"] == 3  # 首次 + 2 次重试

    def test_does_not_retry_auth_failure(self):
        """401 重试只会白花钱，必须立刻失败并给出排查提示。"""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(401, json={"error": {"message": "invalid api key"}})

        embedder = make_api_embedder(handler)
        with pytest.raises(RuntimeError, match="401"):
            embedder.encode(["a"])
        assert attempts["n"] == 1

    def test_network_error_is_retried_then_reported(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        embedder = make_api_embedder(handler)
        embedder.max_retries = 1
        with pytest.raises(RuntimeError, match="connection refused"):
            embedder.encode(["a"])
        assert attempts["n"] == 2

    def test_official_endpoint_without_key_raises_before_calling(self):
        embedder = make_api_embedder(ok_response({}), api_key=None, base_url=None)
        with pytest.raises(RuntimeError, match="API Key"):
            embedder.encode(["a"])
        assert embedder.requests == []

    def test_custom_endpoint_without_key_is_allowed(self):
        """本地推理服务（Ollama / vLLM）不需要 Key，不能因为没 Key 就拒绝。"""
        embedder = make_api_embedder(
            ok_response({"a": [1.0, 0.0]}), api_key=None, base_url="http://localhost:11434/v1"
        )
        assert embedder.encode(["a"]) == [[1.0, 0.0]]
        assert "Authorization" not in embedder.requests[0].headers

    def test_sends_authorization_and_hits_v1_endpoint(self):
        embedder = make_api_embedder(ok_response({"a": [1.0, 0.0]}))
        embedder.encode(["a"])
        request = embedder.requests[0]
        assert str(request.url) == "https://example.test/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-test"
        assert json.loads(request.content) == {"model": "text-embedding-3-small", "input": ["a"]}

    def test_endpoint_does_not_duplicate_v1(self):
        """``base_url`` 已经带 ``/v1`` 时不能再拼一次，否则 404。"""
        embedder = make_api_embedder(ok_response({"a": [1.0, 0.0]}), base_url="https://x.test/v1/")
        embedder.encode(["a"])
        assert str(embedder.requests[0].url) == "https://x.test/v1/embeddings"

    def test_close_is_safe_before_any_call_and_after(self):
        embedder = make_api_embedder(ok_response({"a": [1.0, 0.0]}))
        embedder.close()  # 还没发过请求，不能炸
        embedder.encode(["a"])
        embedder.close()
        assert embedder._client is None

    def test_protocol_isinstance_raises_before_first_encode(self):
        """守卫「惰性维度」这个设计，同时把它的代价钉在测试里。

        ``Embedder`` 是 ``runtime_checkable`` 协议，而 ``isinstance`` 会逐个读取
        协议成员 —— 读到 ``dimension`` 时就会触发下面这个 ``RuntimeError``。
        也就是说：**首次编码前不能用 isinstance 做分派**，要用 ``is_local`` / ``name``。

        这条测试的另一半价值是防止有人"顺手把 dimension 改成主动加载模型"：
        那会让一次属性读取变成 100MB 下载，这里会立刻变红。
        """
        embedder = make_api_embedder(ok_response({"a": [1.0, 0.0]}))
        with pytest.raises(RuntimeError, match="首次 encode"):
            isinstance(embedder, Embedder)

        # 其余成员在首次调用前就是可用的
        assert embedder.name == "api"
        assert embedder.is_local is False
        assert callable(embedder.encode)
        assert callable(embedder.close)

        embedder.encode(["a"])
        assert isinstance(embedder, Embedder)  # 编过一次之后协议检查才成立


# --------------------------------------------------------------------------- #
# build_embedder
# --------------------------------------------------------------------------- #


class TestBuildEmbedder:
    """工厂只做装配，不做 IO。"""

    def test_local_provider(self):
        settings = Settings(embedding_provider="local", embedding_model="BAAI/bge-small-zh-v1.5")
        embedder = build_embedder(settings)
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.is_local is True
        embedder.close()

    def test_local_provider_passes_batch_size(self):
        settings = Settings(embedding_provider="local", embedding_batch_size=7)
        embedder = build_embedder(settings)
        assert isinstance(embedder, LocalEmbedder)
        assert embedder._batch_size == 7
        embedder.close()

    def test_api_provider_inherits_llm_key(self):
        """单独配了 embedding key 就用它，否则回落到 LLM key。"""
        settings = Settings(
            embedding_provider="api",
            embedding_model="text-embedding-3-small",
            embedding_base_url="https://api.siliconflow.cn/v1",
            llm_api_key="sk-from-llm",
        )
        embedder = build_embedder(settings)
        assert isinstance(embedder, ApiEmbedder)
        assert embedder.model == "text-embedding-3-small"
        assert embedder.api_key == "sk-from-llm"
        assert embedder.base_url == "https://api.siliconflow.cn/v1"
        embedder.close()

    def test_api_provider_prefers_dedicated_key(self):
        settings = Settings(
            embedding_provider="api",
            embedding_api_key="sk-embedding",
            llm_api_key="sk-from-llm",
        )
        embedder = build_embedder(settings)
        assert isinstance(embedder, ApiEmbedder)
        assert embedder.api_key == "sk-embedding"


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "https://api.openai.com/v1/embeddings"),
        ("https://api.openai.com", "https://api.openai.com/v1/embeddings"),
        ("https://api.deepseek.com/", "https://api.deepseek.com/v1/embeddings"),
        ("https://api.siliconflow.cn/v1", "https://api.siliconflow.cn/v1/embeddings"),
        ("https://gw.test/v1/embeddings", "https://gw.test/v1/embeddings"),
        ("https://gw.test/openai/v2", "https://gw.test/openai/v2/embeddings"),
    ],
)
def test_endpoint_for(base_url: str | None, expected: str) -> None:
    assert embed_module._endpoint_for(base_url) == expected


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """兜底：任何走到 time.sleep 的测试都不该真的等待。"""
    monkeypatch.setattr(embed_module.time, "sleep", lambda _seconds: None)
    yield
