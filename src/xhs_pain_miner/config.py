"""配置层 —— 全部通过环境变量 / ``.env`` 注入。

配置分四组：

* **LLM**：文本分析用（簇命名、竞品归纳、难度评估）。默认指向 DeepSeek，成本最优。
* **VLM**：图片分析用（``--deep``）。**默认继承 LLM 配置**；由于 DeepSeek 无视觉能力，
  实际使用多模态分析时需要单独配置一个支持视觉的模型（Qwen-VL / GPT-4o / Claude）。
* **Embedding**：向量化用。默认本地 ``bge-small-zh``（零成本，是降本关键）。
* **Collector**：采集后端。默认 ``fixture``（内置脱敏样例），不联网、不触碰平台。

优先级：显式环境变量 > ``.env`` 文件 > 本模块默认值。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover
    from xhs_pain_miner.scoring.opportunity import ScoreWeights

LLMProtocol = Literal["chat", "responses", "messages"]
"""LLM 协议：

* ``chat`` —— OpenAI Chat Completions。覆盖 OpenAI、DeepSeek、Qwen、Kimi、GLM 等。
* ``responses`` —— OpenAI Responses API。
* ``messages`` —— Anthropic Messages API。
"""

CollectorBackendName = Literal["fixture", "plugin", "mcp"]
"""采集后端：

* ``fixture`` —— 内置脱敏样例数据，用于 CI / Demo / 离线验证。
* ``plugin`` —— 加载用户本地自备的采集器插件（见 docs/collector-plugin.md）。
* ``mcp`` —— 对接 xiaohongshu-mcp。
"""


class Settings(BaseSettings):
    """XHS Pain Miner 的全部可配置项。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # 允许用字段名（llm_api_key）而非仅别名（LLM_API_KEY）初始化，
        # load_settings() 与测试都依赖这一点。
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ LLM --
    llm_protocol: LLMProtocol = "chat"
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DASHSCOPE_API_KEY",
        ),
    )
    llm_base_url: str | None = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_concurrency: int = 4
    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    # ------------------------------------------------------------------ VLM --
    # None 表示继承对应的 LLM 配置。DeepSeek 无视觉能力，跑 --deep 时必须单独配置。
    vlm_protocol: LLMProtocol | None = None
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_max_concurrency: int = 2
    vlm_image_max_edge: int = 512
    """上传前把图片长边压缩到该像素值 —— 直接决定 VLM 的 token 消耗。"""
    vlm_max_images_per_note: int = 3
    max_vlm_calls: int | None = None
    """单次运行的 VLM 调用上限。``None`` 表示仅预告成本，不设硬上限。"""

    # ------------------------------------------------------------ Embedding --
    embedding_provider: Literal["local", "api"] = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_batch_size: int = 64

    # -------------------------------------------------------------- Collector --
    collector_backend: CollectorBackendName = "fixture"
    collector_plugin: str | None = Field(
        default=None,
        # 两个名字都接受：XHS_ 前缀更明确、不易与其它工具冲突，是文档推荐写法；
        # 裸名保留兼容。pydantic-settings 不会自动加 XHS_ 前缀，必须在这里显式声明 ——
        # 否则文档里写的 XHS_COLLECTOR_PLUGIN 会被静默忽略，插件永远加载不上。
        validation_alias=AliasChoices("XHS_COLLECTOR_PLUGIN", "COLLECTOR_PLUGIN"),
    )
    """``collector_backend=plugin`` 时的 Python 模块路径，形如 ``my_pkg.my_collector``。"""

    # -- MCP 后端（对接本机运行的 xiaohongshu-mcp 服务）---------------------------
    # 登录态由那个服务自己保管（扫码登录后存在服务端本机），所以这里**没有**
    # cookie 配置项；本程序只负责把地址与可选的鉴权 token 传过去。
    xhs_mcp_url: str = "http://127.0.0.1:18060"
    """``xiaohongshu-mcp`` 服务的地址（环境变量 ``XHS_MCP_URL``）。

    默认值是那个服务的默认监听地址。**不要指向公网地址** —— 采集应当在本机完成，
    把登录态交给远端服务既超出本工具的定位，也让"数据不离开本机"这条设计失效。
    """
    xhs_mcp_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("XHS_MCP_TOKEN", "XHS_MCP_AUTH_TOKEN"),
    )
    """服务端启用了鉴权（``AUTH_TOKEN``）时的 Bearer token。未启用则留空。

    两个名字都接受：``XHS_MCP_TOKEN`` 是本项目的主名，``XHS_MCP_AUTH_TOKEN`` 与被对接
    服务自己的 ``AUTH_TOKEN`` 对得上，配置时不容易张冠李戴。只保存在本机，
    与 LLM 的 API Key 同级对待。
    """
    xhs_mcp_timeout: float = 120.0
    """调用 ``xiaohongshu-mcp`` 的单次请求超时（秒）。

    比其它渠道（15 秒）大一个量级：该服务用浏览器自动化取数据，加载全部评论时
    几十秒是常态。定小了会把"慢"误报成"采集失败"。默认值与
    :data:`~xhs_pain_miner.collectors.mcp.DEFAULT_TIMEOUT` 保持一致 ——
    两处不一致会让"直接构造后端"与"走配置构造"得到不同行为。
    """

    # --------------------------------------------------------- 痛点发现方式 --
    pain_discovery: Literal["taxonomy", "cluster"] = "taxonomy"
    """痛点归集方式。

    * ``taxonomy`` —— LLM 归纳痛点清单 + embedding 分类。**默认**。
    * ``cluster``  —— HDBSCAN 聚类。

    默认用 ``taxonomy`` 是**实测结论**而非偏好：中文短文本（中位 19 字）的
    语义信噪比只有 0.06，HDBSCAN 会把一个 130 条的真实痛点切成 16 片，
    ``size`` 随之失真；而"给定清单做分类"的准确率可达 0.80+。完整数据见
    :mod:`~xhs_pain_miner.pipeline.taxonomy`。

    ``cluster`` 保留下来，是为了在有真实语料时可以对比两者。
    """

    pain_taxonomy_sample_size: int = 240
    """归纳痛点清单时送入 LLM 的样本条数。"""

    pain_max_pains: int = 20
    """痛点清单的上限。压得太低会让不同问题被合并成笼统类别。"""

    pain_match_threshold: float = 0.30
    """判为「命中某个痛点」的最低余弦相似度。

    刻意偏低：短文本的余弦相似度整体压缩在 0.5 附近，阈值定高会把大量真实
    证据挡在门外，让 ``size`` 系统性偏小 —— 那比偶尔混进一条不相干文本更糟。
    """

    # ------------------------------------------------------------ 竞品调研 --
    research_enabled: bool = True
    """是否启用竞品调研。

    关闭后「竞品空白度」因子一律取中性值 —— 注意**不是**取满分：没查过就
    不知道有没有人做过，给高分等于凭空造机会（见 scoring.opportunity 的
    公允性规则）。
    """

    github_token: str | None = None
    """GitHub Token。匿名调用搜索接口约 10 次/分钟，簇多时必然被限流。"""

    research_max_clusters: int = 12
    """最多对多少个痛点簇做竞品调研（按提及量降序）。

    GitHub 搜索接口匿名调用约 10 次/分钟，每个簇要 2-3 次查询。不设上限的话，
    一个 30 簇的分析要跑 9 分钟以上，且必然撞限流 —— 而被限流的簇会退化成
    "调研失败"，白白浪费前面的调用。**超出的簇按中性值处理并如实告知用户**，
    这比"跑一半失败"诚实得多。
    """

    research_max_queries_per_cluster: int = 4
    """每个簇最多发出几条检索词（跨渠道合计）。

    检索词生成阶段的提示词最多让它给 6 条，这里默认再压到 4 条 —— **每一条都是一次
    平台请求**，而 GitHub 的匿名额度约 10 次/分钟：12 个簇 × 4 条 = 48 次请求，
    按 :data:`~xhs_pain_miner.research.github.SEARCH_INTERVAL_ANONYMOUS` 节流也要跑
    近 5 分钟。调大它之前请先配好 ``GITHUB_TOKEN``（额度提到 30 次/分钟），
    否则多出来的词多半以"被限流"收场，反而把该簇的结论打回中性值。
    """

    appstore_country: str = "cn"
    """App Store 检索的商店区域。

    默认中国区：小红书痛点的解法基本是面向中文用户的 App，而同一个词在不同区域的
    召回完全不同。取值传给 iTunes Search API 的 ``country`` 参数（见
    :data:`~xhs_pain_miner.research.appstore.DEFAULT_COUNTRY`）。
    """

    # ----------------------------------------------------------------- 评分 --
    # 五个因子的权重。可调是本产品与黑箱评分产品的差异点之一 —— 觉得竞品更
    # 重要就把 weight_competitor_gap 调到 0.4，分数会立刻重算。
    weight_pain_strength: float = 0.25
    weight_mention_volume: float = 0.20
    weight_growth_trend: float = 0.20
    weight_competitor_gap: float = 0.25
    weight_feasibility: float = 0.10

    # ----------------------------------------------------------------- 限额 --
    max_notes: int = 100
    """开源版单次分析额度上限。"""
    max_comments_per_note: int = 20
    min_cluster_size: int = 3
    """HDBSCAN 最小簇大小。设得过小会产生大量碎片化痛点。"""

    # ----------------------------------------------------------------- 路径 --
    output_dir: Path = Field(default_factory=lambda: Path.cwd())
    db_path: Path = Field(default_factory=lambda: Path.home() / ".xhs-pain-miner" / "db.sqlite")

    # ----------------------------------------------------------------- 隐私 --
    share_results: bool = False
    """是否参与社区众包（上传脱敏结论）。**默认关闭**，必须由用户显式开启。"""
    hash_salt: str = ""
    """本地哈希盐值，用于增强 UID 哈希的不可逆性。

    注意：修改该值会让历史数据的哈希失效（无法跨运行去重）。
    """

    # ------------------------------------------------------------- 继承属性 --
    @property
    def effective_vlm_protocol(self) -> LLMProtocol:
        """VLM 实际使用的协议（未单独配置时继承 LLM）。"""
        return self.vlm_protocol or self.llm_protocol

    @property
    def effective_vlm_model(self) -> str:
        """VLM 实际使用的模型（未单独配置时继承 LLM）。"""
        return self.vlm_model or self.llm_model

    @property
    def effective_vlm_base_url(self) -> str | None:
        """VLM 实际使用的端点（未单独配置时继承 LLM）。"""
        return self.vlm_base_url if self.vlm_base_url is not None else self.llm_base_url

    @property
    def effective_vlm_api_key(self) -> str | None:
        """VLM 实际使用的 Key（未单独配置时继承 LLM）。"""
        return self.vlm_api_key or self.llm_api_key

    @property
    def effective_embedding_api_key(self) -> str | None:
        """Embedding 实际使用的 Key（未单独配置时继承 LLM）。"""
        return self.embedding_api_key or self.llm_api_key

    @property
    def max_notes_hard_limit(self) -> int:
        """开源版的笔记数硬上限（至少为 1）。

        .. note::
           目前**尚未被 src/ 中的任何代码消费** —— 额度限制在 M4 商业化地基阶段接入。
           现在保留它是为了让「免费版额度」这个概念在配置层就有单一事实来源，
           避免 M4 时在多处硬编码。
        """
        return max(self.max_notes, 1)

    # ------------------------------------------------------------------ 工具 --
    def to_weights(self) -> ScoreWeights:
        """把扁平的配置字段组装成评分权重对象。

        延迟 import 是刻意的：``config`` 是核心模块，任何一次 ``import
        xhs_pain_miner`` 都会加载它；而 ``scoring`` 只在真正评分时才需要。
        模块级 import 会让只想跑 ``collect`` 的用户也承担评分模块的加载。
        """
        from xhs_pain_miner.scoring.opportunity import ScoreWeights

        return ScoreWeights(
            pain_strength=self.weight_pain_strength,
            mention_volume=self.weight_mention_volume,
            growth_trend=self.weight_growth_trend,
            competitor_gap=self.weight_competitor_gap,
            feasibility=self.weight_feasibility,
        )

    def masked_llm_key(self) -> str:
        """返回脱敏后的 LLM Key，用于 ``doctor`` 输出。"""
        return _mask(self.llm_api_key)

    def masked_vlm_key(self) -> str:
        """返回脱敏后的 VLM Key，用于 ``doctor`` 输出。"""
        return _mask(self.effective_vlm_api_key)

    def ensure_dirs(self) -> None:
        """确保输出目录与数据库目录存在。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def _mask(secret: str | None) -> str:
    """把密钥脱敏成 ``sk-abc***xyz`` 形式，避免在终端/日志中泄露完整值。"""
    if not secret:
        return "(未设置)"
    if len(secret) <= 10:
        return secret[:2] + "***"
    return f"{secret[:6]}***{secret[-4:]}"


def load_settings(**overrides: object) -> Settings:
    """加载配置，允许用关键字参数覆盖（供 CLI 选项覆盖环境变量使用）。

    Args:
        **overrides: 需要覆盖的字段，值为 ``None`` 的项会被忽略。

    Returns:
        组装好的 :class:`Settings`。

    Raises:
        ValueError: 传入了不存在的配置项。由于 ``Settings`` 允许多余的环境变量
            （``extra="ignore"``），拼错字段名会被静默忽略 —— 那会导致「用户传了
            API Key 却没生效」这类难以排查的问题，所以这里显式拦截。
    """
    clean = {k: v for k, v in overrides.items() if v is not None}
    unknown = sorted(set(clean) - set(Settings.model_fields))
    if unknown:
        available = ", ".join(sorted(Settings.model_fields))
        raise ValueError(f"未知的配置项: {', '.join(unknown)}。可用的配置项: {available}")
    return Settings(**clean)  # type: ignore[arg-type]
