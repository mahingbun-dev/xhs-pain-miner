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
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    xhs_cookie: str | None = None
    """仅在使用 MCP 后端时需要，且仅保存在本机。"""

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
