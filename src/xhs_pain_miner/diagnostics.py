"""环境诊断 —— ``xhs-pain-miner doctor`` 的实现。

诊断只做本地检查，**不发任何网络请求**（避免拖慢命令，也避免在用户不知情时
把 API Key 送到网络上做连通性测试）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.config import Settings
from xhs_pain_miner.llm.factory import describe_protocol

Status = Literal["ok", "warn", "fail"]
"""``ok`` 正常 / ``warn`` 影响部分功能 / ``fail`` 阻塞运行。"""

MIN_PYTHON = (3, 10)

CORE_MODULES = ("click", "pydantic", "pydantic_settings", "openai", "anthropic", "rich")
ANALYSIS_MODULES = ("numpy", "sklearn", "sentence_transformers")
VLM_MODULES = ("PIL",)


@dataclass(slots=True)
class Check:
    """一项诊断结果。"""

    name: str
    status: Status
    detail: str
    hint: str = ""

    @property
    def passed(self) -> bool:
        """该项是否未阻塞运行。"""
        return self.status != "fail"


def _has_module(name: str) -> bool:
    """检查模块是否可导入（不触发实际导入）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _check_python() -> Check:
    """检查 Python 版本。"""
    version = sys.version_info
    current = f"{version.major}.{version.minor}.{version.micro}"
    required = ".".join(str(p) for p in MIN_PYTHON)
    if (version.major, version.minor) >= MIN_PYTHON:
        return Check("Python 版本", "ok", f"{current}（要求 >= {required}）")
    return Check(
        "Python 版本",
        "fail",
        f"{current} 低于要求的 {required}",
        hint=f"请升级到 Python {required}+",
    )


def _check_modules(name: str, modules: tuple[str, ...], *, optional: str) -> Check:
    """检查一组依赖是否安装。"""
    missing = [m for m in modules if not _has_module(m)]
    if not missing:
        return Check(name, "ok", f"{len(modules)} 个依赖均已安装")
    return Check(
        name,
        "warn",
        f"缺少: {', '.join(missing)}",
        hint=f'pip install -e ".[{optional}]"',
    )


def _check_llm(settings: Settings) -> Check:
    """检查文本分析用的 LLM 配置。"""
    if not settings.llm_api_key:
        return Check(
            "LLM 配置",
            "fail",
            f"未配置 API Key（协议 {settings.llm_protocol}，模型 {settings.llm_model}）",
            hint=(
                "在 .env 或环境变量中设置 LLM_API_KEY"
                "（也兼容 DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY）"
            ),
        )
    protocol_note = describe_protocol(settings.llm_protocol)
    return Check(
        "LLM 配置",
        "ok",
        f"{settings.llm_model} | {settings.masked_llm_key()} | {protocol_note}",
    )


def _check_vlm(settings: Settings) -> Check:
    """检查 ``--deep`` 用的视觉模型配置。

    DeepSeek 系模型没有视觉能力，因此当 VLM 配置完全继承 LLM 且协议指向
    DeepSeek 时，需要提醒用户单独配置一个多模态模型。
    """
    if settings.vlm_model is None:
        base_url = (settings.effective_vlm_base_url or "").lower()
        looks_text_only = "deepseek" in base_url or settings.effective_vlm_model.startswith(
            "deepseek"
        )
        if looks_text_only:
            return Check(
                "VLM 配置",
                "warn",
                f"当前继承的模型 {settings.effective_vlm_model} 没有视觉能力，--deep 会失败",
                hint=(
                    "单独配置一个多模态模型，例如：\n"
                    "      VLM_MODEL=qwen-vl-max\n"
                    "      VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                    "      VLM_API_KEY=sk-xxx"
                ),
            )
    if not settings.effective_vlm_api_key:
        return Check("VLM 配置", "warn", "未配置 API Key，--deep 不可用")
    return Check(
        "VLM 配置",
        "ok",
        f"{settings.effective_vlm_model} | {settings.masked_vlm_key()}",
    )


def _check_embedding(settings: Settings) -> Check:
    """检查 embedding 配置。"""
    if settings.embedding_provider == "local":
        if _has_module("sentence_transformers"):
            return Check(
                "Embedding 配置",
                "ok",
                f"本地模型 {settings.embedding_model}（零 API 成本）",
            )
        return Check(
            "Embedding 配置",
            "warn",
            "配置为本地模型但未安装 sentence-transformers",
            hint='pip install -e ".[analysis]"',
        )
    return Check("Embedding 配置", "ok", f"远程 API：{settings.embedding_model}")


def _check_collector(settings: Settings) -> Check:
    """检查采集后端是否可用。"""
    try:
        collector = build_collector(settings)
    except CollectorError as exc:
        return Check(
            "采集后端",
            "fail",
            f"{settings.collector_backend} 不可用: {exc}",
            hint="改用内置样例数据重试：--backend fixture",
        )

    if collector.available():
        return Check("采集后端", "ok", collector.name)

    # 插件抛出的异常原因（如「cookie 已失效」）是最有可操作性的信息，不能吞掉
    reason = getattr(collector, "last_error", None)
    detail = f"{collector.name} 当前不可用"
    if reason:
        detail = f"{detail}：{reason}"

    return Check(
        "采集后端",
        "warn",
        detail,
        hint=(
            "若为自备插件，请检查登录态；若尚未配置采集器，可先用 --backend fixture 体验完整流程"
        ),
    )


def _nearest_existing(path: Path) -> Path:
    """返回 ``path`` 本身或其最近的已存在祖先。"""
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _check_paths(settings: Settings) -> Check:
    """检查输出目录与数据库目录是否可写。

    这里**不会创建任何目录**。``doctor`` 被文档描述为"只做本地检查"，
    不该在用户看到诊断结论之前就先写用户主目录 —— 检查最近一个已存在祖先的
    可写性即可达到同样目的。
    """
    problems: list[str] = []
    for label, path in (("输出目录", settings.output_dir), ("数据库目录", settings.db_path.parent)):
        target = path if path.exists() else _nearest_existing(path)
        if path.exists():
            writable = path.is_dir() and os.access(path, os.W_OK)
            shown = str(path)
        else:
            writable = os.access(target, os.W_OK)
            shown = f"{path}（尚不存在，将在实际运行时创建）"
        if not writable:
            problems.append(f"{label} {shown}: 不可写（检查 {target} 的权限）")

    if problems:
        return Check("路径权限", "fail", "; ".join(problems))
    return Check(
        "路径权限",
        "ok",
        f"输出 {settings.output_dir} | 数据库 {settings.db_path}",
    )


def _check_privacy(settings: Settings) -> Check:
    """确认众包上传处于关闭状态（默认且推荐）。"""
    if settings.share_results:
        return Check(
            "隐私设置",
            "warn",
            "结果共享已开启 —— 脱敏结论会上传到云端",
            hint="如非有意为之，请设置 SHARE_RESULTS=false",
        )
    return Check("隐私设置", "ok", "结果共享已关闭（原文与个人信息不会离开本机）")


def run_diagnostics(settings: Settings) -> list[Check]:
    """执行全部诊断项。

    Args:
        settings: 全局配置。

    Returns:
        诊断结果列表，顺序固定，便于输出对齐。
    """
    return [
        _check_python(),
        _check_modules("核心依赖", CORE_MODULES, optional="dev"),
        _check_modules("分析依赖", ANALYSIS_MODULES, optional="analysis"),
        _check_modules("图片依赖", VLM_MODULES, optional="vlm"),
        _check_llm(settings),
        _check_vlm(settings),
        _check_embedding(settings),
        _check_collector(settings),
        _check_paths(settings),
        _check_privacy(settings),
    ]


def has_blocking_issue(checks: list[Check]) -> bool:
    """判断诊断结果中是否存在阻塞项。"""
    return any(check.status == "fail" for check in checks)
