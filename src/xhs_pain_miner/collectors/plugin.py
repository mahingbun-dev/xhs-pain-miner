"""插件后端 —— 加载**用户本机自备**的采集器。

为什么需要插件机制
------------------
第三方采集器的许可证通常不允许商业使用或再分发（例如 MediaCrawler 的
「非商业学习许可」）。把它们的代码放进本仓库，会让整个项目从内部违反上游许可，
进而污染所有下游使用者的权属。

因此本仓库采取与 Playwright 相同的策略：**只定义协议，不携带实现**。
拥有合法采集器的用户按 :mod:`xhs_pain_miner.collectors.base` 的协议写一个薄适配器，
通过 ``XHS_COLLECTOR_PLUGIN`` 指向它即可。

用法
----
插件可以是一个模块路径，也可以是一个 ``.py`` 文件路径::

    # 方式一：模块路径（需在 sys.path 上）
    export XHS_COLLECTOR_PLUGIN=my_collectors.xhs_backend

    # 方式二：文件路径
    export XHS_COLLECTOR_PLUGIN=/Users/me/xhs_plugin.py

插件模块需提供以下三者之一：

* ``BACKEND`` —— 一个 :class:`~xhs_pain_miner.collectors.base.CollectorBackend` 实例
* ``backend`` —— 同上（小写）
* ``create_backend()`` —— 返回该实例的工厂函数

安全提示
--------
插件会在你的 Python 进程内执行任意代码。请只加载你信任的来源 —— 与安装任何
pip 包的风险等级相同。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from xhs_pain_miner.collectors.base import CollectorBackend, CollectorError
from xhs_pain_miner.models import RawComment, RawCorpus, RawNote

_ATTR_CANDIDATES = ("BACKEND", "backend", "create_backend")


def _module_name_for(path: Path) -> str:
    """为插件文件生成稳定的唯一模块名。

    用固定名字（如 ``xhs_collector_plugin``）会让同进程加载第二个插件时覆盖掉第一个，
    也会让插件的内部自引用拿到错误的模块。
    """
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    return f"xhs_collector_plugin_{path.stem}_{digest}"


def _check_elements(items: object, expected: type, field: str, plugin_name: str) -> None:
    """校验 ``RawCorpus`` 内部列表的元素类型。

    只检查最外层 ``isinstance(result, RawCorpus)`` 是不够的 ——
    插件很容易构造出 ``RawCorpus(notes=[{"title": "x"}])`` 这种「外壳正确、
    内脏错误」的对象，而它会在下游的 ``note.images`` 处抛出
    ``AttributeError``，给用户一个完全无法定位问题的裸 traceback。

    **必须是 ``list``**：``list`` 之外的「可迭代但非序列」对象（生成器、
    ``dict.keys()`` 等）虽然能通过 ``iter()``，却会让下游的 ``len()`` 抛
    ``TypeError``；而且校验过程本身会把生成器耗尽，造成静默丢数据。

    Raises:
        CollectorError: 字段不是 list，或存在类型不符的元素。
    """
    if not isinstance(items, list):
        raise CollectorError(
            f"插件 {plugin_name!r} 返回的 RawCorpus.{field} 必须是 list，"
            f"收到 {type(items).__name__}。请参考 docs/collector-plugin.md 修正插件实现。"
        )

    for index, item in enumerate(items):
        if not isinstance(item, expected):
            raise CollectorError(
                f"插件 {plugin_name!r} 返回的 RawCorpus.{field}[{index}] 是 "
                f"{type(item).__name__}，而不是 {expected.__name__}。"
                "请参考 docs/collector-plugin.md 修正插件实现。"
            )


class _ValidatedBackend:
    """包装插件后端，在返回值层面兜住协议契约。

    ``@runtime_checkable`` 的 Protocol 只检查方法存在，不检查返回类型或结构。
    这一层把插件可能造成的失败提前转成可读的 :class:`CollectorError`。
    """

    def __init__(self, inner: CollectorBackend) -> None:
        self._inner = inner
        self.name = inner.name  # type: ignore[attr-defined]
        self.last_error: str | None = None
        """最近一次 ``available()`` 失败的原因，供 ``doctor`` 向用户展示。

        没有它的话，插件抛出的「cookie 已失效，请重新登录」这类可操作信息
        会被完全吞掉，用户只能看到一句无用的「当前不可用」。
        """

    def collect(
        self,
        keyword: str,
        *,
        limit: int,
        max_comments_per_note: int = 20,
    ) -> RawCorpus:
        """调用插件后端并校验其返回值（含内部元素类型）。"""
        try:
            result: Any = self._inner.collect(
                keyword, limit=limit, max_comments_per_note=max_comments_per_note
            )
        except (KeyboardInterrupt, SystemExit):
            # 用户中断与进程退出语义必须原样透传 —— 不能被包装成「采集失败」
            raise
        except Exception as exc:  # noqa: BLE001 - 插件代码可能抛出任意异常
            raise CollectorError(
                f"插件 {self.name!r} 的 collect() 抛出 {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(result, RawCorpus):
            raise CollectorError(
                f"插件 {self.name!r} 的 collect() 返回了 {type(result).__name__}，"
                "而不是 RawCorpus。请参考 docs/collector-plugin.md 修正插件实现。"
            )

        _check_elements(result.notes, RawNote, "notes", self.name)
        _check_elements(result.comments, RawComment, "comments", self.name)
        return result

    def available(self) -> bool:
        """转发可用性检查，并把插件抛出的异常转成「不可用」+ 记录原因。"""
        try:
            ok = bool(self._inner.available())
        except Exception as exc:  # noqa: BLE001 - 插件代码可能抛出任意异常
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.last_error = None
        return ok


def _load_module(module_ref: str) -> ModuleType:
    """按模块路径或文件路径加载模块。

    Args:
        module_ref: ``my_pkg.my_module`` 形式，或以 ``.py`` 结尾的文件路径。

    Returns:
        已加载的模块对象。

    Raises:
        CollectorError: 模块无法导入。
    """
    candidate = Path(module_ref).expanduser()
    is_file_ref = module_ref.endswith(".py") or candidate.is_file()

    if is_file_ref:
        path = candidate.resolve()
        if not path.is_file():
            raise CollectorError(f"插件文件不存在: {path}")
        spec = importlib.util.spec_from_file_location(_module_name_for(path), path)
        if spec is None or spec.loader is None:
            raise CollectorError(f"无法从 {path} 构造模块（不是合法的 Python 文件？）")
        module = importlib.util.module_from_spec(spec)
        # 先注册再执行，保证插件内部的自引用能正常工作
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - 插件代码可能抛出任意异常
            raise CollectorError(f"执行插件 {path} 时出错: {exc}") from exc
        return module

    try:
        return importlib.import_module(module_ref)
    except ImportError as exc:
        raise CollectorError(
            f"无法导入插件模块 {module_ref!r}: {exc}。"
            "请确认它已安装且在 sys.path 上，或改用 .py 文件路径。"
        ) from exc


def _extract_backend(module: ModuleType) -> object:
    """从插件模块中取出后端对象。

    Raises:
        CollectorError: 三个候选属性都不存在或都无法调用。
    """
    for attr in _ATTR_CANDIDATES:
        candidate = getattr(module, attr, None)
        if candidate is None:
            continue
        if callable(candidate) and attr == "create_backend":
            try:
                return candidate()
            except Exception as exc:  # noqa: BLE001 - 插件代码可能抛出任意异常
                raise CollectorError(f"插件的 create_backend() 执行失败: {exc}") from exc
        return candidate

    raise CollectorError(
        f"插件模块 {module.__name__!r} 未提供后端对象。"
        f"请在其中定义 {' / '.join(_ATTR_CANDIDATES)} 之一。"
    )


def _validate_backend(backend: object, module_ref: str) -> CollectorBackend:
    """校验后端对象是否满足协议。

    Raises:
        CollectorError: 缺少必要属性。
    """
    missing = [name for name in ("collect", "available") if not hasattr(backend, name)]
    if missing:
        raise CollectorError(
            f"插件 {module_ref!r} 返回的对象缺少必要方法: {', '.join(missing)}。"
            "请参考 docs/collector-plugin.md 的模板实现 CollectorBackend 协议。"
        )
    if not hasattr(backend, "name"):
        # name 不是硬性要求，缺失时补一个可读的默认值
        backend.name = type(backend).__name__  # type: ignore[attr-defined]
    return backend  # type: ignore[return-value]


def load_plugin_backend(module_ref: str) -> CollectorBackend:
    """加载并校验一个插件后端。

    Args:
        module_ref: 模块路径或 ``.py`` 文件路径。

    Returns:
        满足 :class:`~xhs_pain_miner.collectors.base.CollectorBackend` 协议的后端。

    Raises:
        CollectorError: 加载失败或对象不符合协议。
    """
    if not module_ref or not module_ref.strip():
        raise CollectorError(
            "未指定采集插件。请设置环境变量 XHS_COLLECTOR_PLUGIN 指向你的采集器模块，"
            "例如：export XHS_COLLECTOR_PLUGIN=/path/to/my_xhs_backend.py"
        )
    module = _load_module(module_ref.strip())
    return _ValidatedBackend(_validate_backend(_extract_backend(module), module_ref))
