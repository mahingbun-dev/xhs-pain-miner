"""采集层测试。

重点覆盖三件事：

1. 内置样例后端能按限额正确裁剪数据。
2. 样例语料确实包含预期的痛点簇（否则 M1 的聚类验证没有意义）。
3. 插件加载器的成功路径与各类失败路径都有可读的错误信息。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xhs_pain_miner.collectors.base import CollectorError
from xhs_pain_miner.collectors.factory import build_collector
from xhs_pain_miner.collectors.fixture import FixtureBackend, load_fixture_data, parse_corpus
from xhs_pain_miner.collectors.plugin import _ValidatedBackend, load_plugin_backend
from xhs_pain_miner.config import Settings
from xhs_pain_miner.models import RawCorpus, RawNote

EXPECTED_CLUSTERS = (
    "假白",
    "搓泥",
    "闭口",
    "防水",
    "卸",
    "油腻",
    "刺痛",
    "贵",
)


def _settings(**overrides: object) -> Settings:
    """构造不读取 .env 的配置对象。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestFixtureData:
    """内置样例语料。"""

    def test_loads(self):
        data = load_fixture_data()
        assert data["notes"], "样例语料不能为空"
        assert data["comments"], "样例语料必须包含评论"

    def test_uses_unroutable_domain(self):
        """样例里的 URL 必须落在 .invalid 顶级域，确保永远不会被真实请求。"""
        data = load_fixture_data()
        for note in data["notes"]:
            assert "example.invalid" in note["url"]
            for image in note.get("images", []):
                assert "example.invalid" in image

    def test_covers_expected_pain_clusters(self):
        """语料必须覆盖足够多的痛点主题，否则聚类效果验证不出问题。"""
        data = load_fixture_data()
        text = " ".join(c["content"] for c in data["comments"])
        missing = [kw for kw in EXPECTED_CLUSTERS if kw not in text]
        assert not missing, f"样例语料缺少这些痛点主题: {missing}"

    def test_has_second_level_comments(self):
        """二级评论是「评论区的评论区」这类深层痛点的主要来源，必须有。"""
        data = load_fixture_data()
        assert any(c.get("parent_id") for c in data["comments"])

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(CollectorError, match="不存在"):
            load_fixture_data(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(CollectorError, match="JSON"):
            load_fixture_data(bad)


class TestParseCorpus:
    """JSON → 领域模型。"""

    def test_parses_datetimes(self):
        corpus = parse_corpus(load_fixture_data())
        note = corpus.notes[0]
        assert note.publish_time is not None
        assert note.publish_time.year >= 2020

    def test_keyword_override(self):
        corpus = parse_corpus(load_fixture_data(), keyword="降噪耳机")
        assert corpus.keyword == "降噪耳机"


class TestFixtureBackend:
    """样例后端。"""

    def test_available(self):
        assert FixtureBackend().available()

    def test_name_discloses_it_is_not_real_data(self):
        """后端名必须如实说明这是样例数据，避免用户误以为采到了真实语料。"""
        assert "样例" in FixtureBackend().name

    def test_respects_note_limit(self):
        corpus = FixtureBackend().collect("防晒霜", limit=3)
        assert len(corpus.notes) == 3

    def test_filters_comments_to_kept_notes(self):
        """被 limit 截掉的笔记，其评论也必须一并剔除。"""
        corpus = FixtureBackend().collect("防晒霜", limit=2)
        kept = {n.note_id for n in corpus.notes}
        assert all(c.note_id in kept for c in corpus.comments)

    def test_respects_comment_limit(self):
        corpus = FixtureBackend().collect("防晒霜", limit=12, max_comments_per_note=2)
        per_note: dict[str, int] = {}
        for comment in corpus.comments:
            per_note[comment.note_id] = per_note.get(comment.note_id, 0) + 1
        assert per_note, "应当有评论"
        assert max(per_note.values()) <= 2

    def test_zero_limit_returns_nothing(self):
        corpus = FixtureBackend().collect("防晒霜", limit=0)
        assert corpus.notes == []
        assert corpus.comments == []

    def test_full_collect_returns_all(self):
        corpus = FixtureBackend().collect("防晒霜", limit=999, max_comments_per_note=999)
        assert len(corpus.notes) == 12
        assert len(corpus.comments) == 65

    def test_hashes_not_raw_ids(self):
        """★ 合规：语料里不得出现未哈希的用户标识。"""
        corpus = FixtureBackend().collect("防晒霜", limit=12, max_comments_per_note=999)
        for comment in corpus.comments:
            assert len(comment.user_hash) == 16
        for note in corpus.notes:
            assert len(note.author_hash) == 16


class TestBuildCollector:
    """后端工厂。"""

    def test_builds_fixture(self):
        collector = build_collector(_settings(collector_backend="fixture"))
        assert isinstance(collector, FixtureBackend)

    def test_mcp_reports_pending_milestone(self):
        """MCP 后端尚未实现，报错信息必须指明里程碑与替代方案。"""
        with pytest.raises(CollectorError, match="M3"):
            build_collector(_settings(collector_backend="mcp"))

    def test_plugin_without_reference_gives_actionable_error(self):
        with pytest.raises(CollectorError, match="XHS_COLLECTOR_PLUGIN"):
            build_collector(_settings(collector_backend="plugin", collector_plugin=None))


_VALID_PLUGIN = '''
"""测试用插件。"""
from xhs_pain_miner.models import RawCorpus


class FakeBackend:
    name = "fake-plugin"

    def collect(self, keyword, *, limit, max_comments_per_note=20):
        return RawCorpus(keyword=keyword, backend=self.name)

    def available(self):
        return True


BACKEND = FakeBackend()
'''

_FACTORY_PLUGIN = '''
"""使用 create_backend 工厂的插件。"""
from xhs_pain_miner.models import RawCorpus


class Backend:
    name = "factory-plugin"

    def collect(self, keyword, *, limit, max_comments_per_note=20):
        return RawCorpus(keyword=keyword, backend=self.name)

    def available(self):
        return True


def create_backend():
    return Backend()
'''

_NO_BACKEND_PLUGIN = '''
"""缺少后端对象的插件。"""

VALUE = 42
'''

_INCOMPLETE_PLUGIN = '''
"""后端对象缺少必要方法。"""


class Incomplete:
    name = "incomplete"


BACKEND = Incomplete()
'''


def _write_plugin(tmp_path: Path, source: str, name: str = "plugin_mod.py") -> Path:
    """把插件源码写到临时文件。"""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


class TestPluginLoader:
    """插件加载器。"""

    def test_loads_backend_attribute(self, tmp_path: Path):
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _VALID_PLUGIN)))
        assert backend.name == "fake-plugin"
        assert backend.available()

    def test_loads_create_backend_factory(self, tmp_path: Path):
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _FACTORY_PLUGIN)))
        assert backend.name == "factory-plugin"

    def test_collect_works_through_protocol(self, tmp_path: Path):
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _VALID_PLUGIN)))
        corpus = backend.collect("防晒霜", limit=10)
        assert corpus.keyword == "防晒霜"
        assert corpus.backend == "fake-plugin"

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(CollectorError, match="不存在"):
            load_plugin_backend(str(tmp_path / "missing.py"))

    def test_module_without_backend(self, tmp_path: Path):
        path = _write_plugin(tmp_path, _NO_BACKEND_PLUGIN)
        with pytest.raises(CollectorError, match="BACKEND"):
            load_plugin_backend(str(path))

    def test_backend_missing_methods(self, tmp_path: Path):
        path = _write_plugin(tmp_path, _INCOMPLETE_PLUGIN)
        with pytest.raises(CollectorError, match="collect"):
            load_plugin_backend(str(path))

    def test_empty_reference(self):
        with pytest.raises(CollectorError, match="未指定采集插件"):
            load_plugin_backend("   ")

    def test_failing_module_import(self):
        with pytest.raises(CollectorError, match="无法导入"):
            load_plugin_backend("definitely_not_a_real_module_xyz")


_BAD_RETURN_PLUGIN = '''
"""collect() 返回类型不符合协议的插件。"""


class BadBackend:
    name = "bad-return"

    def collect(self, keyword, *, limit, max_comments_per_note=20):
        return {"not": "a RawCorpus"}

    def available(self):
        return True


BACKEND = BadBackend()
'''

_BROKEN_AVAILABLE_PLUGIN = '''
"""available() 会抛异常的插件。"""

from xhs_pain_miner.models import RawCorpus


class FlakyBackend:
    name = "flaky"

    def collect(self, keyword, *, limit, max_comments_per_note=20):
        return RawCorpus(keyword=keyword, backend=self.name)

    def available(self):
        raise RuntimeError("登录态已过期")


BACKEND = FlakyBackend()
'''


class TestPluginReturnValidation:
    """插件返回值与异常必须被兜住 —— 插件作者不应该能让主程序崩掉。"""

    def test_non_corpus_return_raises_readable_error(self, tmp_path: Path):
        """返回 dict 的插件必须被转成可读的 CollectorError，而不是下游的 AttributeError。"""
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _BAD_RETURN_PLUGIN)))
        with pytest.raises(CollectorError, match="RawCorpus"):
            backend.collect("防晒霜", limit=10)

    def test_error_message_names_the_plugin(self, tmp_path: Path):
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _BAD_RETURN_PLUGIN)))
        with pytest.raises(CollectorError, match="bad-return"):
            backend.collect("防晒霜", limit=10)

    def test_broken_available_reports_unavailable(self, tmp_path: Path):
        """available() 抛异常时应报告不可用，而不是把异常抛给用户。"""
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _BROKEN_AVAILABLE_PLUGIN)))
        assert backend.available() is False

    def test_valid_plugin_still_works(self, tmp_path: Path):
        backend = load_plugin_backend(str(_write_plugin(tmp_path, _VALID_PLUGIN)))
        assert backend.collect("防晒霜", limit=1).backend == "fake-plugin"


class TestPluginModuleNaming:
    """同进程加载多个插件时，模块名必须唯一，否则会互相覆盖。"""

    def test_two_plugins_do_not_overwrite_each_other(self, tmp_path: Path):
        first = _write_plugin(tmp_path, _VALID_PLUGIN, "first.py")
        second = _write_plugin(tmp_path, _FACTORY_PLUGIN, "second.py")

        backend_a = load_plugin_backend(str(first))
        backend_b = load_plugin_backend(str(second))

        assert backend_a.name == "fake-plugin"
        assert backend_b.name == "factory-plugin"

    def test_same_file_loaded_twice_is_stable(self, tmp_path: Path):
        path = _write_plugin(tmp_path, _VALID_PLUGIN)
        assert load_plugin_backend(str(path)).name == "fake-plugin"
        assert load_plugin_backend(str(path)).name == "fake-plugin"


class _StubBackend:
    """符合协议但可注入任意返回值的桩后端。"""

    name = "stub"

    def __init__(self, result: object) -> None:
        self._result = result

    def collect(self, keyword: str, *, limit: int, max_comments_per_note: int = 20) -> object:
        return self._result

    def available(self) -> bool:
        return True


class TestDeepReturnValidation:
    """``RawCorpus`` 内部元素类型也必须校验。

    只查最外层 ``isinstance(result, RawCorpus)`` 是不够的：插件很容易构造出
    「外壳正确、内脏错误」的对象（如 ``notes=[{"title": "x"}]``），
    它会在下游的 ``note.images`` 处抛出用户完全无法定位的裸 traceback。
    """

    def test_notes_containing_dict_is_rejected(self):
        backend = _ValidatedBackend(_StubBackend(RawCorpus(keyword="x", notes=[{"a": 1}])))
        with pytest.raises(CollectorError, match="notes"):
            backend.collect("防晒霜", limit=1)

    def test_comments_as_string_is_rejected(self):
        backend = _ValidatedBackend(_StubBackend(RawCorpus(keyword="x", comments="oops")))
        with pytest.raises(CollectorError, match="comments"):
            backend.collect("防晒霜", limit=1)

    def test_non_list_field_is_rejected(self):
        backend = _ValidatedBackend(_StubBackend(RawCorpus(keyword="x", notes=42)))
        with pytest.raises(CollectorError, match="必须是 list"):
            backend.collect("防晒霜", limit=1)

    def test_generator_is_rejected(self):
        """生成器能通过 iter()，但会让下游 len() 崩，且校验过程本身会把它耗尽。"""
        generator = (RawNote(note_id="n1") for _ in range(1))
        backend = _ValidatedBackend(_StubBackend(RawCorpus(keyword="x", notes=generator)))
        with pytest.raises(CollectorError, match="必须是 list"):
            backend.collect("防晒霜", limit=1)

    def test_error_names_the_field_index(self):
        bad = RawCorpus(keyword="x", notes=[], comments=[{"c": 1}])
        backend = _ValidatedBackend(_StubBackend(bad))
        with pytest.raises(CollectorError, match=r"comments\[0\]"):
            backend.collect("防晒霜", limit=1)

    def test_bad_element_after_good_one_is_caught(self):
        """★ 混合用例：校验必须遍历全部元素，而不是只查第 0 个。"""
        bad = RawCorpus(keyword="x", notes=[RawNote(note_id="n1"), {"b": 1}])
        backend = _ValidatedBackend(_StubBackend(bad))
        with pytest.raises(CollectorError, match=r"notes\[1\]"):
            backend.collect("防晒霜", limit=1)

    def test_valid_corpus_passes_through(self):
        good = RawCorpus(keyword="x", backend="stub")
        backend = _ValidatedBackend(_StubBackend(good))
        assert backend.collect("防晒霜", limit=1) is good


class TestAvailableErrorReason:
    """``available()`` 抛异常时，原因必须被保留下来给 doctor 展示。

    否则「cookie 已失效，请重新登录」这类最有可操作性的信息会被完全吞掉，
    用户只看到一句无用的「当前不可用」。
    """

    def test_reason_is_recorded(self):
        class Boom:
            name = "boom"

            def collect(self, *args: object, **kwargs: object) -> None:
                return None

            def available(self) -> bool:
                raise RuntimeError("cookie 已失效，请重新登录后再试")

        backend = _ValidatedBackend(Boom())  # type: ignore[arg-type]
        assert backend.available() is False
        assert "cookie 已失效" in (backend.last_error or "")

    def test_reason_is_cleared_after_recovery(self):
        """★ 同一个实例先失败再成功，last_error 必须被清空。

        原版用的是**全新**的 backend（``__init__`` 里 ``last_error`` 本来就是
        ``None``），删掉「成功时清空」那行代码它照样通过 —— 是条恒真的空壳测试。
        """
        state = {"fail": True}

        class Flaky:
            name = "flaky"

            def collect(self, *args: object, **kwargs: object) -> None:
                return None

            def available(self) -> bool:
                if state["fail"]:
                    raise RuntimeError("临时故障")
                return True

        backend = _ValidatedBackend(Flaky())  # type: ignore[arg-type]

        assert backend.available() is False
        assert backend.last_error is not None

        state["fail"] = False
        assert backend.available() is True
        assert backend.last_error is None


class TestPluginExceptionHandling:
    """插件抛出的异常必须被转成可读的 ``CollectorError``。

    这是 ``plugin.py`` 自己文档里承诺的目标（"插件作者不应该能让主程序崩掉"），
    但 ``_ValidatedBackend.collect`` 一度只兜返回值、没包 ``try``。
    """

    def test_runtime_error_is_wrapped(self):
        class Boom:
            name = "boom"

            def collect(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("内部采集器炸了：连接被重置")

            def available(self) -> bool:
                return True

        backend = _ValidatedBackend(Boom())  # type: ignore[arg-type]
        with pytest.raises(CollectorError, match="连接被重置"):
            backend.collect("防晒霜", limit=1)

    def test_error_names_the_plugin(self):
        class Boom:
            name = "boom-plugin"

            def collect(self, *args: object, **kwargs: object) -> None:
                raise ValueError("bad")

            def available(self) -> bool:
                return True

        backend = _ValidatedBackend(Boom())  # type: ignore[arg-type]
        with pytest.raises(CollectorError, match="boom-plugin"):
            backend.collect("防晒霜", limit=1)

    def test_keyboard_interrupt_is_not_wrapped(self):
        """Ctrl-C 与进程退出必须原样透传，不能被包装成「采集失败」。"""

        class Interrupting:
            name = "interrupting"

            def collect(self, *args: object, **kwargs: object) -> None:
                raise KeyboardInterrupt

            def available(self) -> bool:
                return True

        backend = _ValidatedBackend(Interrupting())  # type: ignore[arg-type]
        with pytest.raises(KeyboardInterrupt):
            backend.collect("防晒霜", limit=1)
