from __future__ import annotations

import pytest

from agentic_rag.config import Settings, load_settings
from agentic_rag.errors import ConfigurationError, UnknownComponent
from agentic_rag.registry import Registries, load_plugins

KEY = "k" * 20


def test_unknown_component_lists_valid_names() -> None:
    regs = Registries()
    regs.chunker.register("a", lambda: 1)
    regs.chunker.register("b", lambda: 2)
    with pytest.raises(UnknownComponent) as err:
        regs.chunker.create("nope")
    assert err.value.details == {"kind": "chunker", "name": "nope", "valid": ["a", "b"]}


def test_duplicate_registration_needs_replace() -> None:
    regs = Registries()
    regs.chunker.register("a", lambda: 1)
    with pytest.raises(ConfigurationError):
        regs.chunker.register("a", lambda: 2)
    regs.chunker.register("a", lambda: 2, replace=True)
    assert regs.chunker.create("a") == 2


def test_unknown_extension_point_is_typed() -> None:
    with pytest.raises(UnknownComponent):
        Registries()["vector_stoer"]


def test_plugin_module_can_register_and_a_bad_one_fails_typed() -> None:
    regs = Registries()
    load_plugins(regs, ["tests.plugin_fixture"])
    assert regs.chunker.create("tiny").version == "tiny-1"
    with pytest.raises(ConfigurationError, match="could not be imported"):
        load_plugins(regs, ["no.such.module"])
    with pytest.raises(ConfigurationError, match="register"):
        load_plugins(regs, ["json"])


def test_settings_validate_at_start_and_never_echo_values() -> None:
    ok = load_settings({"AGENTIC_RAG_API_KEYS": f"{KEY},{KEY}x", "AGENTIC_RAG_PARSERS": "text,pdf"})
    assert [k.get_secret_value() for k in ok.api_keys] == [KEY, KEY + "x"]
    assert ok.parsers == ["text", "pdf"]
    assert "k" * 20 not in repr(ok)
    for env in (
        {"AGENTIC_RAG_NOPE": "1"},
        {"AGENTIC_RAG_MAX_TOP_K": "999"},
        {"AGENTIC_RAG_API_KEYS": "short"},
        {"AGENTIC_RAG_API_KEYS": ",".join([KEY] * 3)},
        {"AGENTIC_RAG_OPENAI_API_KEY": "sk-very-secret", "AGENTIC_RAG_HNSW_M": "abc"},
        {"AGENTIC_RAG_CHUNK_MAX_CHARS": "200", "AGENTIC_RAG_CHUNK_OVERLAP_CHARS": "150"},
    ):
        with pytest.raises(ConfigurationError) as err:
            load_settings(env)
        assert "sk-very-secret" not in str(err.value)


def test_settings_ignore_the_ambient_environment_when_given_a_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_RAG_MAX_TOP_K", "7")
    assert load_settings({}).max_top_k == 50
    assert load_settings().max_top_k == 7


def test_client_deadline_default_exceeds_server_deadline_is_documented_by_one_number() -> None:
    assert Settings().request_deadline_seconds == 55.0
