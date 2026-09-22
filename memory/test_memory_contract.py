"""Unit tests for the memory extraction JSON scope contract."""

from __future__ import annotations

import json

import pytest

from memory import manager


def _model_json(scope: str | None) -> str:
    memory = {
        "content": "A durable candidate.",
        "memory_type": "fact",
        "importance": 3,
        "confidence": 0.95,
    }
    if scope is not None:
        memory["scope"] = scope
    return json.dumps({"memories": [memory]})


@pytest.mark.parametrize(
    ("scope", "expected_scope"),
    [
        (None, "general"),
        ("not-a-scope", "general"),
        ("general", "general"),
        ("user", "user"),
        ("project", "project"),
    ],
)
def test_model_json_scope_is_normalized(scope: str | None, expected_scope: str) -> None:
    candidates = manager._parse_candidates(_model_json(scope))

    assert len(candidates) == 1
    assert candidates[0].scope == expected_scope


class _JsonAnalyzer:
    def __init__(self, raw: str):
        self.raw = raw

    def analyze(self, message: str, recent_context=None):
        return manager._parse_candidates(self.raw)

    def resolve(self, candidate, existing_memory):
        raise AssertionError("Durable candidates should have no related memory in this test.")


@pytest.mark.parametrize(
    ("scope", "should_store"),
    [
        (None, False),
        ("invalid", False),
        ("general", False),
        ("user", True),
        ("project", True),
    ],
)
def test_only_valid_durable_scopes_reach_storage(
    monkeypatch: pytest.MonkeyPatch,
    scope: str | None,
    should_store: bool,
) -> None:
    stored_scopes: list[str] = []

    monkeypatch.setattr(manager, "_find_related_memory", lambda content: None)
    monkeypatch.setattr(
        manager,
        "_store_new_memory",
        lambda candidate, source_message_id: stored_scopes.append(candidate.scope) or (1, True),
    )

    result = manager.process_message(
        "A user message.",
        analyzer=_JsonAnalyzer(_model_json(scope)),
    )

    assert bool(result) is should_store
    assert stored_scopes == ([scope] if scope in {"user", "project"} else [])


def test_runtime_uses_the_modelfile_extraction_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    analyzer = manager.OllamaMemoryAnalyzer()
    captured: dict[str, str] = {}

    def fake_chat(system_prompt: str, user_prompt: str) -> str:
        captured["system_prompt"] = system_prompt
        return '{"memories": []}'

    monkeypatch.setattr(analyzer, "_chat", fake_chat)

    assert analyzer.analyze("No durable details.") == []
    assert captured["system_prompt"] == manager._memory_extraction_prompt()
    assert '"scope"' in captured["system_prompt"]
