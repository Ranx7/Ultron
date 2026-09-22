"""Endpoint tests for the memory inspection API."""

import pytest


pytest.importorskip("flask")
pytest.importorskip("ollama")

import Ultron_Interface as interface


def test_memory_returns_database_counts(monkeypatch):
    """The status endpoint exposes the counts supplied by the database API."""

    monkeypatch.setattr(interface, "get_counts", lambda: (9, 4))

    response = interface.app.test_client().get("/memory")

    assert response.status_code == 200
    assert response.get_json() == {
        "stored_messages": 9,
        "long_term_memories": 4,
    }


def test_memory_search_uses_hybrid_retrieval(monkeypatch):
    """Searching the inspection endpoint uses the public retrieval API only."""

    expected_memories = [{"id": 12, "content": "Uses a standing desk."}]
    calls = []

    def fake_retrieve_memories(query, *, limit):
        calls.append((query, limit))
        return expected_memories

    monkeypatch.setattr(
        interface,
        "retrieve_memories",
        fake_retrieve_memories,
    )

    response = interface.app.test_client().get("/memory/search?q=desk")

    assert response.status_code == 200
    assert response.get_json() == {"memories": expected_memories}
    assert calls == [("desk", 20)]


def test_memory_search_requires_query(monkeypatch):
    """A missing query is rejected before retrieval is attempted."""

    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("retrieval should not run without a query")

    monkeypatch.setattr(
        interface,
        "retrieve_memories",
        unexpected_retrieval,
    )

    response = interface.app.test_client().get("/memory/search")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing search query."}
