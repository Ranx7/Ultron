"""Table-driven tests for the deterministic memory eligibility boundary."""

from __future__ import annotations

import pytest

from memory import manager


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("/reset the session", "command_or_request"),
        ("Hello!", "greeting"),
        ("Tell me a joke", "joke_request"),
        ("12 * (3 + 4)", "calculation"),
        ("How does semantic search work?", "ordinary_question"),
        ("I am hungry right now.", "explicitly_temporary"),
        ("I prefer Python for automation.", None),
        ("I use Fedora on my laptop.", None),
        ("My project uses PostgreSQL.", None),
        ("I am building Atlas.", None),
    ],
)
def test_current_message_classifier_is_small_and_deterministic(
    message: str,
    reason: str | None,
) -> None:
    assert manager._message_ineligibility_reason(message) == reason


def _candidate(content: str, scope: str = "general") -> manager.MemoryCandidate:
    return manager.MemoryCandidate(content, "technical", 3, 0.9, scope)


@pytest.mark.parametrize(
    ("message", "candidates", "scopes", "reasons"),
    [
        (
            "Python is an interpreted language.",
            [_candidate("Python is an interpreted language.")],
            [],
            ["general_candidate_without_current_message_ownership"],
        ),
        (
            "I prefer Python for automation.",
            [_candidate("The user prefers Python for automation.")],
            ["user"],
            ["current_message_ownership"],
        ),
        (
            "I use Fedora on my laptop.",
            [_candidate("The user uses Fedora on a laptop.")],
            ["user"],
            ["current_message_ownership"],
        ),
        (
            "My main laptop runs Windows.",
            [_candidate("The user's main laptop runs Windows.")],
            ["user"],
            ["current_message_ownership"],
        ),
        (
            "My project uses PostgreSQL.",
            [_candidate("The project uses PostgreSQL.")],
            ["project"],
            ["current_message_ownership"],
        ),
        (
            "I am building Atlas.",
            [_candidate("Atlas is being built by the user.")],
            ["project"],
            ["current_message_ownership"],
        ),
        (
            "I prefer Python. I am building Atlas.",
            [
                _candidate("The user prefers Python."),
                _candidate("Atlas is being built by the user."),
            ],
            ["user", "project"],
            ["current_message_ownership", "current_message_ownership"],
        ),
    ],
)
def test_candidate_validation_only_corrects_current_message_owned_content(
    message: str,
    candidates: list[manager.MemoryCandidate],
    scopes: list[str],
    reasons: list[str],
) -> None:
    accepted, diagnostics = manager._validate_candidates(candidates, message)

    assert [candidate.scope for candidate in accepted] == scopes
    actual_reasons = [diagnostic["reason"] for diagnostic in diagnostics]
    assert all(
        expected in actual
        for expected, actual in zip(reasons, actual_reasons)
    )
    assert all(
        candidate.validation_reason is not None
        for candidate in accepted
    )


def test_context_cannot_make_an_ineligible_current_message_reach_analyzer() -> None:
    class Analyzer:
        def analyze(self, message, recent_context=None):
            raise AssertionError("ineligible messages must not reach the analyzer")

        def resolve(self, candidate, existing_memory):
            raise AssertionError("not reached")

    assert manager.process_message(
        "How does semantic search work?",
        recent_context=[{"role": "user", "content": "I prefer Python."}],
        analyzer=Analyzer(),
    ) == []
