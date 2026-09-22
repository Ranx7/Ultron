"""
Ultron memory manager.

Responsibilities:

    CURRENT USER MESSAGE
            |
            v
    memory candidate analysis
            |
            v
    scope eligibility gate
            |
            v
    semantic duplicate search
            |
            v
    conflict / duplicate resolution
            |
            v
    add / replace / merge / ignore

Important architectural rules:

    - Only USER messages enter this pipeline.
    - The current message is the primary source of truth.
    - Recent context may be supplied for resolving references.
    - Old conversation history is NOT automatically searched here.
    - Assistant messages are never turned into memories.
    - The LLM proposes memory candidates, but Python enforces
      the final memory scope eligibility gate.
    - General knowledge is never allowed into long-term memory.
    - Memories are stored as durable facts, preferences, goals,
      project information, decisions, corrections, etc.
    - Semantic similarity is used for deduplication.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from .database import (
    add_or_update_memory,
    archive_memory,
)
from .retrieval import semantic_search
from .embeddings import embed_memory


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = os.getenv(
    "ULTRON_OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "ULTRON_MEMORY_MODEL",
    "Ultron-Memory:latest",
)

# Similarity above this value means an existing memory is
# probably talking about the same thing.
RELATED_MEMORY_THRESHOLD = 0.75

# Extremely strong semantic match:
# treat it as a duplicate without asking the small resolver model.
AUTO_DUPLICATE_THRESHOLD = 0.85

# Never allow one user message to create an unreasonable
# number of durable memories.
MAX_MEMORIES_PER_MESSAGE = 3

# Maximum candidate length stored in the database.
MAX_MEMORY_LENGTH = 1200

# Allowed durable memory categories.
ALLOWED_MEMORY_TYPES = {
    "fact",
    "preference",
    "goal",
    "project",
    "technical",
    "decision",
    "correction",
}

# Candidate scope controls whether information is allowed
# to enter long-term memory.
ALLOWED_MEMORY_SCOPES = {
    "user",
    "project",
    "general",
}

# Only these scopes are allowed through the Python memory gate.
STORABLE_MEMORY_SCOPES = {
    "user",
    "project",
}


def _memory_extraction_prompt() -> str:
    """Return the extraction contract embedded in the Modelfile."""

    modelfile = Path(__file__).with_name("Modelfile")
    text = modelfile.read_text(encoding="utf-8")
    prefix = 'SYSTEM """'

    try:
        return text.split(prefix, 1)[1].rsplit('"""', 1)[0].strip()
    except IndexError as exc:
        raise RuntimeError(
            "memory/Modelfile must define a SYSTEM prompt."
        ) from exc


# ============================================================
# Data structures
# ============================================================

@dataclass
class MemoryCandidate:
    """
    A proposed durable memory extracted from the current
    user message.

    scope:
        user    -> durable information about the user
        project -> durable information about a user project
        general -> general knowledge / explanation / answer
    """

    content: str
    memory_type: str
    importance: float
    confidence: float
    scope: str


@dataclass
class MemoryAction:
    """
    Final action after comparing a candidate against existing
    memories.
    """

    action: str
    content: str
    memory_type: str
    importance: float


# ============================================================
# Analyzer interface
# ============================================================

class MemoryAnalyzer(Protocol):
    """
    Interface used by the memory manager.

    The manager does not care which model performs the analysis.

    This keeps the memory architecture separate from Ollama.
    """

    def analyze(
        self,
        message: str,
        recent_context: list[dict] | None = None,
    ) -> list[MemoryCandidate]:
        ...

    def resolve(
        self,
        candidate: MemoryCandidate,
        existing_memory: dict,
    ) -> MemoryAction:
        ...


# ============================================================
# Ollama analyzer
# ============================================================

class OllamaMemoryAnalyzer:
    """
    Uses an Ollama model to determine whether the current
    message contains durable information.

    The model is explicitly instructed to return structured JSON.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "options": {
                # Keep generation deterministic at runtime too,
                # rather than relying only on the Modelfile.
                "temperature": 0,
                "top_p": 0.9,
                "seed": 42,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Ollama returned an unexpected response."
            ) from exc

    # --------------------------------------------------------
    # First-stage memory extraction
    # --------------------------------------------------------

    def analyze(
        self,
        message: str,
        recent_context: list[dict] | None = None,
    ) -> list[MemoryCandidate]:

        context_text = _format_recent_context(
            recent_context
        )

        system_prompt = _memory_extraction_prompt()

        user_prompt = (
            "CURRENT USER MESSAGE:\n"
            f"{message}\n\n"
        )

        if context_text:
            user_prompt += (
                "RECENT CONTEXT ONLY:\n"
                f"{context_text}\n"
            )

        raw = self._chat(
            system_prompt,
            user_prompt,
        )

        return _parse_candidates(raw)

    # --------------------------------------------------------
    # Duplicate/conflict resolution
    # --------------------------------------------------------

    def resolve(
        self,
        candidate: MemoryCandidate,
        existing_memory: dict,
    ) -> MemoryAction:

        system_prompt = """
You are Ultron's memory-consolidation subsystem.

Compare ONE NEW USER/PROJECT MEMORY against ONE EXISTING
MEMORY.

Determine whether the new information should be:

"duplicate"
    The existing memory already contains essentially the
    same information.

"replace"
    The new information supersedes or corrects the old memory.

"merge"
    Both contain useful compatible information and should be
    combined into one concise durable memory.

"new"
    They are related enough to inspect but are actually
    different facts and both should remain.

Important:

- The NEW memory represents the current user message.
- The current information has priority when it explicitly
  corrects the old information.
- Do not preserve outdated information when the user clearly
  says it has changed.
- Do not invent information.
- Return only the final durable memory statement.

Return ONLY valid JSON:

{
  "action": "duplicate",
  "content": "final memory text",
  "memory_type": "fact",
  "importance": 0.0
}

The content field must contain only the final durable memory
statement.

Do not mention this comparison process.
Do not explain your reasoning.
"""

        user_prompt = (
            "NEW MEMORY CANDIDATE:\n"
            f"{candidate.content}\n\n"
            "NEW TYPE:\n"
            f"{candidate.memory_type}\n\n"
            "NEW SCOPE:\n"
            f"{candidate.scope}\n\n"
            "EXISTING MEMORY:\n"
            f"{existing_memory.get('content', '')}\n\n"
            "EXISTING TYPE:\n"
            f"{existing_memory.get('memory_type', 'fact')}"
        )

        raw = self._chat(
            system_prompt,
            user_prompt,
        )

        return _parse_action(
            raw,
            fallback=candidate,
        )


# ============================================================
# Parsing
# ============================================================

def _strip_json_fences(text: str) -> str:
    """
    Remove markdown JSON fences if a model adds them despite
    being instructed not to.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def _safe_json(text: str) -> dict:
    """
    Parse model JSON safely.
    """

    cleaned = _strip_json_fences(text)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Memory analyzer returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            "Memory analyzer JSON must be an object."
        )

    return value


def _normalize_type(value: Any) -> str:
    """
    Normalize memory category.
    """

    value = str(value or "fact").strip().lower()

    if value not in ALLOWED_MEMORY_TYPES:
        return "fact"

    return value


def _normalize_scope(value: Any) -> str:
    """
    Normalize candidate scope.

    Missing or invalid scopes are treated as "general"
    rather than being allowed through as user/project memory.
    """

    value = str(value or "").strip().lower()

    if value not in ALLOWED_MEMORY_SCOPES:
        return "general"

    return value


def _normalize_importance(value: Any) -> float:
    """
    Clamp importance to 0..5.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 2.5

    return max(
        0.0,
        min(5.0, number),
    )


def _normalize_confidence(value: Any) -> float:
    """
    Clamp confidence to 0..1.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0

    return max(
        0.0,
        min(1.0, number),
    )


def _parse_candidates(
    raw: str,
) -> list[MemoryCandidate]:

    data = _safe_json(raw)

    memories = data.get(
        "memories",
        [],
    )

    if not isinstance(memories, list):
        return []

    results: list[MemoryCandidate] = []

    for item in memories:
        if not isinstance(item, dict):
            continue

        content = str(
            item.get("content", "")
        ).strip()

        if not content:
            continue

        if len(content) > MAX_MEMORY_LENGTH:
            content = content[
                :MAX_MEMORY_LENGTH
            ].rstrip()

        memory_type = _normalize_type(
            item.get("memory_type")
        )

        importance = _normalize_importance(
            item.get("importance")
        )

        confidence = _normalize_confidence(
            item.get("confidence")
        )

        scope = _normalize_scope(
            item.get("scope")
        )

        # Low-confidence candidates are discarded here instead
        # of polluting long-term memory.
        if confidence < 0.60:
            continue

        results.append(
            MemoryCandidate(
                content=content,
                memory_type=memory_type,
                importance=importance,
                confidence=confidence,
                scope=scope,
            )
        )

        if len(results) >= MAX_MEMORIES_PER_MESSAGE:
            break

    return results


def _parse_action(
    raw: str,
    fallback: MemoryCandidate,
) -> MemoryAction:

    try:
        data = _safe_json(raw)
    except ValueError:
        return MemoryAction(
            action="new",
            content=fallback.content,
            memory_type=fallback.memory_type,
            importance=fallback.importance,
        )

    action = str(
        data.get("action", "new")
    ).strip().lower()

    if action not in {
        "duplicate",
        "replace",
        "merge",
        "new",
    }:
        action = "new"

    content = str(
        data.get(
            "content",
            fallback.content,
        )
    ).strip()

    if not content:
        content = fallback.content

    content = content[
        :MAX_MEMORY_LENGTH
    ].rstrip()

    memory_type = _normalize_type(
        data.get(
            "memory_type",
            fallback.memory_type,
        )
    )

    importance = _normalize_importance(
        data.get(
            "importance",
            fallback.importance,
        )
    )

    return MemoryAction(
        action=action,
        content=content,
        memory_type=memory_type,
        importance=importance,
    )


# ============================================================
# Recent-context handling
# ============================================================

def _format_recent_context(
    recent_context: list[dict] | None,
) -> str:

    if not recent_context:
        return ""

    lines: list[str] = []

    # Only keep a small local window.
    for item in recent_context[-8:]:

        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role", "")
        ).strip().lower()

        content = str(
            item.get("content", "")
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        lines.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(lines)


# ============================================================
# Duplicate detection
# ============================================================

def _find_related_memory(
    content: str,
) -> dict | None:
    """
    Find the strongest existing semantic match.

    This uses semantic search directly rather than the public
    retrieve_memories() function so merely comparing a candidate
    does NOT increment the memory's access counter.
    """

    results = semantic_search(
        content,
        limit=3,
    )

    if not results:
        return None

    best = results[0]

    similarity = float(
        best.get(
            "semantic_similarity",
            0.0,
        )
    )

    if similarity < RELATED_MEMORY_THRESHOLD:
        return None

    from .database import get_memory

    memory = get_memory(
        int(best["memory_id"])
    )

    if memory is None:
        return None

    memory = dict(memory)

    memory["_similarity"] = similarity

    return memory


# ============================================================
# Storage
# ============================================================

def _store_new_memory(
    candidate: MemoryCandidate,
    source_message_id: int | None,
) -> Any:
    """
    Store a new memory and immediately create its semantic
    embedding so it can participate in future retrieval.
    """

    result = add_or_update_memory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        importance=candidate.importance,
        source_message_id=source_message_id,
    )

    # add_or_update_memory() returns:
    # (memory_id, created)
    if isinstance(result, tuple) and result:
        memory_id = result[0]

        try:
            embed_memory(
                int(memory_id),
                candidate.content,
            )
        except Exception as exc:
            print(
                f"Warning: failed to embed memory "
                f"{memory_id}: {exc}"
            )

    return result


def _replace_memory(
    existing_memory: dict,
    action: MemoryAction,
    source_message_id: int | None,
    scope: str,
) -> Any:
    """
    Replace an old memory while preserving the old record as
    archived history.

    The replacement keeps the scope of the current candidate.
    """

    archive_memory(
        int(existing_memory["id"])
    )

    candidate = MemoryCandidate(
        content=action.content,
        memory_type=action.memory_type,
        importance=action.importance,
        confidence=1.0,
        scope=scope,
    )

    return _store_new_memory(
        candidate,
        source_message_id,
    )


def _merge_memory(
    existing_memory: dict,
    action: MemoryAction,
    source_message_id: int | None,
    scope: str,
) -> Any:
    """
    Merge old + new semantic information into one durable
    memory.

    The old record is archived and the consolidated record is
    stored as the active memory.

    The merged memory keeps the scope of the current candidate.
    """

    archive_memory(
        int(existing_memory["id"])
    )

    candidate = MemoryCandidate(
        content=action.content,
        memory_type=action.memory_type,
        importance=action.importance,
        confidence=1.0,
        scope=scope,
    )

    return _store_new_memory(
        candidate,
        source_message_id,
    )


# ============================================================
# Main pipeline
# ============================================================

def process_message(
    message: str,
    *,
    source_message_id: int | None = None,
    recent_context: list[dict] | None = None,
    analyzer: MemoryAnalyzer | None = None,
) -> list[dict]:
    """
    Process ONE current user message.

    Returns a list describing what the memory system did.

    Example return:

    [
        {
            "action": "stored",
            "memory_id": 12,
            "content": "...",
        }
    ]

    Critical rules:

        This function receives the current user message.

        It does NOT retrieve historical conversation to determine
        what the user meant.

        Candidates marked "general" are rejected before semantic
        retrieval or storage.
    """

    if not isinstance(message, str):
        raise TypeError(
            "message must be a string"
        )

    message = message.strip()

    if not message:
        return []

    if analyzer is None:
        analyzer = OllamaMemoryAnalyzer()

    # --------------------------------------------------------
    # Phase 1:
    # Determine whether the current user message contains
    # durable information.
    # --------------------------------------------------------

    candidates = analyzer.analyze(
        message,
        recent_context,
    )

    if not candidates:
        return []

    # --------------------------------------------------------
    # Phase 1.5:
    # Hard Python scope gate.
    #
    # The LLM can classify something as "general", but the
    # LLM is NOT allowed to override this final decision.
    # --------------------------------------------------------

    candidates = [
        candidate
        for candidate in candidates
        if candidate.scope in STORABLE_MEMORY_SCOPES
    ]

    if not candidates:
        return []

    results: list[dict] = []

    # --------------------------------------------------------
    # Phase 2:
    # Handle each candidate independently.
    # --------------------------------------------------------

    for candidate in candidates:

        existing = _find_related_memory(
            candidate.content
        )

        # No semantically related memory exists.
        if existing is None:

            stored = _store_new_memory(
                candidate,
                source_message_id,
            )

            results.append(
                {
                    "action": "stored",
                    "content": candidate.content,
                    "memory_type": candidate.memory_type,
                    "importance": candidate.importance,
                    "scope": candidate.scope,
                    "result": stored,
                }
            )

            continue

        # ----------------------------------------------------
        # Phase 3:
        # Related memory exists.
        #
        # Ask the analyzer whether this is:
        #
        # duplicate / replacement / merge / separate fact
        # ----------------------------------------------------

        similarity = float(
            existing.get(
                "_similarity",
                0.0,
            )
        )

        # Very strong semantic match:
        # don't let the small LLM override the embedding signal.
        if similarity >= AUTO_DUPLICATE_THRESHOLD:
            action = MemoryAction(
                action="duplicate",
                content=candidate.content,
                memory_type=candidate.memory_type,
                importance=candidate.importance,
            )
        else:
            action = analyzer.resolve(
                candidate,
                existing,
            )

        if action.action == "duplicate":

            results.append(
                {
                    "action": "duplicate",
                    "content": candidate.content,
                    "existing_memory_id": existing["id"],
                    "similarity": existing.get(
                        "_similarity",
                        0.0,
                    ),
                }
            )

            continue

        if action.action == "replace":

            stored = _replace_memory(
                existing,
                action,
                source_message_id,
                candidate.scope,
            )

            results.append(
                {
                    "action": "replaced",
                    "content": action.content,
                    "archived_memory_id": existing["id"],
                    "scope": candidate.scope,
                    "result": stored,
                }
            )

            continue

        if action.action == "merge":

            stored = _merge_memory(
                existing,
                action,
                source_message_id,
                candidate.scope,
            )

            results.append(
                {
                    "action": "merged",
                    "content": action.content,
                    "archived_memory_id": existing["id"],
                    "scope": candidate.scope,
                    "result": stored,
                }
            )

            continue

        # ----------------------------------------------------
        # "new"
        #
        # The memories are related, but they contain distinct
        # durable information.
        # ----------------------------------------------------

        stored = _store_new_memory(
            candidate,
            source_message_id,
        )

        results.append(
            {
                "action": "stored_related",
                "content": candidate.content,
                "related_memory_id": existing["id"],
                "similarity": existing.get(
                    "_similarity",
                    0.0,
                ),
                "scope": candidate.scope,
                "result": stored,
            }
        )

    return results


# ============================================================
# Diagnostics
# ============================================================

def get_manager_info() -> dict:
    """
    Configuration information for debugging.
    """

    return {
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "related_memory_threshold": (
            RELATED_MEMORY_THRESHOLD
        ),
        "auto_duplicate_threshold": (
            AUTO_DUPLICATE_THRESHOLD
        ),
        "max_memories_per_message": (
            MAX_MEMORIES_PER_MESSAGE
        ),
        "max_memory_length": (
            MAX_MEMORY_LENGTH
        ),
        "allowed_memory_types": sorted(
            ALLOWED_MEMORY_TYPES
        ),
        "allowed_memory_scopes": sorted(
            ALLOWED_MEMORY_SCOPES
        ),
        "storable_memory_scopes": sorted(
            STORABLE_MEMORY_SCOPES
        ),
    }


# ============================================================
# Self-test
# ============================================================

class _MockAnalyzer:
    """
    Deterministic analyzer for testing the manager without
    launching Ollama.
    """

    def analyze(
        self,
        message: str,
        recent_context: list[dict] | None = None,
    ) -> list[MemoryCandidate]:

        if message == "STORE_TEST":
            return [
                MemoryCandidate(
                    content=(
                        "The user is testing the "
                        "Ultron memory manager."
                    ),
                    memory_type="technical",
                    importance=3.0,
                    confidence=1.0,
                    scope="user",
                )
            ]

        if message == "GENERAL_TEST":
            return [
                MemoryCandidate(
                    content=(
                        "Python is a general-purpose "
                        "programming language."
                    ),
                    memory_type="fact",
                    importance=2.0,
                    confidence=1.0,
                    scope="general",
                )
            ]

        return []

    def resolve(
        self,
        candidate: MemoryCandidate,
        existing_memory: dict,
    ) -> MemoryAction:

        return MemoryAction(
            action="duplicate",
            content=candidate.content,
            memory_type=candidate.memory_type,
            importance=candidate.importance,
        )


def self_test() -> None:
    """
    Test the manager's parsing and configuration logic without
    writing anything to the real database and without starting
    Ollama.
    """

    print("Memory manager self-test starting...")

    # --------------------------------------------------------
    # Candidate parsing
    # --------------------------------------------------------

    raw = """
    {
        "memories": [
            {
                "content": "The user uses Python.",
                "memory_type": "technical",
                "importance": 4,
                "confidence": 0.95,
                "scope": "user"
            }
        ]
    }
    """

    candidates = _parse_candidates(raw)

    if len(candidates) != 1:
        raise RuntimeError(
            "Candidate parsing failed."
        )

    candidate = candidates[0]

    if candidate.content != "The user uses Python.":
        raise RuntimeError(
            "Candidate content parsing failed."
        )

    if candidate.memory_type != "technical":
        raise RuntimeError(
            "Candidate type parsing failed."
        )

    if candidate.scope != "user":
        raise RuntimeError(
            "Candidate scope parsing failed."
        )

    # --------------------------------------------------------
    # General scope parsing
    # --------------------------------------------------------

    general_raw = """
    {
        "memories": [
            {
                "content": "Python is a programming language.",
                "memory_type": "fact",
                "importance": 2,
                "confidence": 0.95,
                "scope": "general"
            }
        ]
    }
    """

    general_candidates = _parse_candidates(
        general_raw
    )

    if len(general_candidates) != 1:
        raise RuntimeError(
            "General scope parsing failed."
        )

    if general_candidates[0].scope != "general":
        raise RuntimeError(
            "General scope normalization failed."
        )

    # --------------------------------------------------------
    # Invalid scope handling
    # --------------------------------------------------------

    invalid_scope_raw = """
    {
        "memories": [
            {
                "content": "Test invalid scope.",
                "memory_type": "fact",
                "importance": 2,
                "confidence": 0.95,
                "scope": "something_invalid"
            }
        ]
    }
    """

    invalid_candidates = _parse_candidates(
        invalid_scope_raw
    )

    if len(invalid_candidates) != 1:
        raise RuntimeError(
            "Invalid scope parsing failed."
        )

    if invalid_candidates[0].scope != "general":
        raise RuntimeError(
            "Invalid scope was not safely normalized."
        )

    # --------------------------------------------------------
    # Missing scope handling
    # --------------------------------------------------------

    missing_scope_raw = """
    {
        "memories": [
            {
                "content": "Test missing scope.",
                "memory_type": "fact",
                "importance": 2,
                "confidence": 0.95
            }
        ]
    }
    """

    missing_scope_candidates = _parse_candidates(
        missing_scope_raw
    )

    if len(missing_scope_candidates) != 1:
        raise RuntimeError(
            "Missing scope parsing failed."
        )

    if missing_scope_candidates[0].scope != "general":
        raise RuntimeError(
            "Missing scope was not safely normalized."
        )

    # --------------------------------------------------------
    # Importance/confidence clamping
    # --------------------------------------------------------

    if _normalize_importance(999) != 5.0:
        raise RuntimeError(
            "Importance clamp failed."
        )

    if _normalize_confidence(-10) != 0.0:
        raise RuntimeError(
            "Confidence clamp failed."
        )

    # --------------------------------------------------------
    # Scope normalization
    # --------------------------------------------------------

    if _normalize_scope("USER") != "user":
        raise RuntimeError(
            "User scope normalization failed."
        )

    if _normalize_scope("PROJECT") != "project":
        raise RuntimeError(
            "Project scope normalization failed."
        )

    if _normalize_scope("GENERAL") != "general":
        raise RuntimeError(
            "General scope normalization failed."
        )

    if _normalize_scope("invalid") != "general":
        raise RuntimeError(
            "Invalid scope fallback failed."
        )

    if _normalize_scope(None) != "general":
        raise RuntimeError(
            "Missing scope fallback failed."
        )

    # --------------------------------------------------------
    # Context formatting
    # --------------------------------------------------------

    context = _format_recent_context(
        [
            {
                "role": "user",
                "content": "I am working on Ultron.",
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ]
    )

    if "I am working on Ultron." not in context:
        raise RuntimeError(
            "Recent-context formatting failed."
        )

    # --------------------------------------------------------
    # Mock analyzer
    # --------------------------------------------------------

    analyzer = _MockAnalyzer()

    results = analyzer.analyze(
        "STORE_TEST"
    )

    if len(results) != 1:
        raise RuntimeError(
            "Mock analyzer failed."
        )

    if results[0].scope != "user":
        raise RuntimeError(
            "Mock analyzer scope failed."
        )

    action = analyzer.resolve(
        results[0],
        {
            "id": 1,
            "content": (
                "The user is testing the "
                "Ultron memory manager."
            ),
        },
    )

    if action.action != "duplicate":
        raise RuntimeError(
            "Mock resolution failed."
        )

    # --------------------------------------------------------
    # Configuration check
    # --------------------------------------------------------

    info = get_manager_info()

    if not info["ollama_model"]:
        raise RuntimeError(
            "Ollama model configuration is empty."
        )

    if "general" not in info["allowed_memory_scopes"]:
        raise RuntimeError(
            "General scope missing from configuration."
        )

    if "user" not in info["storable_memory_scopes"]:
        raise RuntimeError(
            "User scope missing from storable scopes."
        )

    if "project" not in info["storable_memory_scopes"]:
        raise RuntimeError(
            "Project scope missing from storable scopes."
        )

    print("Memory manager self-test passed.")


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    self_test()