"""
Real behavior tests for Ultron's memory manager.

Uses the real Ollama memory model + MiniLM semantic retrieval.

Each run creates a unique test project so previous test data cannot
cause false duplicate matches.

All memories created by this test are removed at the end.
"""

from __future__ import annotations

import traceback
import uuid

from memory.manager import OllamaMemoryAnalyzer, process_message
from memory.database import get_connection


TEST_ID = uuid.uuid4().hex[:8].upper()
PROJECT_NAME = f"Quartz Garden {TEST_ID}"


def delete_memory_ids(memory_ids: set[int]) -> None:
    """Delete only memories created by this test run."""

    if not memory_ids:
        return

    conn = get_connection()

    try:
        placeholders = ",".join("?" for _ in memory_ids)

        # Remove embeddings first.
        conn.execute(
            f"""
            DELETE FROM memory_embeddings
            WHERE memory_id IN ({placeholders})
            """,
            tuple(memory_ids),
        )

        # Memory FTS synchronization is handled by the database triggers.
        conn.execute(
            f"""
            DELETE FROM memories
            WHERE id IN ({placeholders})
            """,
            tuple(memory_ids),
        )

        conn.commit()

    finally:
        conn.close()


def print_candidates(
    analyzer: OllamaMemoryAnalyzer,
    message: str,
) -> list:
    print()
    print("=" * 80)
    print("ANALYZER TEST")
    print("=" * 80)
    print("MESSAGE:")
    print(message)

    candidates = analyzer.analyze(message)

    print()
    print("CANDIDATES:")

    if not candidates:
        print("  []")
    else:
        for candidate in candidates:
            print(f"  content     : {candidate.content}")
            print(f"  memory_type : {candidate.memory_type}")
            print(f"  importance  : {candidate.importance}")
            print(f"  confidence  : {candidate.confidence}")

    return candidates


def run_manager_test(
    name: str,
    message: str,
) -> list:
    print()
    print("=" * 80)
    print(f"MANAGER TEST: {name}")
    print("=" * 80)
    print("MESSAGE:")
    print(message)

    result = process_message(message)

    print()
    print("RESULT:")
    print(result)

    return result


def collect_created_ids(
    result: list,
    created_ids: set[int],
) -> None:
    """
    Track memory IDs belonging to this test.

    Stored results look like:
        {'result': (memory_id, created)}

    We also track existing_memory_id because a resolver may operate
    on a test-created memory.
    """

    for item in result:
        if not isinstance(item, dict):
            continue

        existing_id = item.get("existing_memory_id")

        if isinstance(existing_id, int):
            created_ids.add(existing_id)

        stored_result = item.get("result")

        if (
            isinstance(stored_result, tuple)
            and len(stored_result) >= 1
            and isinstance(stored_result[0], int)
        ):
            created_ids.add(stored_result[0])


def main() -> None:
    print("Ultron memory manager behavior test starting...")
    print(f"Test ID: {TEST_ID}")
    print(f"Unique project: {PROJECT_NAME}")

    analyzer = OllamaMemoryAnalyzer()

    created_memory_ids: set[int] = set()

    try:
        # ================================================================
        # 1. Durable information
        # ================================================================

        durable_message = (
            f"I am building a small automation project called "
            f"{PROJECT_NAME} on my Windows laptop."
        )

        candidates = print_candidates(
            analyzer,
            durable_message,
        )

        if candidates:
            print("\n✅ PASS: durable message produced a candidate.")
        else:
            print("\n❌ FAIL: durable message produced no candidates.")

        # ================================================================
        # 2. NEW MEMORY
        # ================================================================

        first_result = run_manager_test(
            "NEW DURABLE MEMORY",
            durable_message,
        )

        collect_created_ids(first_result, created_memory_ids)

        stored = any(
            isinstance(item, dict)
            and item.get("action") == "stored"
            for item in first_result
        )

        if stored:
            print("\n✅ PASS: brand-new memory was stored.")
        else:
            print(
                "\n❌ FAIL: brand-new memory was not stored."
            )

        # ================================================================
        # 3. DUPLICATE
        # ================================================================

        duplicate_message = (
            f"{PROJECT_NAME} is my Windows laptop automation project."
        )

        duplicate_result = run_manager_test(
            "DUPLICATE",
            duplicate_message,
        )

        collect_created_ids(
            duplicate_result,
            created_memory_ids,
        )

        duplicate = any(
            isinstance(item, dict)
            and item.get("action") == "duplicate"
            for item in duplicate_result
        )

        if duplicate:
            print("\n✅ PASS: duplicate was detected.")
        else:
            print("\n❌ FAIL: duplicate was not detected.")

        # ================================================================
        # 4. CORRECTION
        # ================================================================

        correction_message = (
            f"{PROJECT_NAME} is no longer running on Windows; "
            f"the project has moved to Linux."
        )

        correction_candidates = print_candidates(
            analyzer,
            correction_message,
        )

        if correction_candidates:
            print(
                "\n✅ PASS: correction produced a candidate."
            )
        else:
            print(
                "\n❌ FAIL: correction produced no candidate."
            )

        correction_result = run_manager_test(
            "CORRECTION / RELATED MEMORY",
            correction_message,
        )

        collect_created_ids(
            correction_result,
            created_memory_ids,
        )

        print("\nExpected:")
        print("  replace / merge the existing Windows project memory")

        print("\nActual:")
        print(f"  {correction_result}")

        # ================================================================
        # 5. RELATED BUT DIFFERENT MEMORY
        # ================================================================

        preference_message = (
            f"For {PROJECT_NAME}, I prefer Python for automation scripts."
        )

        preference_candidates = print_candidates(
            analyzer,
            preference_message,
        )

        if preference_candidates:
            print(
                "\n✅ PASS: preference produced a candidate."
            )
        else:
            print(
                "\n❌ FAIL: preference produced no candidate."
            )

        preference_result = run_manager_test(
            "RELATED BUT DIFFERENT MEMORY",
            preference_message,
        )

        collect_created_ids(
            preference_result,
            created_memory_ids,
        )

        stored_or_updated = any(
            isinstance(item, dict)
            and item.get("action") in {
                "stored",
                "replace",
                "merge",
            }
            for item in preference_result
        )

        if stored_or_updated:
            print(
                "\n✅ PASS: preference was handled as durable information."
            )
        else:
            print(
                "\n⚠️ CHECK: preference result was:"
            )
            print(preference_result)

        # ================================================================
        # 6. TEMPORARY INFORMATION
        # ================================================================

        temporary_message = "I am hungry right now."

        temporary_candidates = print_candidates(
            analyzer,
            temporary_message,
        )

        if not temporary_candidates:
            print("\n✅ PASS: temporary statement was rejected.")
        else:
            print(
                "\n❌ FAIL: temporary statement became a candidate."
            )

        temporary_result = run_manager_test(
            "TEMPORARY INFORMATION",
            temporary_message,
        )

        if not temporary_result:
            print(
                "\n✅ PASS: temporary information was not stored."
            )
        else:
            print(
                "\n❌ FAIL: temporary information reached storage:"
            )
            print(temporary_result)

        # ================================================================
        # 7. ORDINARY QUESTION — ANALYZER ×5
        # ================================================================

        question_message = "How does semantic search work?"

        print()
        print("=" * 80)
        print("ANALYZER CONSISTENCY TEST")
        print("=" * 80)
        print("MESSAGE:")
        print(question_message)

        question_runs = []

        for i in range(5):
            result = analyzer.analyze(question_message)
            question_runs.append(result)

            print()
            print(f"RUN {i + 1}:")
            print(result)

        if all(not result for result in question_runs):
            print(
                "\n✅ PASS: all 5 analyzer runs rejected the question."
            )
        else:
            print(
                "\n❌ FAIL: analyzer inconsistently classified the question."
            )

        # ================================================================
        # 8. ORDINARY QUESTION — FULL MANAGER
        # ================================================================

        question_result = run_manager_test(
            "ORDINARY QUESTION",
            question_message,
        )

        collect_created_ids(
            question_result,
            created_memory_ids,
        )

        if not question_result:
            print(
                "\n✅ PASS: ordinary question produced no memory action."
            )
        else:
            print(
                "\n❌ FAIL: ordinary question reached the memory manager:"
            )
            print(question_result)

    except Exception:
        print()
        print("=" * 80)
        print("❌ TEST CRASHED")
        print("=" * 80)

        traceback.print_exc()

    finally:
        print()
        print("=" * 80)
        print("CLEANUP")
        print("=" * 80)

        print(
            f"Removing {len(created_memory_ids)} "
            "test-created memory/memories..."
        )

        delete_memory_ids(created_memory_ids)

        print("✅ Test memories removed.")
        print()
        print("Behavior test finished.")


if __name__ == "__main__":
    main()