# Run this ONCE against an existing ultron_memory.db.
# It adds the system facts that were already explicitly established
# in the original conversation, without duplicating chat history.

from memory.database import initialize_database, add_or_update_memory

initialize_database()

facts = [
    ("The user's name is Randy.", "identity", 5),
    ("The assistant's name is Ultron.", "system", 5),
    ("Ultron runs locally through Ollama on the user's computer.", "system", 5),
    ("Ultron uses a Llama 3.2 3B Q4 underlying model.", "system", 5),
    ("Ultron is accessed through a local Flask web interface.", "system", 5),
    ("Ultron has persistent memory stored in a SQLite database.", "system", 5),
]

for content, memory_type, importance in facts:
    memory_id, created = add_or_update_memory(
        content,
        memory_type=memory_type,
        importance=importance,
    )
    print(
        ("ADDED" if created else "ALREADY EXISTS"),
        memory_id,
        content,
    )

print("\nBootstrap complete.")
