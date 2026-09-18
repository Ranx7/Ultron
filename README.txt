ULTRON MEMORY UPGRADE
=====================

What this changes
-----------------
1. Conversation history is permanently stored in SQLite.
2. Long-term memories are stored separately from raw chat history.
3. SQLite FTS5 provides cheap keyword retrieval.
4. A small set of relevant memories is injected into Ultron's context.
5. Ultron has authoritative system facts so the 3B model does not
   repeatedly deny its own local setup.
6. Memory survives Flask restarts.
7. /clear no longer deletes persistent memory.

Install
-------
No new Python package is required for the memory system.

Your existing Ollama + Flask + ollama Python package are enough.

Files
-----
Ultron_Interface.py
memory/database.py
memory/retrieval.py
memory/manager.py
memory/__init__.py
bootstrap_memory.py

Migration
---------
1. Put the `memory` folder beside Ultron_Interface.py.
2. Put your existing `data/ultron_memory.db` in:
       data\ultron_memory.db
3. Run:
       python bootstrap_memory.py
4. Start Ultron normally:
       python Ultron_Interface.py

IMPORTANT
---------
The bootstrap script assumes the model identity "Llama 3.2 3B Q4"
because that was explicitly established in the conversation.

The raw conversation remains in SQLite. The bootstrap only adds
persistent facts; it does not erase the existing messages.

Testing
-------
After starting the Flask server, test:

    GET /memory

You should see at least:
    "stored_messages": 6
    "long_term_memories": 6

Then ask Ultron:

    what's my name?

and:

    what are you?

It should receive the persistent facts before generation.

Later upgrades
--------------
Phase 2 can add:
- smarter memory extraction
- memory correction/update
- session/conversation IDs
- summaries for very old conversations
- optional embeddings/hybrid semantic search
- memory management UI
- SQLite backups
- memory importance/confidence
