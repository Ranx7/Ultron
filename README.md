# Ultron

Ultron is a local Flask and Ollama assistant with external, inspectable
long-term memory. SQLite is the source of truth; FTS5 provides lexical
search and MiniLM embeddings provide semantic retrieval.

## Architecture

* `Ultron_Interface.py` stores messages, builds recent conversational
  context, retrieves relevant durable memories, and streams the response
  from Ollama.
* `memory/database.py` owns SQLite persistence, FTS5 synchronization, and
  embedding serialization.
* `memory/embeddings.py` creates and stores MiniLM embeddings.
* `memory/retrieval.py` performs hybrid lexical and semantic ranking.
* `memory/manager.py` extracts durable user/project candidates, rejects
  general knowledge through a Python scope gate, and resolves duplicate,
  replacement, and merge actions.

## Setup

Create and activate a virtual environment, then install the Python
dependencies:

```bash
python -m pip install -r requirements.txt
```

Ollama must be running locally and expose the configured models. By default
the interface uses `Ultron:latest`; memory extraction uses
`Ultron-Memory:latest`. Configure the memory service when needed with
`ULTRON_OLLAMA_URL`, `ULTRON_MEMORY_MODEL`, and `ULTRON_DB_PATH`.

Run the interface with:

```bash
flask --app Ultron_Interface run
```

## Verification

The lightweight module self-tests do not require a running Ollama service:

```bash
python -m memory.database
python -m memory.retrieval
python -m memory.manager
```

`memory/test_manager_behavior.py` is a real end-to-end test and requires
both the local Ollama memory model and the MiniLM model download/cache.
