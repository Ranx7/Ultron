from flask import Flask, render_template, request, jsonify, Response
from ollama import Client
from pathlib import Path

from memory.database import (
    initialize_database,
    add_message,
)

from memory.retrieval import (
    get_recent_messages,
    search_memories,
    search_conversation,
    get_counts,
    touch_memories,
)

from memory.manager import process_message


app = Flask(__name__)

MODEL = "Ultron:latest"

client = Client(
    host="http://127.0.0.1:11434"
)

initialize_database()


# ============================================================
# PROJECT / CODE INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


ALLOWED_CODE_FILES = {
    "Ultron_Interface.py",
    "bootstrap_memory.py",
    "memory/__init__.py",
    "memory/database.py",
    "memory/retrieval.py",
    "memory/manager.py",
}


MAX_CODE_CHARS = 12000


# ============================================================
# ULTRON SYSTEM IDENTITY
# ============================================================

ULTRON_SYSTEM = """
You are Ultron, the user's local AI assistant.

Persistent system facts:
- Your assistant name is Ultron.
- You run locally through Ollama on the user's computer.
- Your underlying model is Llama 3.2 3B Q4.
- You are accessed through a local Flask web interface.
- The application has a persistent SQLite memory database.
- The database stores conversation history and extracted long-term memories.
- The user has explicitly told you that their name is Randy.

The application also has a /code command.
When /code is used, the application can provide you with the actual
source code of authorized Python files from your local project.

IMPORTANT CONVERSATION RULES:

1. Conversation history is chronological.
2. Each timestamp belongs to the message immediately following it.
3. Messages with different topics are NOT automatically repetitive.
4. Do not accuse the user of repetition merely because similar words
   appeared in older messages.
5. Only describe something as repetitive when the current user message
   is genuinely repeating the same request or statement.
6. Do not assume that older messages are part of the user's current
   topic unless the content clearly connects them.
7. Prefer the user's newest message when determining what they want now.
8. Historical context exists to help you remember, not to force the
   conversation back onto an old topic.
9. A topic change is normal conversation and should be handled naturally.
10. Do not invent previous conversations or claim the user said
    something that does not appear in the supplied context.

When source code is supplied through /code:
- Treat the supplied source as real source code from the local application.
- Analyze only the source that is actually supplied.
- Do not pretend the code is fictional.
- Distinguish the Llama model from the Python application that controls
  its interface.

Do not claim that these facts are fictional or that this system is a cloud chat.
Do not claim to be ChatGPT.

You can be conversational and humorous, but follow the rules above.

TIMESTAMP RULES:
- The application/database is the only source of timestamps.
- Never invent, estimate, or calculate a timestamp.
- Never output a [Timestamp: ...] label unless the application explicitly provides that timestamp.
- Never claim something happened earlier, recently, today, or at a specific time unless the supplied conversation data proves it.

CONVERSATION MEMORY RULES:
- Retrieved conversation snippets are evidence, not assumptions.
- Never claim the user previously said something unless that exact topic or statement appears in the supplied conversation.
- If you cannot find evidence for a claimed previous conversation, say you do not have evidence of it.
- Do not invent missing conversation history.
- Do not interpret a normal conversational statement as a memory lookup unless the user clearly asks about the past.

CODE INSPECTION RULES:
- The supplied source code is authoritative.
- Analyze only the source code actually provided to you.
- Never invent source code.
- Never replace the supplied source with a simplified example.
- Never claim code exists unless it appears in the supplied source.
- If asked to reproduce code, reproduce the supplied code exactly.
- If something cannot be found in the supplied source, say:
  "I cannot find that in the supplied source."
- Do not fabricate imports, functions, classes, routes, variables, or implementations.
""".strip()


# ============================================================
# MEMORY RETRIEVAL GATING
# ============================================================

MEMORY_HINTS = (
    "remember",
    "recall",
    "forgot",
    "forget",
    "earlier",
    "before",
    "yesterday",
    "last time",
    "previous",
    "what did i say",
    "what was",
    "we discussed",
    "we talked about",
    "do you remember",
)


SIMPLE_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "hi ultron",
    "hello ultron",
    "hey ultron",
    "yo ultron",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "lol",
}


def should_retrieve_memory(user_message):
    """
    Decide whether the user's message is asking about
    previous information.

    This prevents ordinary conversation from automatically
    searching the long-term memory database.
    """

    text = " ".join(
        user_message.casefold().split()
    )

    if not text:
        return False

    # Simple social messages should never trigger memory retrieval.
    if text in SIMPLE_MESSAGES:
        return False

    # Explicit references to previous information trigger retrieval.
    return any(
        hint in text
        for hint in MEMORY_HINTS
    )


# ============================================================
# CODE FILE HELPERS
# ============================================================

def list_code_files():

    return sorted(
        ALLOWED_CODE_FILES
    )


def read_own_code(filename):

    filename = (
        filename
        .strip()
        .replace("\\", "/")
    )

    if filename not in ALLOWED_CODE_FILES:
        return None

    path = PROJECT_ROOT / filename

    if not path.is_file():
        return None

    try:

        code = path.read_text(
            encoding="utf-8"
        )

    except (
        OSError,
        UnicodeDecodeError
    ):

        return None

    if len(code) > MAX_CODE_CHARS:

        code = (
            code[:MAX_CODE_CHARS]
            + "\n\n"
            + "# ============================================================\n"
            + "# SOURCE TRUNCATED FOR CONTEXT LIMIT\n"
            + "# ============================================================\n"
            + f"# Original file length: {len(code)} characters\n"
            + f"# Characters shown: {MAX_CODE_CHARS}\n"
        )

    return code


# ============================================================
# NORMAL CHAT CONTEXT
# ============================================================

def build_context(user_message):

    context = [
        {
            "role": "system",
            "content": ULTRON_SYSTEM,
        }
    ]

    retrieve_memory = should_retrieve_memory(
        user_message
    )


    # --------------------------------------------------------
    # LONG-TERM MEMORY
    # --------------------------------------------------------

    if retrieve_memory:

        memories = search_memories(
            user_message,
            limit=6
        )

        if memories:

            memory_ids = [
                m["id"]
                for m in memories
            ]

            touch_memories(
                memory_ids
            )

            memory_text = "\n".join(
                f"- {m['content']}"
                for m in memories
            )

            context.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant long-term memories:\n"
                        + memory_text
                    ),
                }
            )


        # ----------------------------------------------------
        # RELEVANT OLD CONVERSATION
        # ----------------------------------------------------

        old = search_conversation(
            user_message,
            limit=4
        )

        if old:

            old_text = "\n".join(
                (
                    f"[{m['created_at']}] "
                    f"{m['role']}: "
                    f"{m['content']}"
                )
                for m in old
            )

            context.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant historical conversation excerpts.\n"
                        "These may be from an earlier point in time and "
                        "are provided only because they matched the topic:\n\n"
                        + old_text
                    ),
                }
            )


    # --------------------------------------------------------
    # RECENT CONVERSATION
    # --------------------------------------------------------

    normalized_message = " ".join(
        user_message.casefold().split()
    )

    # Greetings and other simple social messages don't need
    # previous conversation injected into the prompt.
    if normalized_message not in SIMPLE_MESSAGES:

        recent = get_recent_messages(
            limit=6
        )

        if recent:

            recent_context = [
                {
                    "role": "system",
                    "content": (
                        "Recent conversation, in chronological order. "
                        "Use timestamps to distinguish older messages "
                        "from the newest request."
                    ),
                }
            ]

            for message in recent:

                recent_context.append(
                    {
                        "role": message["role"],
                        "content": (
                            f"[Timestamp: {message['created_at']}]\n"
                            f"{message['content']}"
                        ),
                    }
                )

            context.extend(
                recent_context
            )


    # --------------------------------------------------------
    # CURRENT MESSAGE
    # --------------------------------------------------------

    context.append(
        {
            "role": "user",
            "content": (
                "[Timestamp: CURRENT]\n"
                + user_message
            ),
        }
    )

    return context


# ============================================================
# CODE INSPECTION CONTEXT
# ============================================================

def build_code_context(
    filename,
    code
):

    return [
        {
            "role": "system",
            "content": ULTRON_SYSTEM,
        },
        {
            "role": "system",
            "content": """
You are now performing a self-inspection of your own software.

The source code supplied below is authoritative.

Rules:
- Analyze only the code that is actually provided.
- Do not invent functions, variables, files, or behavior.
- Explain relationships between components only when supported
  by the supplied source.
- You may identify bugs, weaknesses, inefficiencies, and improvements.
- The code belongs to the local application that runs your interface.
""".strip(),
        },
        {
            "role": "user",
            "content": (
                f"Inspect this source file:\n\n"
                f"FILE: {filename}\n\n"
                f"```python\n"
                f"{code}\n"
                f"```\n\n"
                "Explain what this file does and how it relates "
                "to your operation."
            ),
        },
    ]


# ============================================================
# OLLAMA STREAM
# ============================================================

def stream_model(context):

    stream = client.chat(
        model=MODEL,
        messages=context,
        stream=True,
        think=True
    )

    for chunk in stream:

        text = chunk.message.content

        if not text:
            continue

        yield text


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat():

    data = request.get_json()

    if not data:

        return jsonify(
            {
                "error": "No JSON data received."
            }
        ), 400

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:

        return jsonify(
            {
                "error": "Message is empty."
            }
        ), 400


    # ========================================================
    # /CODE LIST
    # ========================================================

    if user_message.lower() == "/code list":

        files = list_code_files()

        return Response(
            "Files Ultron can inspect:\n\n"
            + "\n".join(
                f"- {filename}"
                for filename in files
            ),
            mimetype="text/plain; charset=utf-8",
        )


    # ========================================================
    # /CODE INSPECTION
    # ========================================================

    if user_message.lower().startswith("/code"):

        filename = user_message[
            5:
        ].strip()


        if not filename:

            return Response(
                (
                    "Usage:\n"
                    "/code list\n"
                    "/code Ultron_Interface.py\n"
                    "/code memory/retrieval.py\n"
                    "/code memory/database.py\n"
                    "/code memory/manager.py"
                ),
                mimetype="text/plain; charset=utf-8",
            )


        code = read_own_code(
            filename
        )


        if code is None:

            return Response(
                (
                    f"I cannot inspect '{filename}'.\n\n"
                    "Use '/code list' to see the files "
                    "I am allowed to inspect."
                ),
                mimetype="text/plain; charset=utf-8",
            )


        # ----------------------------------------------------
        # SAVE THE CODE COMMAND TO CONVERSATION HISTORY
        # ----------------------------------------------------

        add_message(
            "user",
            user_message
        )


        context = build_code_context(
            filename,
            code
        )


        def generate_code_response():

            full_response = ""

            try:

                for text in stream_model(
                    context
                ):

                    full_response += text

                    yield text


                # Save Ultron's code-inspection response.
                add_message(
                    "assistant",
                    full_response
                )


            except Exception as e:

                yield (
                    f"\n[Ollama error: {str(e)}]"
                )


        return Response(
            generate_code_response(),
            mimetype="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


    # ========================================================
    # NORMAL CHAT
    # ========================================================

    context = build_context(
        user_message
    )

    user_id = add_message(
        "user",
        user_message
    )


    process_message(
        user_message,
        source_message_id=user_id,
    )


    def generate():

        full_response = ""

        try:

            for text in stream_model(
                context
            ):

                full_response += text

                yield text


            add_message(
                "assistant",
                full_response
            )


        except Exception as e:

            yield (
                f"\n[Ollama error: {str(e)}]"
            )


    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# CLEAR
# ============================================================

@app.post("/clear")
def clear():

    return jsonify(
        {
            "success": True,
            "message": (
                "Conversation context reset. "
                "Persistent memory preserved."
            ),
        }
    )


# ============================================================
# MEMORY STATUS
# ============================================================

@app.get("/memory")
def memory():

    messages, memories = get_counts()

    return jsonify(
        {
            "stored_messages": messages,
            "long_term_memories": memories,
        }
    )


# ============================================================
# MEMORY SEARCH
# ============================================================

@app.get("/memory/search")
def memory_search():

    query = request.args.get(
        "q",
        ""
    ).strip()

    if not query:

        return jsonify(
            {
                "error": "Missing search query."
            }
        ), 400

    return jsonify(
        {
            "memories": search_memories(
                query,
                limit=20
            ),
            "conversation": search_conversation(
                query,
                limit=20
            ),
        }
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )