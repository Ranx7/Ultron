import re
import unicodedata

from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

from .database import add_or_update_memory


# ============================================================
# NLTK SETUP
# ============================================================

# No downloaded NLTK corpus required.
TOKENIZER = RegexpTokenizer(
    r"[A-Za-z0-9_']+"
)

STEMMER = SnowballStemmer(
    "english"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_MEMORY_SCORE = 4
MAX_MEMORY_TEXT = 500

MAX_TOPICS = 8


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been",
    "being", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "more", "most",
    "my", "of", "on", "or", "our", "ours", "she", "so",
    "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "too",
    "us", "very", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "yours",
}


# ============================================================
# COMMON NON-TOPIC WORDS
# ============================================================

GENERIC_TOPIC_WORDS = {
    "thing",
    "things",
    "stuff",
    "something",
    "someone",
    "somebody",
    "really",
    "good",
    "great",
    "nice",
    "new",
    "old",
    "current",
    "currently",
    "today",
    "tomorrow",
    "yesterday",
    "time",
    "way",
    "part",
    "idea",
    "problem",
    "question",
    "answer",
    "work",
    "working",
    "use",
    "using",
    "used",
    "make",
    "making",
    "made",
    "build",
    "building",
    "built",
    "want",
    "need",
    "like",
    "love",
    "think",
    "know",
    "got",
    "get",
}


# ============================================================
# TECHNICAL TOPICS
# ============================================================

TECHNICAL_TOPICS = {
    "python",
    "javascript",
    "typescript",
    "java",
    "csharp",
    "c++",
    "rust",
    "flask",
    "ollama",
    "llama",
    "qwen",
    "sqlite",
    "nltk",
    "esp32",
    "esp32s3",
    "blender",
    "linux",
    "windows",
    "android",
    "github",
    "gpu",
    "cpu",
    "model",
    "database",
    "server",
    "api",
    "html",
    "css",
    "javascript",
    "node",
    "nodejs",
    "react",
    "pyqt",
    "pytorch",
    "tensorflow",
    "machine",
    "learning",
    "neural",
    "network",
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(text):

    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        str(text)
    ).strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def _split_sentences(text):

    text = _normalize(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# TOKENIZATION
# ============================================================

def _tokenize(text):

    return [
        token.lower()
        for token in TOKENIZER.tokenize(
            _normalize(text)
        )
    ]


# ============================================================
# STEMMING
# ============================================================

def _stems(tokens):

    return [
        STEMMER.stem(token)
        for token in tokens
        if len(token) >= 3
    ]


# ============================================================
# NAME EXTRACTION
# ============================================================

def _extract_name(text):

    patterns = [
        r"\bmy name is ([A-Za-z][A-Za-z0-9_-]{1,30})\b",
        r"\bi am ([A-Za-z][A-Za-z0-9_-]{1,30})\b",
        r"\bi'm ([A-Za-z][A-Za-z0-9_-]{1,30})\b",
    ]

    invalid_names = {
        "here",
        "fine",
        "good",
        "okay",
        "ok",
        "trying",
        "working",
        "learning",
        "going",
        "using",
        "building",
        "starting",
        "looking",
    }

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        candidate = match.group(1).strip()

        if candidate.lower() in invalid_names:
            continue

        return candidate

    return None


# ============================================================
# TOPIC EXTRACTION
# ============================================================

def _extract_topics(text):

    """
    Extract meaningful topic words from a memory.

    This is intentionally lightweight.

    Example:

        "I'm building a Python automation application"

    becomes approximately:

        ["python", "automation", "application"]
    """

    tokens = _tokenize(text)

    topics = []
    seen = set()

    for token in tokens:

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        if token in GENERIC_TOPIC_WORDS:
            continue

        if token in seen:
            continue

        seen.add(token)

        topics.append(token)

    # Give known technical terms priority.
    topics.sort(
        key=lambda x: (
            x not in TECHNICAL_TOPICS,
            -len(x)
        )
    )

    return topics[:MAX_TOPICS]


# ============================================================
# MEMORY TYPE
# ============================================================

def _detect_memory_type(
    text,
    lower
):

    # Identity
    if _extract_name(text):
        return "identity"

    # Preferences
    if any(
        phrase in lower
        for phrase in (
            "i like",
            "i love",
            "i enjoy",
            "my favorite",
            "i prefer",
            "i usually",
            "i tend to",
            "i don't like",
            "i dislike",
            "i hate",
            "i avoid",
            "i don't enjoy",
        )
    ):
        return "preference"

    # Projects
    if any(
        phrase in lower
        for phrase in (
            "i am building",
            "i'm building",
            "i am making",
            "i'm making",
            "i am working on",
            "i'm working on",
            "i am developing",
            "i'm developing",
            "my project",
            "my app",
            "my application",
            "my system",
            "i created",
            "i built",
            "i made",
        )
    ):
        return "project"

    # Goals
    if any(
        phrase in lower
        for phrase in (
            "i plan to",
            "i'm planning to",
            "i am planning to",
            "i want to",
            "i intend to",
            "i'm going to",
            "i am going to",
            "i will",
            "my goal is",
            "i hope to",
        )
    ):
        return "goal"

    # Personal facts
    if any(
        phrase in lower
        for phrase in (
            "i live in",
            "i'm from",
            "i am from",
            "i use",
            "i switched to",
            "i started",
            "i learned",
            "i studied",
            "i have",
            "i own",
        )
    ):
        return "personal"

    # Technical
    tokens = set(_tokenize(text))

    if tokens.intersection(TECHNICAL_TOPICS):
        return "technical"

    # Date/event
    if re.search(
        r"\b(19\d{2}|20\d{2})\b",
        lower
    ):
        return "event"

    return "general"


# ============================================================
# MEMORY CANDIDATE ANALYSIS
# ============================================================

def _analyze_sentence(sentence):

    text = _normalize(sentence)

    if not text:
        return None

    if len(text) < 8:
        return None

    if len(text) > MAX_MEMORY_TEXT:
        text = text[:MAX_MEMORY_TEXT]

    lower = text.lower()

    tokens = _tokenize(text)

    useful_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    stems = _stems(
        useful_tokens
    )

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    question = (
        "?" in text
        or re.search(
            r"^\s*(what|why|how|when|where|who|can|could|do|does|did|is|are|would|will)\b",
            lower
        )
    )

    if question:

        if not re.search(
            r"\bremember\b",
            lower
        ):
            return None

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0

    memory_type = _detect_memory_type(
        text,
        lower
    )

    # --------------------------------------------------------
    # Explicit memory request
    # --------------------------------------------------------

    if re.search(
        r"\bremember\b",
        lower
    ):
        score += 6
        memory_type = "explicit"

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    name = _extract_name(text)

    if name:

        return {
            "save": True,
            "type": "identity",
            "importance": 8,
            "content": f"The user's name is {name}.",
            "topics": ["identity", "name"],
        }

    # --------------------------------------------------------
    # Preferences
    # --------------------------------------------------------

    if memory_type == "preference":
        score += 5

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    if memory_type == "project":
        score += 5

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    if memory_type == "goal":
        score += 4

    # --------------------------------------------------------
    # Personal
    # --------------------------------------------------------

    if memory_type == "personal":
        score += 3

    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    technical_hits = sum(
        1
        for token in useful_tokens
        if token in TECHNICAL_TOPICS
    )

    if technical_hits:

        score += min(
            technical_hits * 2,
            4
        )

    # --------------------------------------------------------
    # Date / event
    # --------------------------------------------------------

    date_hit = re.search(
        r"\b(19\d{2}|20\d{2})\b",
        lower
    )

    month_names = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }

    month_hit = any(
        month in lower
        for month in month_names
    )

    if date_hit or month_hit:

        score += 2

        if memory_type == "general":
            memory_type = "event"

    # --------------------------------------------------------
    # Temporary information
    # --------------------------------------------------------

    temporary_terms = {
        "today",
        "tonight",
        "yesterday",
        "right",
        "now",
        "currently",
        "just",
        "earlier",
        "this morning",
        "this afternoon",
        "this evening",
    }

    temporary_hits = sum(
        1
        for phrase in temporary_terms
        if phrase in lower
    )

    if temporary_hits:
        score -= 2

    # --------------------------------------------------------
    # Short messages
    # --------------------------------------------------------

    if len(useful_tokens) <= 2:
        score -= 3

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    if question and not re.search(
        r"\bremember\b",
        lower
    ):
        score -= 5

    # --------------------------------------------------------
    # Meaningful vocabulary
    # --------------------------------------------------------

    unique_stems = set(stems)

    if len(unique_stems) >= 3:
        score += 1

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    topics = _extract_topics(text)

    # A memory without meaningful topics is less useful
    # for topic-based retrieval.
    if not topics:
        score -= 2

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    save = (
        score >= MIN_MEMORY_SCORE
    )

    if not save:
        return None

    return {
        "save": True,
        "type": memory_type,
        "importance": min(
            max(score, 1),
            10
        ),
        "content": text,
        "topics": topics,
    }


# ============================================================
# MEMORY STORAGE
# ============================================================

def _save_candidate(
    candidate,
    source_message_id=None
):

    if not candidate:
        return

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Your current database schema stores:
    #
    #   content
    #   memory_type
    #   importance
    #
    # It does not currently have a topics column.
    #
    # Therefore we keep topics attached to the returned
    # candidate for the new retrieval logic, while preserving
    # compatibility with your existing database.py.
    #
    # The topic information is also appended to the stored
    # memory in a lightweight metadata form.
    #
    # --------------------------------------------------------

    topics = candidate.get(
        "topics",
        []
    )

    content = candidate["content"]

    if topics:

        topic_text = ", ".join(
            topics
        )

        content_for_storage = (
            f"{content}\n"
            f"[Topics: {topic_text}]"
        )

    else:

        content_for_storage = content

    try:

        add_or_update_memory(
            content=content_for_storage,
            memory_type=candidate["type"],
            importance=candidate["importance"],
            source_message_id=source_message_id,
        )

    except TypeError:

        add_or_update_memory(
            content_for_storage,
            candidate["type"],
            candidate["importance"],
            source_message_id,
        )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def process_message(
    message,
    source_message_id=None
):

    """
    Analyze one user message and save durable information.

    Returns the memories that were classified and saved.
    """

    message = _normalize(message)

    if not message:
        return []

    sentences = _split_sentences(
        message
    )

    saved = []

    for sentence in sentences:

        candidate = _analyze_sentence(
            sentence
        )

        if not candidate:
            continue

        _save_candidate(
            candidate,
            source_message_id
        )

        saved.append(
            candidate
        )

    return saved