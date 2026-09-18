import re
import unicodedata

from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

from .database import add_or_update_memory


# ============================================================
# NLTK SETUP
# ============================================================

# These require no downloaded NLTK corpus.
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


# Words that are extremely common and provide little useful
# information for memory classification.

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(text):
    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        text
    ).strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def _split_sentences(text):
    """
    Lightweight sentence splitter.

    We deliberately avoid nltk.sent_tokenize() because that
    normally requires an NLTK Punkt resource download.
    """

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
    """
    Tokenize using NLTK.

    Returns lowercase word-like tokens.
    """

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
# LINGUISTIC FEATURES
# ============================================================

def _features(sentence):

    text = sentence.lower()

    tokens = _tokenize(
        sentence
    )

    useful_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    stems = _stems(
        useful_tokens
    )

    return {
        "text": text,
        "tokens": tokens,
        "useful_tokens": useful_tokens,
        "stems": stems,
    }


# ============================================================
# PATTERN HELPERS
# ============================================================

def _matches(pattern, text):
    return re.search(
        pattern,
        text,
        re.IGNORECASE
    ) is not None


def _contains_any(text, phrases):
    return any(
        phrase in text
        for phrase in phrases
    )


# ============================================================
# IDENTITY EXTRACTION
# ============================================================

def _extract_name(text):
    """
    Detect explicit name statements.

    Examples:

        my name is Randy
        I'm Randy
        i am Randy

    We intentionally avoid treating every "I'm X" as a name.
    """

    patterns = [
        r"\bmy name is ([A-Za-z][A-Za-z0-9_-]{1,30})\b",

        r"\bi am ([A-Za-z][A-Za-z0-9_-]{1,30})\b",

        r"\bi'm ([A-Za-z][A-Za-z0-9_-]{1,30})\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        candidate = match.group(
            1
        ).strip()

        # Avoid common false positives.
        if candidate.lower() in {
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
        }:
            continue

        return candidate

    return None


# ============================================================
# MEMORY CANDIDATE ANALYSIS
# ============================================================

def _analyze_sentence(sentence):
    """
    Decide whether a single user sentence contains information
    that is likely to remain useful later.

    Returns:

        {
            "save": bool,
            "type": str,
            "importance": int,
            "content": str,
        }

    """

    text = _normalize(
        sentence
    )

    if not text:
        return None

    if len(text) < 8:
        return None

    if len(text) > MAX_MEMORY_TEXT:

        text = text[
            :MAX_MEMORY_TEXT
        ]

    lower = text.lower()

    features = _features(
        text
    )

    tokens = features["tokens"]
    useful_tokens = features["useful_tokens"]
    stems = features["stems"]


    # --------------------------------------------------------
    # Ignore obvious questions
    # --------------------------------------------------------

    question = (
        "?" in text
        or _matches(
            r"^\s*(what|why|how|when|where|who|can|could|do|does|did|is|are|would|will)\b",
            lower
        )
    )

    if question:

        # Explicit "remember..." is an exception.
        if not _matches(
            r"\bremember\b",
            lower
        ):
            return None


    # --------------------------------------------------------
    # Base scoring
    # --------------------------------------------------------

    score = 0

    memory_type = "general"


    # --------------------------------------------------------
    # Explicit memory request
    # --------------------------------------------------------

    if _matches(
        r"\bremember\b",
        lower
    ):

        score += 6

        memory_type = "explicit"


    # --------------------------------------------------------
    # NAME / IDENTITY
    # --------------------------------------------------------

    name = _extract_name(
        text
    )

    if name:

        score += 8

        memory_type = "identity"

        return {
            "save": True,
            "type": memory_type,
            "importance": min(score, 10),
            "content": f"The user's name is {name}.",
        }


    # --------------------------------------------------------
    # LIKES / PREFERENCES
    # --------------------------------------------------------

    preference_patterns = [
        r"\bi like\b",
        r"\bi love\b",
        r"\bi enjoy\b",
        r"\bmy favorite\b",
        r"\bi prefer\b",
        r"\bi usually\b",
        r"\bi tend to\b",
    ]

    dislike_patterns = [
        r"\bi don't like\b",
        r"\bi dislike\b",
        r"\bi hate\b",
        r"\bi avoid\b",
        r"\bi don't enjoy\b",
    ]


    if _contains_any(
        lower,
        preference_patterns
    ):

        score += 5
        memory_type = "preference"


    if _contains_any(
        lower,
        dislike_patterns
    ):

        score += 5
        memory_type = "preference"


    # --------------------------------------------------------
    # CURRENT PROJECT / WORK
    # --------------------------------------------------------

    project_patterns = [
        r"\bi am building\b",
        r"\bi'm building\b",
        r"\bi am making\b",
        r"\bi'm making\b",
        r"\bi am working on\b",
        r"\bi'm working on\b",
        r"\bi am developing\b",
        r"\bi'm developing\b",
        r"\bmy project\b",
        r"\bmy app\b",
        r"\bmy application\b",
        r"\bmy system\b",
        r"\bi created\b",
        r"\bi built\b",
        r"\bi made\b",
    ]

    if _contains_any(
        lower,
        project_patterns
    ):

        score += 5
        memory_type = "project"


    # --------------------------------------------------------
    # FUTURE PLANS / GOALS
    # --------------------------------------------------------

    future_patterns = [
        r"\bi plan to\b",
        r"\bi'm planning to\b",
        r"\bi am planning to\b",
        r"\bi want to\b",
        r"\bi intend to\b",
        r"\bi'm going to\b",
        r"\bi am going to\b",
        r"\bi will\b",
        r"\bmy goal is\b",
        r"\bi hope to\b",
    ]

    if _contains_any(
        lower,
        future_patterns
    ):

        score += 4
        memory_type = "goal"


    # --------------------------------------------------------
    # PERSONAL FACTS
    # --------------------------------------------------------

    personal_patterns = [
        r"\bi live in\b",
        r"\bi'm from\b",
        r"\bi am from\b",
        r"\bi use\b",
        r"\bi switched to\b",
        r"\bi started\b",
        r"\bi learned\b",
        r"\bi studied\b",
        r"\bi have\b",
        r"\bi own\b",
    ]

    if _contains_any(
        lower,
        personal_patterns
    ):

        score += 3

        if memory_type == "general":
            memory_type = "personal"


    # --------------------------------------------------------
    # TECHNOLOGY / SYSTEM FACTS
    # --------------------------------------------------------

    technical_terms = {
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
        "sqlite",
        "nltk",
        "esp32",
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
    }

    technical_hits = sum(
        1
        for token in useful_tokens
        if token in technical_terms
    )

    if technical_hits:

        score += min(
            technical_hits * 2,
            4
        )

        if memory_type == "general":
            memory_type = "technical"


    # --------------------------------------------------------
    # DATE / TIME / LONG-TERM EVENT
    # --------------------------------------------------------

    date_or_future = _matches(
        r"\b(20\d{2}|19\d{2})\b",
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

    if date_or_future or month_hit:

        score += 2

        if memory_type == "general":
            memory_type = "event"


    # --------------------------------------------------------
    # TEMPORARY / LOW-VALUE LANGUAGE
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
    # VERY SHORT CASUAL CHAT
    # --------------------------------------------------------

    if len(useful_tokens) <= 2:

        score -= 3


    # --------------------------------------------------------
    # STANDALONE QUESTIONS
    # --------------------------------------------------------

    if question and not _matches(
        r"\bremember\b",
        lower
    ):

        score -= 5


    # --------------------------------------------------------
    # STEM SIGNAL
    # --------------------------------------------------------

    # This is not a huge scoring factor. It simply rewards
    # sentences containing several meaningful lexical terms.

    unique_stems = set(
        stems
    )

    if len(unique_stems) >= 3:

        score += 1


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
    }


# ============================================================
# MEMORY STORAGE
# ============================================================

def _save_candidate(
    candidate,
    source_message_id=None
):
    """
    Send a classified memory to the existing SQLite manager.
    """

    if not candidate:
        return

    try:

        add_or_update_memory(
            content=candidate["content"],
            memory_type=candidate["type"],
            importance=candidate["importance"],
            source_message_id=source_message_id,
        )

    except TypeError:
        # Compatibility fallback in case your current
        # database.py uses an older function signature.

        add_or_update_memory(
            candidate["content"],
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
    Analyze one user message and save only durable candidates.

    Returns a list of memories that were saved.

    Example:

        "my name is Randy and I like Python"

    can produce:

        [
            {
                "type": "identity",
                "importance": 8,
                ...
            },
            {
                "type": "preference",
                "importance": 6,
                ...
            }
        ]
    """

    message = _normalize(
        message
    )

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