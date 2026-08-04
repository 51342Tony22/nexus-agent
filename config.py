# config.py
# 读取env
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv(
    "SILICONFLOW_MODEL",
    "deepseek-ai/DeepSeek-V4-Flash",
)

# Keep one provider attempt bounded. A user or higher-level Agent policy can
# retry deliberately; automatic SDK retries multiply the timeout while sending
# the exact same long prompt again.
LLM_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
)
LLM_MAX_RETRIES = max(
    0,
    int(os.getenv("LLM_MAX_RETRIES", "0")),
)
LLM_MAX_OUTPUT_TOKENS = max(
    128,
    int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200")),
)

SHOW_DEBUG = os.getenv("SHOW_DEBUG", "false").lower() == "true"

# Embedding retrieval configuration.
# load_dotenv() runs before sentence-transformers is imported, so HF_ENDPOINT
# from .env can be picked up by huggingface_hub during its import.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
EMBEDDING_LOCAL_FILES_ONLY = (
    os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE") or None
EMBEDDING_CPU_THREADS = max(
    1,
    int(os.getenv("EMBEDDING_CPU_THREADS", "1"))
)
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")

# Character budgets are deterministic and provider-independent. The Debug
# output also reports a rough token estimate, while the API usage remains the
# source of truth for actual prompt tokens.
CONTEXT_SUMMARY_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_SUMMARY_MAX_CHARS", "1200"))
)
CONTEXT_HISTORY_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_HISTORY_MAX_CHARS", "2400"))
)
CONTEXT_MEMORY_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_MEMORY_MAX_CHARS", "1600"))
)
CONTEXT_PERSISTENT_MEMORY_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_PERSISTENT_MEMORY_MAX_CHARS", "1200"))
)
CONTEXT_PLAN_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_PLAN_MAX_CHARS", "1000"))
)
CONTEXT_TOOL_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_TOOL_MAX_CHARS", "2000"))
)
CONTEXT_EXTERNAL_DATA_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_EXTERNAL_DATA_MAX_CHARS", "4000"))
)
CONTEXT_EVIDENCE_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_EVIDENCE_MAX_CHARS", "6000"))
)
CONTEXT_DECISION_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_DECISION_MAX_CHARS", "800"))
)
CONTEXT_EVALUATION_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_EVALUATION_MAX_CHARS", "1200"))
)
CONTEXT_AGENT_LOOP_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_AGENT_LOOP_MAX_CHARS", "1200"))
)
CONTEXT_OUTPUT_SCHEMA_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_OUTPUT_SCHEMA_MAX_CHARS", "5000"))
)
CONTEXT_OUTPUT_REPAIR_MAX_CHARS = max(
    0,
    int(os.getenv("CONTEXT_OUTPUT_REPAIR_MAX_CHARS", "3500"))
)
STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS = min(
    1,
    max(0, int(os.getenv("STRUCTURED_OUTPUT_MAX_REPAIR_ATTEMPTS", "1"))),
)

# Runtime Context Evaluation thresholds. These defaults were calibrated on the
# six local CoT-distillation papers and BAAI/bge-small-en; they are observable
# configuration, not universal relevance probabilities.
CONTEXT_MIN_EVIDENCE_CHARS = max(
    1,
    int(os.getenv("CONTEXT_MIN_EVIDENCE_CHARS", "80"))
)
CONTEXT_STRONG_QUERY_COVERAGE = float(
    os.getenv("CONTEXT_STRONG_QUERY_COVERAGE", "0.50")
)
CONTEXT_MODERATE_QUERY_COVERAGE = float(
    os.getenv("CONTEXT_MODERATE_QUERY_COVERAGE", "0.25")
)
CONTEXT_STRONG_SEMANTIC_SCORE = float(
    os.getenv("CONTEXT_STRONG_SEMANTIC_SCORE", "0.82")
)
CONTEXT_MODERATE_SEMANTIC_SCORE = float(
    os.getenv("CONTEXT_MODERATE_SEMANTIC_SCORE", "0.80")
)

# Agentic RAG loop guards. A hard retry limit prevents an insufficient query
# from turning into an unbounded retrieval loop.
AGENTIC_RAG_MAX_RETRIES = max(
    0,
    int(os.getenv("AGENTIC_RAG_MAX_RETRIES", "1"))
)
AGENTIC_RAG_MAX_MERGED_CHUNKS = max(
    1,
    int(os.getenv("AGENTIC_RAG_MAX_MERGED_CHUNKS", "6"))
)

# Observation-driven tool-loop guard. This is separate from retrieval retries:
# it limits newly proposed tools, not repeated retrieval attempts.
AGENT_TOOL_LOOP_MAX_STEPS = max(
    0,
    int(os.getenv("AGENT_TOOL_LOOP_MAX_STEPS", "2"))
)

# In-process document index cache. The cache is intentionally small and
# disappears when the Agent stops; this keeps v1.3.4 observable and avoids
# introducing persistent database lifecycle concerns.
DOCUMENT_CHUNK_SIZE = max(
    100,
    int(os.getenv("DOCUMENT_CHUNK_SIZE", "1200"))
)
DOCUMENT_CHUNK_OVERLAP = max(
    0,
    int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "200"))
)
DOCUMENT_CHUNK_OVERLAP = min(
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE - 1,
)
DOCUMENT_INDEX_CACHE_MAX_ENTRIES = max(
    1,
    int(os.getenv("DOCUMENT_INDEX_CACHE_MAX_ENTRIES", "6"))
)

# Persistent typed memory. Relative paths are resolved from the project root,
# not from whichever directory happened to launch main.py.
_PROJECT_ROOT = Path(__file__).resolve().parent
_configured_memory_path = os.getenv("MEMORY_STORE_PATH")
if _configured_memory_path:
    _memory_path = Path(_configured_memory_path).expanduser()
    if not _memory_path.is_absolute():
        _memory_path = _PROJECT_ROOT / _memory_path
else:
    _memory_path = _PROJECT_ROOT / "data" / "memories.json"
MEMORY_STORE_PATH = str(_memory_path.resolve())

# Persistent-memory retrieval stays small and observable. Relevance is an
# admission gate; importance, confidence and recency rank admitted records.
MEMORY_RETRIEVAL_TOP_K = max(
    0,
    int(os.getenv("MEMORY_RETRIEVAL_TOP_K", "3"))
)
MEMORY_RETRIEVAL_MIN_RELEVANCE = min(
    1.0,
    max(
        0.0,
        float(os.getenv("MEMORY_RETRIEVAL_MIN_RELEVANCE", "0.08")),
    ),
)

# Semantic Scholar paper-search provider. The API key is optional for public
# endpoints and is never included in Tool results or Debug output.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "10")),
)

def validate_config():
    if not SILICONFLOW_API_KEY:
        raise ValueError(
            "Missing SILICONFLOW_API_KEY. Please set it in your .env file."
        )
