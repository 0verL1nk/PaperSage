import os
from dataclasses import dataclass

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_LOCAL_EMBEDDING_FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_EMBEDDING_CACHE_DIR = "./models/embeddings"
DEFAULT_RAG_CHUNK_SIZE = 500
DEFAULT_RAG_CHUNK_OVERLAP = 80
DEFAULT_RAG_DENSE_CANDIDATE_K = 30
DEFAULT_RAG_SPARSE_CANDIDATE_K = 30
DEFAULT_RAG_RRF_CANDIDATE_K = 40
DEFAULT_RAG_RERANK_CANDIDATE_K = 50
DEFAULT_RAG_TOP_K = 8
DEFAULT_RAG_RERANK_ENABLED = False
DEFAULT_RAG_PROJECT_MAX_CHARS = 300000
DEFAULT_RAG_PROJECT_MAX_CHUNKS = 1200
DEFAULT_RAG_PROJECT_RERANK_ENABLED = False
DEFAULT_RAG_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RAG_HYBRID_ENABLED = True
DEFAULT_RAG_NEIGHBOR_EXPANSION = True
DEFAULT_RAG_NEIGHBOR_COUNT = 1
DEFAULT_RAG_QUERY_PREPROCESS_ENABLED = False
DEFAULT_AGENT_TEMPERATURE = 0.1
DEFAULT_AGENT_ENABLE_THINKING = False
DEFAULT_AGENT_REASONING_EFFORT = ""
DEFAULT_AGENT_TEAM_MAX_MEMBERS = 3
DEFAULT_AGENT_TEAM_MAX_ROUNDS = 2
DEFAULT_AGENT_TEAM_MEMBERS_HARD_CAP = 6
DEFAULT_AGENT_TEAM_ROUNDS_HARD_CAP = 4
DEFAULT_AGENT_ROUTING_CONTEXT_RECENT_LIMIT = 6
DEFAULT_AGENT_ROUTING_CONTEXT_MAX_CHARS = 2200
DEFAULT_AGENT_ROUTING_CONTEXT_ITEM_MAX_CHARS = 180
DEFAULT_AGENT_ROUTING_CONTEXT_REASON_MAX_CHARS = 120
DEFAULT_AGENT_ROUTING_CONTEXT_ROLES_PREVIEW_COUNT = 3
DEFAULT_AGENT_POLICY_TEXT_LEN_MEDIUM = 140
DEFAULT_AGENT_POLICY_TEXT_LEN_HIGH = 240
DEFAULT_AGENT_POLICY_SENTENCE_THRESHOLD = 3
DEFAULT_AGENT_POLICY_COMMA_THRESHOLD = 4
DEFAULT_AGENT_POLICY_QUESTION_THRESHOLD = 2
DEFAULT_AGENT_POLICY_ENUM_STEPS_THRESHOLD = 2
DEFAULT_AGENT_POLICY_CONJUNCTION_THRESHOLD = 2
DEFAULT_AGENT_POLICY_CONTEXT_CHARS_THRESHOLD = 180
DEFAULT_AGENT_POLICY_CONTEXT_LINES_THRESHOLD = 4
DEFAULT_AGENT_POLICY_SCORE_PLAN = 2
DEFAULT_AGENT_POLICY_SCORE_TEAM = 4
DEFAULT_AGENT_PLANNER_MIN_STEPS = 2
DEFAULT_AGENT_PLANNER_MAX_STEPS = 4
DEFAULT_AGENT_POLICY_ROUTER_MODEL_NAME = ""
DEFAULT_AGENT_POLICY_ROUTER_BASE_URL = ""
DEFAULT_AGENT_POLICY_ROUTER_TEMPERATURE = 0.0
DEFAULT_AGENT_POLICY_ASYNC_ENABLED = True
DEFAULT_AGENT_POLICY_ASYNC_REFRESH_SECONDS = 4.0
DEFAULT_AGENT_POLICY_ASYNC_MIN_CONFIDENCE = 0.6
DEFAULT_AGENT_POLICY_ASYNC_MAX_STALENESS_SECONDS = 20.0
DEFAULT_AGENT_LLM_REQUEST_TIMEOUT = 120.0


@dataclass(frozen=True)
class AgentSettings:
    openai_compatible_base_url: str
    local_embedding_model: str
    local_embedding_fallback_model: str
    local_embedding_cache_dir: str
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_dense_candidate_k: int
    rag_sparse_candidate_k: int
    rag_rrf_candidate_k: int
    rag_rerank_candidate_k: int
    rag_top_k: int
    rag_rerank_enabled: bool
    rag_project_max_chars: int
    rag_project_max_chunks: int
    rag_project_rerank_enabled: bool
    rag_rerank_model: str
    rag_hybrid_enabled: bool
    rag_neighbor_expansion: bool
    rag_neighbor_count: int
    rag_query_preprocess_enabled: bool
    agent_temperature: float
    agent_enable_thinking: bool
    agent_reasoning_effort: str
    agent_team_max_members: int
    agent_team_max_rounds: int
    agent_team_members_hard_cap: int
    agent_team_rounds_hard_cap: int
    agent_routing_context_recent_limit: int
    agent_routing_context_max_chars: int
    agent_routing_context_item_max_chars: int
    agent_routing_context_reason_max_chars: int
    agent_routing_context_roles_preview_count: int
    agent_policy_text_len_medium: int
    agent_policy_text_len_high: int
    agent_policy_sentence_threshold: int
    agent_policy_comma_threshold: int
    agent_policy_question_threshold: int
    agent_policy_enum_steps_threshold: int
    agent_policy_conjunction_threshold: int
    agent_policy_context_chars_threshold: int
    agent_policy_context_lines_threshold: int
    agent_policy_score_plan: int
    agent_policy_score_team: int
    agent_planner_min_steps: int
    agent_planner_max_steps: int
    agent_policy_router_model_name: str
    agent_policy_router_base_url: str
    agent_policy_router_temperature: float
    agent_policy_async_enabled: bool
    agent_policy_async_refresh_seconds: float
    agent_policy_async_min_confidence: float
    agent_policy_async_max_staleness_seconds: float
    agent_llm_request_timeout: float


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def load_agent_settings() -> AgentSettings:
    return AgentSettings(
        openai_compatible_base_url=os.getenv(
            "OPENAI_COMPATIBLE_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL
        ),
        local_embedding_model=os.getenv(
            "LOCAL_RAG_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL
        ),
        local_embedding_fallback_model=os.getenv(
            "LOCAL_RAG_EMBEDDING_FALLBACK_MODEL",
            DEFAULT_LOCAL_EMBEDDING_FALLBACK_MODEL,
        ),
        local_embedding_cache_dir=os.getenv(
            "LOCAL_RAG_EMBEDDING_CACHE_DIR", DEFAULT_LOCAL_EMBEDDING_CACHE_DIR
        ),
        rag_chunk_size=_env_int("LOCAL_RAG_CHUNK_SIZE", DEFAULT_RAG_CHUNK_SIZE),
        rag_chunk_overlap=_env_int("LOCAL_RAG_CHUNK_OVERLAP", DEFAULT_RAG_CHUNK_OVERLAP),
        rag_dense_candidate_k=_env_int("LOCAL_RAG_DENSE_CANDIDATE_K", DEFAULT_RAG_DENSE_CANDIDATE_K),
        rag_sparse_candidate_k=_env_int("LOCAL_RAG_SPARSE_CANDIDATE_K", DEFAULT_RAG_SPARSE_CANDIDATE_K),
        rag_rrf_candidate_k=_env_int("LOCAL_RAG_RRF_CANDIDATE_K", DEFAULT_RAG_RRF_CANDIDATE_K),
        rag_rerank_candidate_k=_env_int("LOCAL_RAG_RERANK_CANDIDATE_K", DEFAULT_RAG_RERANK_CANDIDATE_K),
        rag_top_k=_env_int("LOCAL_RAG_TOP_K", DEFAULT_RAG_TOP_K),
        rag_rerank_enabled=_env_bool("LOCAL_RAG_RERANK_ENABLED", DEFAULT_RAG_RERANK_ENABLED),
        rag_project_max_chars=_env_int("LOCAL_RAG_PROJECT_MAX_CHARS", DEFAULT_RAG_PROJECT_MAX_CHARS),
        rag_project_max_chunks=_env_int("LOCAL_RAG_PROJECT_MAX_CHUNKS", DEFAULT_RAG_PROJECT_MAX_CHUNKS),
        rag_project_rerank_enabled=_env_bool(
            "LOCAL_RAG_PROJECT_RERANK_ENABLED",
            DEFAULT_RAG_PROJECT_RERANK_ENABLED,
        ),
        rag_rerank_model=os.getenv(
            "LOCAL_RAG_RERANK_MODEL", DEFAULT_RAG_RERANK_MODEL
        ),
        rag_hybrid_enabled=_env_bool("LOCAL_RAG_HYBRID_ENABLED", DEFAULT_RAG_HYBRID_ENABLED),
        rag_neighbor_expansion=_env_bool("LOCAL_RAG_NEIGHBOR_EXPANSION", DEFAULT_RAG_NEIGHBOR_EXPANSION),
        rag_neighbor_count=_env_int("LOCAL_RAG_NEIGHBOR_COUNT", DEFAULT_RAG_NEIGHBOR_COUNT),
        rag_query_preprocess_enabled=_env_bool(
            "LOCAL_RAG_QUERY_PREPROCESS_ENABLED",
            DEFAULT_RAG_QUERY_PREPROCESS_ENABLED,
        ),
        agent_temperature=_env_float("AGENT_TEMPERATURE", DEFAULT_AGENT_TEMPERATURE),
        agent_enable_thinking=_env_bool("AGENT_ENABLE_THINKING", DEFAULT_AGENT_ENABLE_THINKING),
        agent_reasoning_effort=os.getenv(
            "AGENT_REASONING_EFFORT", DEFAULT_AGENT_REASONING_EFFORT
        ).strip(),
        agent_team_max_members=_env_int("AGENT_TEAM_MAX_MEMBERS", DEFAULT_AGENT_TEAM_MAX_MEMBERS),
        agent_team_max_rounds=_env_int("AGENT_TEAM_MAX_ROUNDS", DEFAULT_AGENT_TEAM_MAX_ROUNDS),
        agent_team_members_hard_cap=_env_int(
            "AGENT_TEAM_MEMBERS_HARD_CAP",
            DEFAULT_AGENT_TEAM_MEMBERS_HARD_CAP,
        ),
        agent_team_rounds_hard_cap=_env_int(
            "AGENT_TEAM_ROUNDS_HARD_CAP",
            DEFAULT_AGENT_TEAM_ROUNDS_HARD_CAP,
        ),
        agent_routing_context_recent_limit=_env_int(
            "AGENT_ROUTING_CONTEXT_RECENT_LIMIT",
            DEFAULT_AGENT_ROUTING_CONTEXT_RECENT_LIMIT,
        ),
        agent_routing_context_max_chars=_env_int(
            "AGENT_ROUTING_CONTEXT_MAX_CHARS",
            DEFAULT_AGENT_ROUTING_CONTEXT_MAX_CHARS,
        ),
        agent_routing_context_item_max_chars=_env_int(
            "AGENT_ROUTING_CONTEXT_ITEM_MAX_CHARS",
            DEFAULT_AGENT_ROUTING_CONTEXT_ITEM_MAX_CHARS,
        ),
        agent_routing_context_reason_max_chars=_env_int(
            "AGENT_ROUTING_CONTEXT_REASON_MAX_CHARS",
            DEFAULT_AGENT_ROUTING_CONTEXT_REASON_MAX_CHARS,
        ),
        agent_routing_context_roles_preview_count=_env_int(
            "AGENT_ROUTING_CONTEXT_ROLES_PREVIEW_COUNT",
            DEFAULT_AGENT_ROUTING_CONTEXT_ROLES_PREVIEW_COUNT,
        ),
        agent_policy_text_len_medium=_env_int(
            "AGENT_POLICY_TEXT_LEN_MEDIUM",
            DEFAULT_AGENT_POLICY_TEXT_LEN_MEDIUM,
        ),
        agent_policy_text_len_high=_env_int(
            "AGENT_POLICY_TEXT_LEN_HIGH",
            DEFAULT_AGENT_POLICY_TEXT_LEN_HIGH,
        ),
        agent_policy_sentence_threshold=_env_int(
            "AGENT_POLICY_SENTENCE_THRESHOLD",
            DEFAULT_AGENT_POLICY_SENTENCE_THRESHOLD,
        ),
        agent_policy_comma_threshold=_env_int(
            "AGENT_POLICY_COMMA_THRESHOLD",
            DEFAULT_AGENT_POLICY_COMMA_THRESHOLD,
        ),
        agent_policy_question_threshold=_env_int(
            "AGENT_POLICY_QUESTION_THRESHOLD",
            DEFAULT_AGENT_POLICY_QUESTION_THRESHOLD,
        ),
        agent_policy_enum_steps_threshold=_env_int(
            "AGENT_POLICY_ENUM_STEPS_THRESHOLD",
            DEFAULT_AGENT_POLICY_ENUM_STEPS_THRESHOLD,
        ),
        agent_policy_conjunction_threshold=_env_int(
            "AGENT_POLICY_CONJUNCTION_THRESHOLD",
            DEFAULT_AGENT_POLICY_CONJUNCTION_THRESHOLD,
        ),
        agent_policy_context_chars_threshold=_env_int(
            "AGENT_POLICY_CONTEXT_CHARS_THRESHOLD",
            DEFAULT_AGENT_POLICY_CONTEXT_CHARS_THRESHOLD,
        ),
        agent_policy_context_lines_threshold=_env_int(
            "AGENT_POLICY_CONTEXT_LINES_THRESHOLD",
            DEFAULT_AGENT_POLICY_CONTEXT_LINES_THRESHOLD,
        ),
        agent_policy_score_plan=_env_int(
            "AGENT_POLICY_SCORE_PLAN",
            DEFAULT_AGENT_POLICY_SCORE_PLAN,
        ),
        agent_policy_score_team=_env_int(
            "AGENT_POLICY_SCORE_TEAM",
            DEFAULT_AGENT_POLICY_SCORE_TEAM,
        ),
        agent_planner_min_steps=_env_int(
            "AGENT_PLANNER_MIN_STEPS",
            DEFAULT_AGENT_PLANNER_MIN_STEPS,
        ),
        agent_planner_max_steps=_env_int(
            "AGENT_PLANNER_MAX_STEPS",
            DEFAULT_AGENT_PLANNER_MAX_STEPS,
        ),
        agent_policy_router_model_name=os.getenv(
            "AGENT_POLICY_ROUTER_MODEL_NAME",
            DEFAULT_AGENT_POLICY_ROUTER_MODEL_NAME,
        ).strip(),
        agent_policy_router_base_url=os.getenv(
            "AGENT_POLICY_ROUTER_BASE_URL",
            DEFAULT_AGENT_POLICY_ROUTER_BASE_URL,
        ).strip(),
        agent_policy_router_temperature=_env_float(
            "AGENT_POLICY_ROUTER_TEMPERATURE",
            DEFAULT_AGENT_POLICY_ROUTER_TEMPERATURE,
        ),
        agent_policy_async_enabled=_env_bool(
            "AGENT_POLICY_ASYNC_ENABLED",
            DEFAULT_AGENT_POLICY_ASYNC_ENABLED,
        ),
        agent_policy_async_refresh_seconds=max(
            0.5,
            _env_float(
                "AGENT_POLICY_ASYNC_REFRESH_SECONDS",
                DEFAULT_AGENT_POLICY_ASYNC_REFRESH_SECONDS,
            ),
        ),
        agent_policy_async_min_confidence=min(
            1.0,
            max(
                0.0,
                _env_float(
                    "AGENT_POLICY_ASYNC_MIN_CONFIDENCE",
                    DEFAULT_AGENT_POLICY_ASYNC_MIN_CONFIDENCE,
                ),
            ),
        ),
        agent_policy_async_max_staleness_seconds=max(
            1.0,
            _env_float(
                "AGENT_POLICY_ASYNC_MAX_STALENESS_SECONDS",
                DEFAULT_AGENT_POLICY_ASYNC_MAX_STALENESS_SECONDS,
            ),
        ),
        agent_llm_request_timeout=max(
            10.0,
            _env_float(
                "AGENT_LLM_REQUEST_TIMEOUT",
                DEFAULT_AGENT_LLM_REQUEST_TIMEOUT,
            ),
        ),
    )
