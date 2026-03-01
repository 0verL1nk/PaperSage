from dataclasses import dataclass
import os


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
DEFAULT_RAG_RERANK_ENABLED = True
DEFAULT_RAG_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RAG_HYBRID_ENABLED = True
DEFAULT_RAG_NEIGHBOR_EXPANSION = True
DEFAULT_RAG_NEIGHBOR_COUNT = 1
DEFAULT_RAG_QUERY_PREPROCESS_ENABLED = False
DEFAULT_AGENT_TEMPERATURE = 0.1
DEFAULT_AGENT_ENABLE_THINKING = False
DEFAULT_AGENT_REASONING_EFFORT = ""


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
    rag_rerank_model: str
    rag_hybrid_enabled: bool
    rag_neighbor_expansion: bool
    rag_neighbor_count: int
    rag_query_preprocess_enabled: bool
    agent_temperature: float
    agent_enable_thinking: bool
    agent_reasoning_effort: str


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
        rag_chunk_size=int(
            os.getenv("LOCAL_RAG_CHUNK_SIZE", str(DEFAULT_RAG_CHUNK_SIZE))
        ),
        rag_chunk_overlap=int(
            os.getenv("LOCAL_RAG_CHUNK_OVERLAP", str(DEFAULT_RAG_CHUNK_OVERLAP))
        ),
        rag_dense_candidate_k=int(
            os.getenv("LOCAL_RAG_DENSE_CANDIDATE_K", str(DEFAULT_RAG_DENSE_CANDIDATE_K))
        ),
        rag_sparse_candidate_k=int(
            os.getenv("LOCAL_RAG_SPARSE_CANDIDATE_K", str(DEFAULT_RAG_SPARSE_CANDIDATE_K))
        ),
        rag_rrf_candidate_k=int(
            os.getenv("LOCAL_RAG_RRF_CANDIDATE_K", str(DEFAULT_RAG_RRF_CANDIDATE_K))
        ),
        rag_rerank_candidate_k=int(
            os.getenv("LOCAL_RAG_RERANK_CANDIDATE_K", str(DEFAULT_RAG_RERANK_CANDIDATE_K))
        ),
        rag_top_k=int(os.getenv("LOCAL_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
        rag_rerank_enabled=os.getenv(
            "LOCAL_RAG_RERANK_ENABLED", str(DEFAULT_RAG_RERANK_ENABLED)
        ).lower()
        in {"1", "true", "yes", "on"},
        rag_rerank_model=os.getenv(
            "LOCAL_RAG_RERANK_MODEL", DEFAULT_RAG_RERANK_MODEL
        ),
        rag_hybrid_enabled=os.getenv(
            "LOCAL_RAG_HYBRID_ENABLED", str(DEFAULT_RAG_HYBRID_ENABLED)
        ).lower()
        in {"1", "true", "yes", "on"},
        rag_neighbor_expansion=os.getenv(
            "LOCAL_RAG_NEIGHBOR_EXPANSION", str(DEFAULT_RAG_NEIGHBOR_EXPANSION)
        ).lower()
        in {"1", "true", "yes", "on"},
        rag_neighbor_count=int(
            os.getenv("LOCAL_RAG_NEIGHBOR_COUNT", str(DEFAULT_RAG_NEIGHBOR_COUNT))
        ),
        rag_query_preprocess_enabled=os.getenv(
            "LOCAL_RAG_QUERY_PREPROCESS_ENABLED", str(DEFAULT_RAG_QUERY_PREPROCESS_ENABLED)
        ).lower()
        in {"1", "true", "yes", "on"},
        agent_temperature=float(
            os.getenv("AGENT_TEMPERATURE", str(DEFAULT_AGENT_TEMPERATURE))
        ),
        agent_enable_thinking=os.getenv(
            "AGENT_ENABLE_THINKING", str(DEFAULT_AGENT_ENABLE_THINKING)
        ).lower()
        in {"1", "true", "yes", "on"},
        agent_reasoning_effort=os.getenv(
            "AGENT_REASONING_EFFORT", DEFAULT_AGENT_REASONING_EFFORT
        ).strip(),
    )
