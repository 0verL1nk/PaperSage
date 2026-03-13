import re
from typing import Any, Dict, List, Tuple
import logging
import uuid
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._bm25 = None
        self._tool_names: List[str] = []
        self._is_index_dirty = True
        
        self._dense_retriever = None
        
    def register(self, tool_name: str, tool_obj: Any):
        self._tools[tool_name] = tool_obj
        self._is_index_dirty = True
        
    def get(self, tool_name: str) -> Any | None:
        return self._tools.get(tool_name)
        
    def get_all(self) -> Dict[str, Any]:
        return self._tools.copy()
        
    def _build_indexes(self):
        if not self._is_index_dirty:
            return
            
        self._tool_names = list(self._tools.keys())
        self._is_index_dirty = False
        
        # 1. Build BM25 Index
        try:
            import jieba
            from rank_bm25 import BM25Okapi
            
            tokenized_corpus = []
            for name in self._tool_names:
                tool = self._tools[name]
                desc = getattr(tool, "description", "") or ""
                if isinstance(tool, dict):
                    desc = tool.get("description", "")
                text = f"{name} {desc}"
                tokenized_corpus.append(list(jieba.cut(text.lower())))
                
            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            else:
                self._bm25 = None
        except ImportError:
            self._bm25 = None
            logger.debug("rank_bm25 or jieba not available. Falling back to regex search for tools.")

        # 2. Build Dense Index
        try:
            from agent.rag.vector_store import build_vectorstore
            from agent.settings import load_agent_settings
            settings = load_agent_settings()
            
            embeddings = FastEmbedEmbeddings(
                model_name=settings.local_embedding_model,
                cache_dir=settings.local_embedding_cache_dir,
            )
            
            texts = []
            metadatas = []
            for name in self._tool_names:
                tool = self._tools[name]
                desc = getattr(tool, "description", "") or ""
                if isinstance(tool, dict):
                    desc = tool.get("description", "")
                texts.append(f"Tool Name: {name}\nDescription: {desc}")
                metadatas.append({"tool_name": name})
                
            if texts:
                self._dense_retriever, _ = build_vectorstore(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    collection_prefix="tool_registry",
                    collection_key=f"tool_search_{uuid.uuid4().hex[:8]}"
                )
            else:
                self._dense_retriever = None
        except Exception as e:
            self._dense_retriever = None
            logger.debug(f"Dense retriever initialization failed: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """Hybrid search combining regex intersection, BM25, and Dense Vector search."""
        if not query or not query.strip():
            return []
            
        self._build_indexes()
            
        query_lower = query.lower()
        query_terms = set(re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", query_lower))
        
        scores: Dict[str, float] = {name: 0.0 for name in self._tool_names}
        
        # 1. Regex & Exact Match (Base scores)
        for name, tool in self._tools.items():
            description = getattr(tool, "description", "") or ""
            if isinstance(tool, dict):
                description = tool.get("description", "")
                
            text_to_search = f"{name} {description}".lower()
            tool_terms = set(re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text_to_search))
            
            # Intersection score
            scores[name] += float(len(query_terms.intersection(tool_terms)))
            
            # Exact match bonuses
            if query_lower in text_to_search:
                scores[name] += 5.0
            if query_lower == name.lower():
                scores[name] += 10.0
                
        # 2. BM25 Search
        if self._bm25 is not None:
            try:
                import jieba
                tokenized_query = list(jieba.cut(query_lower))
                bm25_scores = self._bm25.get_scores(tokenized_query)
                for idx, score in enumerate(bm25_scores):
                    scores[self._tool_names[idx]] += float(score) * 2.0  # Weight BM25
            except Exception as e:
                logger.warning(f"BM25 tool search failed: {e}")

        # 3. Dense Search
        if self._dense_retriever is not None:
            try:
                # Get more candidates than top_k to blend
                dense_results = self._dense_retriever.similarity_search_with_relevance_scores(query, k=top_k * 2)
                for doc, relevance in dense_results:
                    name = doc.metadata.get("tool_name")
                    if name in scores:
                        scores[name] += float(relevance) * 15.0  # High weight for dense semantic match
            except Exception as e:
                logger.warning(f"Dense tool search failed: {e}")
                
        # Format results
        results: List[Tuple[float, Any]] = []
        for name, tool in self._tools.items():
            if scores[name] > 0:
                results.append((scores[name], tool))
                
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [tool for score, tool in results[:top_k]]
