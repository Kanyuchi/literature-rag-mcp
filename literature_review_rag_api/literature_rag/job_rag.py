"""Job Collection RAG Wrapper

Wraps a job's ChromaDB collection to provide the same interface
as LiteratureReviewRAG, enabling use with the agentic pipeline.

Supports OpenAI (API) embeddings only.
"""

import logging
from typing import Dict, List, Optional, Any

import chromadb
from langchain_core.embeddings import Embeddings

from .config import load_config
from .database import get_db_session, KnowledgeEntityOccurrenceCRUD, KnowledgeEdgeCRUD, KnowledgeEntityCRUD
from .embeddings import get_embeddings, get_embedding_info

logger = logging.getLogger(__name__)


class JobCollectionRAG:
    """
    RAG wrapper for job-specific ChromaDB collections.

    Provides the same interface as LiteratureReviewRAG so it can be
    used with the agentic pipeline.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_model: str = None,
        term_maps: Optional[Dict[str, List[List[str]]]] = None,
        job_id: Optional[int] = None
    ):
        """
        Initialize the job collection RAG.

        Args:
            collection: ChromaDB collection for the job
            embedding_model: Embedding model name (default from config)
            term_maps: Optional term normalization maps for query expansion
        """
        self.collection = collection
        self.config = load_config()
        self.job_id = job_id
        self.term_maps = term_maps or {}
        self.normalization_enabled = bool(term_maps)

        # Initialize embeddings using unified interface
        self.embeddings = get_embeddings(self.config.embedding)

        # Store embedding info for stats
        self._embedding_info = get_embedding_info(self.embeddings)

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        self._reranker = None
        self._reranker_config = {
            "enabled": self.config.retrieval.use_reranking,
            "model": self.config.retrieval.reranker_model,
            "rerank_top_k": self.config.retrieval.rerank_top_k,
            "device": device
        }
        graph_cfg = getattr(self.config, "graph", None)
        self._graph_config = {
            "enabled": getattr(graph_cfg, "enable_graph_rag", True),
            "query_expansion_enabled": getattr(graph_cfg, "query_expansion_enabled", True),
            "query_expansion_max_terms": getattr(graph_cfg, "query_expansion_max_terms", 12),
            "query_expansion_max_hops": getattr(graph_cfg, "query_expansion_max_hops", 1),
            "query_expansion_seed_entities": getattr(graph_cfg, "query_expansion_seed_entities", 30),
        }

        logger.info(f"JobCollectionRAG initialized for collection: {collection.name} (embedding: {self._embedding_info['provider']})")

    def _load_term_maps(self, yaml_term_maps: dict) -> Dict[str, Dict[str, List[str]]]:
        term_maps = {}
        for category, term_groups in yaml_term_maps.items():
            term_maps[category] = {}
            for term_group in term_groups:
                if len(term_group) > 0:
                    canonical = term_group[0]
                    variants = term_group[1:] if len(term_group) > 1 else []
                    term_maps[category][canonical] = variants
        return term_maps

    def normalize_query(self, question: str) -> str:
        if not self.normalization_enabled or not self.term_maps:
            return question
        detected_terms = []
        query_lower = question.lower()
        expansion_terms = []
        max_expansions = self.config.retrieval.max_expansions

        term_maps = self._load_term_maps(self.term_maps)
        for _, term_dict in term_maps.items():
            for canonical, variants in term_dict.items():
                if canonical.lower() in query_lower:
                    detected_terms.append(canonical)
                    expansion_terms.extend(variants[:max_expansions])
                    continue
                for variant in variants:
                    if variant.lower() in query_lower:
                        detected_terms.append(canonical)
                        expansion_terms.append(canonical)
                        expansion_terms.extend([v for v in variants[:max_expansions] if v != variant])
                        break

        if expansion_terms and self.config.retrieval.expand_queries:
            return question + " " + " ".join(expansion_terms)
        return question

    def _graph_expand_terms(self, seed_doc_ids: List[str]) -> List[str]:
        if not self._graph_config.get("enabled") or not self._graph_config.get("query_expansion_enabled"):
            return []
        if not self.job_id or not seed_doc_ids:
            return []

        db = get_db_session()
        try:
            entity_ids = KnowledgeEntityOccurrenceCRUD.list_entity_ids_for_docs(
                db,
                self.job_id,
                seed_doc_ids,
                limit=self._graph_config.get("query_expansion_seed_entities", 30)
            )
            if not entity_ids:
                return []

            # Traverse graph edges for neighbor entities
            current_ids = set(entity_ids)
            hops = max(0, int(self._graph_config.get("query_expansion_max_hops", 1)))
            for _ in range(hops):
                edges = db.query(KnowledgeEdgeCRUD.model).filter(
                    KnowledgeEdgeCRUD.model.job_id == self.job_id,
                    (KnowledgeEdgeCRUD.model.source_entity_id.in_(current_ids)
                     | KnowledgeEdgeCRUD.model.target_entity_id.in_(current_ids))
                ).all()
                for edge in edges:
                    current_ids.add(edge.source_entity_id)
                    current_ids.add(edge.target_entity_id)

            names = db.query(KnowledgeEntityCRUD.model.name).filter(
                KnowledgeEntityCRUD.model.job_id == self.job_id,
                KnowledgeEntityCRUD.model.id.in_(list(current_ids))
            ).limit(self._graph_config.get("query_expansion_max_terms", 12)).all()
            return [row[0] for row in names if row and row[0]]
        finally:
            db.close()

    def _get_reranker(self):
        if not self._reranker_config.get("enabled"):
            return None
        if self._reranker is None:
            from .reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model_name=self._reranker_config.get("model"),
                device=self._reranker_config.get("device")
            )
        return self._reranker

    def is_ready(self) -> bool:
        """Check if the collection is ready."""
        return self.collection is not None and self.collection.count() > 0

    def query(
        self,
        question: str,
        n_results: int = 5,
        phase_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the job collection.

        Returns results in the same format as LiteratureReviewRAG.query()
        """
        expanded_query = self.normalize_query(question)
        # Embed query
        query_embedding = self.embeddings.embed_query(expanded_query)

        # Build filters
        from .language import detect_language

        base_conditions = []
        if phase_filter:
            base_conditions.append({"phase": phase_filter})
        if topic_filter:
            base_conditions.append({"topic_category": topic_filter})

        language_filter_applied = False
        language = None
        if self.config.retrieval.language_filter_enabled:
            language = detect_language(question)
            if language and language != "unknown":
                language_filter_applied = True
                base_conditions.append({"language": language})

        where_filter = None
        if len(base_conditions) == 1:
            where_filter = base_conditions[0]
        elif len(base_conditions) > 1:
            where_filter = {"$and": base_conditions}

        # Query collection
        rerank_enabled = self._reranker_config.get("enabled") if use_reranking is None else use_reranking
        rerank_top_k = self._reranker_config.get("rerank_top_k", n_results)
        candidate_k = max(n_results, rerank_top_k * 2) if rerank_enabled else max(n_results * 3, 10)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        if (
            language_filter_applied
            and self.config.retrieval.language_filter_fallback
            and (not results or not results.get("documents") or not results["documents"][0])
        ):
            # Retry without language filter for legacy docs or mixed-language KBs
            fallback_conditions = []
            if phase_filter:
                fallback_conditions.append({"phase": phase_filter})
            if topic_filter:
                fallback_conditions.append({"topic_category": topic_filter})
            fallback_filter = None
            if len(fallback_conditions) == 1:
                fallback_filter = fallback_conditions[0]
            elif len(fallback_conditions) > 1:
                fallback_filter = {"$and": fallback_conditions}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where=fallback_filter,
                include=["documents", "metadatas", "distances"]
            )

        results = self._postprocess_results(results, n_results)

        # GraphRAG-style query expansion using related entities
        if self._graph_config.get("enabled") and self._graph_config.get("query_expansion_enabled"):
            try:
                doc_ids = [
                    meta.get("doc_id")
                    for meta in (results.get("metadatas", [[]])[0] or [])
                    if meta and meta.get("doc_id")
                ]
                expansion_terms = self._graph_expand_terms(doc_ids)
                if expansion_terms:
                    expanded_query_graph = expanded_query + " " + " ".join(expansion_terms)
                    graph_results = self.collection.query(
                        query_embeddings=[self.embeddings.embed_query(expanded_query_graph)],
                        n_results=candidate_k,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"]
                    )
                    results = self._merge_results(results, graph_results, n_results)
            except Exception as e:
                logger.warning(f"Graph expansion skipped due to error: {e}")

        # Rerank if enabled
        reranker = self._get_reranker() if rerank_enabled else None
        if reranker and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            scores = reranker.score(question, docs)
            ranked = list(zip(scores, docs, metas, dists))
            ranked.sort(key=lambda x: x[0], reverse=True)
            ranked = ranked[:n_results]
            return {
                "documents": [[item[1] for item in ranked]],
                "metadatas": [[item[2] for item in ranked]],
                "distances": [[item[3] for item in ranked]]
            }

        return results

    def _postprocess_results(self, results: dict, n_results: int) -> dict:
        """Dedupe results by doc_id and apply a light quality-aware rerank."""
        if not results or not results.get("documents") or not results["documents"][0]:
            return results

        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = 1 - dist
            total_pages = meta.get("total_pages") or 0
            try:
                total_pages = int(total_pages)
            except (TypeError, ValueError):
                total_pages = 0
            if total_pages and total_pages < 5:
                score *= 0.7
            if meta.get("doi"):
                score *= 1.05
            items.append((score, doc, meta, dist, meta.get("doc_id")))

        items.sort(key=lambda item: item[0], reverse=True)

        deduped_docs = []
        deduped_metas = []
        deduped_dists = []
        seen_doc_ids = set()

        for _, doc, meta, dist, doc_id in items:
            if doc_id and doc_id in seen_doc_ids:
                continue
            if doc_id:
                seen_doc_ids.add(doc_id)
            deduped_docs.append(doc)
            deduped_metas.append(meta)
            deduped_dists.append(dist)
            if len(deduped_docs) >= n_results:
                break

        return {
            "documents": [deduped_docs],
            "metadatas": [deduped_metas],
            "distances": [deduped_dists],
            "ids": results.get("ids", [[]])
        }

    def _merge_results(self, primary: dict, secondary: dict, n_results: int) -> dict:
        """Merge two result sets then dedupe by doc_id."""
        if not secondary or not secondary.get("documents") or not secondary["documents"][0]:
            return primary
        if not primary or not primary.get("documents") or not primary["documents"][0]:
            return self._postprocess_results(secondary, n_results)

        combined = {
            "documents": [primary["documents"][0] + secondary["documents"][0]],
            "metadatas": [primary["metadatas"][0] + secondary["metadatas"][0]],
            "distances": [primary["distances"][0] + secondary["distances"][0]],
            "ids": [primary.get("ids", [[]])[0] + secondary.get("ids", [[]])[0]]
        }
        return self._postprocess_results(combined, n_results)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns stats in the same format as LiteratureReviewRAG.get_stats()
        """
        try:
            # Get all metadata to compute stats
            all_data = self.collection.get(include=["metadatas"])

            papers = {}
            phases = {}
            topics = {}

            for metadata in all_data.get("metadatas", []):
                doc_id = metadata.get("doc_id")
                if doc_id and doc_id not in papers:
                    papers[doc_id] = metadata

                phase = metadata.get("phase")
                if phase:
                    phases[phase] = phases.get(phase, 0) + 1

                topic = metadata.get("topic_category")
                if topic:
                    topics[topic] = topics.get(topic, 0) + 1

            return {
                "total_chunks": self.collection.count(),
                "total_papers": len(papers),
                "papers_by_phase": phases,
                "papers_by_topic": topics,
                "embedding_provider": self._embedding_info.get("provider", "unknown"),
                "embedding_model": self._embedding_info.get("model", "unknown"),
                "embedding_dimension": self._embedding_info.get("dimension", 768)
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "papers_by_phase": {},
                "papers_by_topic": {},
            }

    def get_context(
        self,
        question: str,
        n_results: int = 5,
        **kwargs
    ) -> str:
        """Get formatted context string for LLM."""
        results = self.query(question, n_results, **kwargs)

        context_parts = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]

            authors = metadata.get("authors", "Unknown")
            year = metadata.get("year", "n.d.")
            title = metadata.get("title", "Untitled")

            citation = f"[{i+1}] {authors} ({year}) - {title}"
            context_parts.append(f"{citation}\n{content}")

        return "\n\n---\n\n".join(context_parts)
