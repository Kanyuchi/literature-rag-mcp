"""Literature Review RAG System - Simple, Reliable, Accurate

Adapted from personality RAG system (100% MBTI accuracy, 15ms queries).
Key innovation: Explicit academic term normalization for improved search relevance.

Supports OpenAI (API) embeddings only.
"""

import chromadb
from langchain_core.embeddings import Embeddings
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from .pool import get_pool, get_pooled_embeddings, get_pooled_chroma_client
from .embeddings import get_embedding_dimension, get_embedding_info, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)


class LiteratureReviewRAG:
    """RAG system for academic literature review with term normalization."""

    def __init__(
        self,
        chroma_path: str,
        config: dict = None,
        use_pool: bool = True,
        openai_model: str = None
    ):
        """
        Initialize Literature Review RAG system.

        Args:
            chroma_path: Path to ChromaDB persistence directory
            config: Configuration dictionary from literature_config.yaml
            use_pool: Whether to use connection pooling (default: True)
            openai_model: OpenAI model name (default: text-embedding-3-small)
        """
        self.config = config or {}
        self.normalization_enabled = self.config.get("normalization_enable", True)
        self._use_pool = use_pool

        # Embedding provider selection (OpenAI only)
        provider = "openai"
        openai_model = openai_model or self.config.get("openai_model", "text-embedding-3-small")

        logger.info(f"Initializing Literature Review RAG (pooled: {use_pool}, provider: {provider})")

        # Initialize embeddings - OpenAI only
        if use_pool:
            self.embeddings = get_pooled_embeddings(
                openai_model=openai_model
            )
            self.client = get_pooled_chroma_client(chroma_path)
        else:
            from .embeddings import get_embeddings
            from .config import EmbeddingConfig
            embed_config = EmbeddingConfig(
                provider="openai",
                openai_model=openai_model
            )
            self.embeddings = get_embeddings(embed_config)
            self.client = chromadb.PersistentClient(path=chroma_path)

        # Store embedding info for stats
        self._embedding_info = get_embedding_info(self.embeddings)

        # Get or create collection
        collection_name = self.config.get("collection_name", "literature_review_chunks")
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            logger.warning(f"Collection {collection_name} not found. Will be created during indexing.")
            self.collection = None

        # Load academic term normalization maps (optional)
        term_maps_config = self.config.get("term_maps", {}) if self.normalization_enabled else {}
        self.term_maps = self._load_term_maps(term_maps_config)
        try:
            import torch
            rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            rerank_device = "cpu"

        self._reranker = None
        self._reranker_config = {
            "enabled": self.config.get("use_reranking", False),
            "model": self.config.get("reranker_model", "BAAI/bge-reranker-base"),
            "rerank_top_k": self.config.get("rerank_top_k", 20),
            "device": rerank_device
        }

        # Initialize hybrid search components (lazy-loaded)
        self._bm25_retriever = None
        self._hybrid_scorer = None
        self._hybrid_config = {
            "enabled": self.config.get("use_hybrid", False),
            "method": self.config.get("hybrid_method", "rrf"),
            "dense_weight": self.config.get("hybrid_weight", 0.7),
            "bm25_candidates": self.config.get("bm25_candidates", 50),
            "bm25_use_stemming": self.config.get("bm25_use_stemming", True),
            "bm25_min_token_length": self.config.get("bm25_min_token_length", 2),
            "indices_path": self.config.get("indices_path", "./indices")
        }

        logger.info(f"Literature Review RAG initialized successfully (hybrid: {self._hybrid_config['enabled']})")

    def _get_reranker(self):
        """Lazy-load the reranker when enabled."""
        if not self._reranker_config.get("enabled"):
            return None
        if self._reranker is None:
            from .reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model_name=self._reranker_config.get("model"),
                device=self._reranker_config.get("device")
            )
        return self._reranker

    def _get_bm25_retriever(self):
        """Lazy-load the BM25 retriever when hybrid search is enabled."""
        if not self._hybrid_config.get("enabled"):
            return None
        if self._bm25_retriever is None:
            from .bm25_retriever import BM25Retriever, BM25Config
            from pathlib import Path

            indices_path = Path(self._hybrid_config.get("indices_path", "./indices"))
            bm25_config = BM25Config(
                index_path=str(indices_path / "bm25_index.pkl"),
                use_stemming=self._hybrid_config.get("bm25_use_stemming", True),
                min_token_length=self._hybrid_config.get("bm25_min_token_length", 2)
            )
            self._bm25_retriever = BM25Retriever(bm25_config)

            # Try to load existing index
            if not self._bm25_retriever.load_index():
                logger.warning("BM25 index not found. Run build_index.py to create it.")

        return self._bm25_retriever

    def _get_hybrid_scorer(self):
        """Lazy-load the hybrid scorer when hybrid search is enabled."""
        if not self._hybrid_config.get("enabled"):
            return None
        if self._hybrid_scorer is None:
            from .bm25_retriever import HybridScorer
            self._hybrid_scorer = HybridScorer(
                method=self._hybrid_config.get("method", "rrf"),
                dense_weight=self._hybrid_config.get("dense_weight", 0.7)
            )
        return self._hybrid_scorer

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata fields for consistency."""
        if not metadata:
            return metadata

        normalized = metadata.copy()
        authors = normalized.get("authors")
        if isinstance(authors, list):
            normalized["authors"] = ", ".join(a for a in authors if a)
        elif authors is not None:
            normalized["authors"] = str(authors).strip()

        year = normalized.get("year")
        try:
            if year is not None:
                normalized["year"] = int(year)
        except (TypeError, ValueError):
            normalized["year"] = None

        doi = normalized.get("doi")
        if doi:
            normalized["doi"] = str(doi).strip().lower().replace("https://doi.org/", "")

        title = normalized.get("title")
        if title:
            normalized["title"] = " ".join(str(title).split())

        return normalized

    def _rerank_results(self, question: str, results: dict, n_results: int) -> dict:
        """Rerank results using a cross-encoder."""
        reranker = self._get_reranker()
        if reranker is None or not results or not results.get("documents"):
            return results

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        if not docs:
            return results

        scores = reranker.score(question, docs)
        ranked = list(zip(scores, docs, metas, dists))
        ranked.sort(key=lambda x: x[0], reverse=True)

        top_ranked = ranked[:n_results]
        return {
            "documents": [[item[1] for item in top_ranked]],
            "metadatas": [[item[2] for item in top_ranked]],
            "distances": [[item[3] for item in top_ranked]],
            "ids": results.get("ids", [[]])
        }

    def _load_term_maps(self, yaml_term_maps: dict) -> Dict[str, Dict[str, List[str]]]:
        """
        Load academic term normalization maps from config.

        Converts from config format:
            {category: [[canonical, variant1, variant2], [...]]}
        To internal format:
            {category: {canonical: [variants], ...}}
        """
        term_maps = {}

        for category, term_groups in yaml_term_maps.items():
            term_maps[category] = {}

            for term_group in term_groups:
                if len(term_group) > 0:
                    canonical = term_group[0]  # First term is canonical
                    variants = term_group[1:] if len(term_group) > 1 else []
                    term_maps[category][canonical] = variants

        logger.info(f"Loaded {len(term_maps)} term map categories with {sum(len(v) for v in term_maps.values())} term groups")
        return term_maps

    def normalize_query(self, question: str) -> Tuple[str, List[str]]:
        """
        Detect and expand academic terms in query (like MBTI normalization).

        This is the KEY PATTERN from personality RAG that achieved 100% accuracy.

        Example:
            "Ruhrgebiet transformation" →
                normalized: "Ruhrgebiet ruhr valley ruhr region transformation"
                detected: ["ruhrgebiet"]

        Args:
            question: User query

        Returns:
            (expanded_query, detected_terms) tuple
        """
        detected_terms = []
        if not self.normalization_enabled:
            return question, detected_terms
        query_lower = question.lower()
        expansion_terms = []
        max_expansions = self.config.get("max_expansions", 2)

        # Iterate through all term maps
        for category, term_dict in self.term_maps.items():
            for canonical, variants in term_dict.items():
                # Check if canonical term appears in query
                if canonical.lower() in query_lower:
                    detected_terms.append(canonical)
                    # Add top variants as expansion terms
                    expansion_terms.extend(variants[:max_expansions])
                    continue  # Found canonical, skip checking variants

                # Check if any variant appears
                for variant in variants:
                    if variant.lower() in query_lower:
                        detected_terms.append(canonical)
                        # Add canonical and other top variants
                        expansion_terms.append(canonical)
                        expansion_terms.extend([v for v in variants[:max_expansions] if v != variant])
                        break

        # Build expanded query
        if expansion_terms and self.config.get("expand_queries", True):
            # Add expansion terms to query
            expanded_query = question + " " + " ".join(expansion_terms)
            logger.debug(f"Query expanded: {question} → {expanded_query}")
        else:
            expanded_query = question

        return expanded_query, detected_terms

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
            meta = self._normalize_metadata(meta)
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

    def query(
        self,
        question: str,
        n_results: int = 5,
        phase_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        methodology_filter: Optional[str] = None,
        geographic_filter: Optional[str] = None,
        research_type_filter: Optional[str] = None,
        **kwargs
    ) -> dict:
        """
        Query academic literature with automatic term expansion and rich filtering.

        Args:
            question: Search query
            n_results: Number of results to return
            phase_filter: Filter by phase (e.g., "Phase 1")
            topic_filter: Filter by topic category (e.g., "Business Formation")
            year_min: Minimum publication year
            year_max: Maximum publication year
            methodology_filter: Filter by methodology
            geographic_filter: Filter by geographic focus
            research_type_filter: Filter by research type (empirical, theoretical, etc.)

        Returns:
            ChromaDB query results dict with documents, metadatas, distances
        """
        if not self.collection:
            logger.error("No collection loaded. Run build_index.py first.")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        # Normalize query (KEY PATTERN - like MBTI normalization)
        expanded_query, detected_terms = self.normalize_query(question)

        if detected_terms:
            logger.info(f"Detected academic terms: {detected_terms}")

        # Build ChromaDB filters
        conditions = []

        if phase_filter:
            conditions.append({"phase": phase_filter})

        if topic_filter:
            conditions.append({"topic_category": topic_filter})

        if year_min or year_max:
            year_conditions = {}
            if year_min:
                year_conditions["$gte"] = year_min
            if year_max:
                year_conditions["$lte"] = year_max
            if year_conditions:
                conditions.append({"year": year_conditions})

        if methodology_filter:
            conditions.append({"methodology": methodology_filter})

        if geographic_filter:
            # Check if geographic filter matches any in the list
            conditions.append({"geographic_focus": {"$contains": geographic_filter}})

        if research_type_filter:
            conditions.append({"research_type": research_type_filter})

        # Combine filters
        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Use hybrid search if enabled and BM25 index is ready
        bm25_retriever = self._get_bm25_retriever()
        if self._hybrid_config.get("enabled") and bm25_retriever and bm25_retriever.is_ready():
            return self._hybrid_query(expanded_query, n_results, where_filter)

        # Fall back to dense-only search
        return self._dense_query(expanded_query, n_results, where_filter)

    def _dense_query(
        self,
        query: str,
        n_results: int,
        where_filter: Optional[dict]
    ) -> dict:
        """Execute dense-only (embedding) query."""
        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Determine candidate size for reranking
        rerank_top_k = self._reranker_config.get("rerank_top_k", n_results)
        candidate_k = max(n_results, rerank_top_k) if self._reranker_config.get("enabled") else n_results

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            logger.debug(f"Dense query returned {len(results['documents'][0])} results")
            results = self._rerank_results(query, results, n_results)
            return self._postprocess_results(results, n_results)

        except Exception as e:
            logger.error(f"Dense query failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    def _hybrid_query(
        self,
        query: str,
        n_results: int,
        where_filter: Optional[dict]
    ) -> dict:
        """
        Execute hybrid BM25 + dense query with score fusion.

        Combines sparse keyword matching (BM25) with dense semantic search
        for improved retrieval quality, especially for author names and
        specific technical terms.
        """
        try:
            bm25_retriever = self._get_bm25_retriever()
            hybrid_scorer = self._get_hybrid_scorer()

            if not bm25_retriever or not hybrid_scorer:
                logger.warning("Hybrid components not available, falling back to dense search")
                return self._dense_query(query, n_results, where_filter)

            # 1. Get BM25 candidates
            bm25_candidates = self._hybrid_config.get("bm25_candidates", 50)
            bm25_results = bm25_retriever.query(query, n_results=bm25_candidates)
            logger.debug(f"BM25 returned {len(bm25_results)} candidates")

            # 2. Get dense candidates (more than needed for fusion)
            query_embedding = self.embeddings.embed_query(query)
            rerank_top_k = self._reranker_config.get("rerank_top_k", n_results)
            dense_k = max(bm25_candidates, rerank_top_k * 2, n_results * 3)

            dense_results_raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=dense_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Convert dense results to (chunk_id, distance) format
            dense_results = []
            if dense_results_raw and dense_results_raw.get("metadatas"):
                for i, (meta, dist) in enumerate(zip(
                    dense_results_raw["metadatas"][0],
                    dense_results_raw["distances"][0]
                )):
                    chunk_id = meta.get(
                        "chunk_id",
                        f"{meta.get('doc_id', 'unknown')}_chunk_{meta.get('chunk_index', i)}"
                    )
                    dense_results.append((chunk_id, dist))

            logger.debug(f"Dense returned {len(dense_results)} candidates")

            # 3. Combine scores with RRF or weighted fusion
            fusion_k = max(n_results * 2, rerank_top_k)
            hybrid_results = hybrid_scorer.combine_scores(
                bm25_results, dense_results, n_results=fusion_k
            )
            logger.debug(f"Hybrid fusion produced {len(hybrid_results)} results")

            # 4. Fetch full results from ChromaDB by chunk IDs
            hybrid_chunk_ids = [chunk_id for chunk_id, _ in hybrid_results]

            # Build a lookup for scores
            score_lookup = {chunk_id: score for chunk_id, score in hybrid_results}

            # Fetch documents and metadata for hybrid results
            # We need to query by IDs, but ChromaDB .get() doesn't support where filters
            # So we filter manually after fetching
            all_hybrid_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            if hybrid_chunk_ids:
                try:
                    fetched = self.collection.get(
                        ids=hybrid_chunk_ids,
                        include=["documents", "metadatas"]
                    )

                    if fetched and fetched.get("ids"):
                        # Build results in the order of hybrid_chunk_ids (preserving fusion order)
                        id_to_idx = {id_: i for i, id_ in enumerate(fetched["ids"])}

                        for chunk_id in hybrid_chunk_ids:
                            if chunk_id in id_to_idx:
                                idx = id_to_idx[chunk_id]
                                meta = fetched["metadatas"][idx]

                                # Apply where_filter manually if present
                                if where_filter and not self._matches_filter(meta, where_filter):
                                    continue

                                all_hybrid_results["documents"][0].append(fetched["documents"][idx])
                                all_hybrid_results["metadatas"][0].append(meta)
                                # Use inverse of hybrid score as "distance" for consistency
                                all_hybrid_results["distances"][0].append(1.0 - score_lookup.get(chunk_id, 0.5))

                except Exception as e:
                    logger.warning(f"Failed to fetch hybrid results by ID: {e}")
                    # Fall back to dense-only
                    return self._dense_query(query, n_results, where_filter)

            # 5. Apply reranking if enabled
            results = self._rerank_results(query, all_hybrid_results, n_results)

            # 6. Postprocess and return
            return self._postprocess_results(results, n_results)

        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            # Fall back to dense-only search
            return self._dense_query(query, n_results, where_filter)

    def _matches_filter(self, metadata: dict, where_filter: dict) -> bool:
        """Check if metadata matches a ChromaDB-style where filter."""
        if not where_filter:
            return True

        # Handle $and
        if "$and" in where_filter:
            return all(self._matches_filter(metadata, cond) for cond in where_filter["$and"])

        # Handle $or
        if "$or" in where_filter:
            return any(self._matches_filter(metadata, cond) for cond in where_filter["$or"])

        # Handle simple key-value or operator conditions
        for key, condition in where_filter.items():
            if key.startswith("$"):
                continue  # Skip operators at top level (handled above)

            value = metadata.get(key)

            if isinstance(condition, dict):
                # Operator condition like {"$gte": 2000}
                for op, op_val in condition.items():
                    if op == "$gte" and (value is None or value < op_val):
                        return False
                    if op == "$lte" and (value is None or value > op_val):
                        return False
                    if op == "$gt" and (value is None or value <= op_val):
                        return False
                    if op == "$lt" and (value is None or value >= op_val):
                        return False
                    if op == "$eq" and value != op_val:
                        return False
                    if op == "$ne" and value == op_val:
                        return False
                    if op == "$contains":
                        if value is None:
                            return False
                        if isinstance(value, str) and op_val not in value:
                            return False
            else:
                # Simple equality
                if value != condition:
                    return False

        return True

    def get_context(self, question: str, **filters) -> str:
        """
        Get formatted context for LLM prompting with citations.

        Args:
            question: Search query
            **filters: Filter parameters (phase_filter, topic_filter, etc.)

        Returns:
            Formatted string with citations and abstracts
        """
        results = self.query(question, **filters)

        if not results["documents"][0]:
            return "No relevant papers found."

        context_parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            # Format citation
            title = meta.get("title", "Unknown Title")
            authors = meta.get("authors", ["Unknown"])
            if isinstance(authors, list):
                author_str = ", ".join(authors[:3])  # First 3 authors
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = str(authors)

            year = meta.get("year", "n.d.")
            phase = meta.get("phase", "")
            topic = meta.get("topic_category", "")

            # Format context entry
            citation = f"[{author_str} ({year}). {title}]"
            metadata_line = f"Phase: {phase} | Topic: {topic}"

            context_parts.append(
                f"{citation}\n{metadata_line}\n\n{doc}"
            )

        return "\n\n---\n\n".join(context_parts)

    def synthesis_query(
        self,
        question: str,
        topics: List[str],
        n_per_topic: int = 2
    ) -> Dict[str, str]:
        """
        Query multiple topic categories and synthesize results (like council_query).

        Args:
            question: Search query
            topics: List of topic categories to query
            n_per_topic: Number of results per topic

        Returns:
            Dictionary mapping topic → context string
        """
        results = {}

        for topic in topics:
            logger.info(f"Querying topic: {topic}")
            context = self.get_context(
                question,
                n_results=n_per_topic,
                topic_filter=topic
            )
            results[topic] = context

        return results

    def find_related_papers(
        self,
        paper_id: str,
        n_results: int = 5
    ) -> dict:
        """
        Find papers similar to a given paper via embedding similarity.

        Args:
            paper_id: Document ID of the paper
            n_results: Number of similar papers to return

        Returns:
            Query results dict
        """
        if not self.collection:
            logger.error("No collection loaded")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            # Get the paper's embedding using doc_id first
            paper_result = None
            try:
                paper_result = self.collection.get(
                    where={"doc_id": paper_id},
                    include=["embeddings"]
                )
            except Exception:
                paper_result = self.collection.get(
                    ids=[paper_id],
                    include=["embeddings"]
                )

            if not paper_result or not paper_result.get("embeddings"):
                logger.warning(f"Paper {paper_id} not found")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            paper_embedding = paper_result["embeddings"][0]

            # Query for similar papers (excluding the original)
            results = self.collection.query(
                query_embeddings=[paper_embedding],
                n_results=n_results + 1,  # +1 because original will be in results
                include=["documents", "metadatas", "distances"]
            )

            # Remove the original paper from results
            filtered_results = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                if meta.get("doc_id") != paper_id:
                    filtered_results["documents"][0].append(doc)
                    filtered_results["metadatas"][0].append(meta)
                    filtered_results["distances"][0].append(dist)

                if len(filtered_results["documents"][0]) >= n_results:
                    break

            return self._postprocess_results(filtered_results, n_results)

        except Exception as e:
            logger.error(f"Error finding related papers: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        # Get embedding info
        embed_info = getattr(self, '_embedding_info', None) or get_embedding_info(self.embeddings)

        def _stats_from_cache() -> Optional[Dict[str, Any]]:
            if not self.config.get("cache_metadata"):
                return None
            cache_path = self.config.get("metadata_cache_path")
            if not cache_path:
                return None
            try:
                from pathlib import Path
                import pickle

                cache_file = Path(cache_path)
                if not cache_file.exists():
                    return None
                with cache_file.open("rb") as f:
                    cache = pickle.load(f)

                papers = cache.get("papers", [])
                papers_by_phase: Dict[str, int] = {}
                papers_by_topic: Dict[str, int] = {}
                years: List[int] = []

                for paper in papers:
                    phase = paper.get("phase") or "Unknown"
                    papers_by_phase[phase] = papers_by_phase.get(phase, 0) + 1

                    topic = paper.get("topic_category") or "Unknown"
                    papers_by_topic[topic] = papers_by_topic.get(topic, 0) + 1

                    year = paper.get("year")
                    if isinstance(year, int) and year > 1900:
                        years.append(year)

                year_range = {
                    "min": min(years) if years else 0,
                    "max": max(years) if years else 0,
                }

                return {
                    "total_chunks": cache.get("total_chunks", 0),
                    "total_papers": cache.get("total_papers", len(papers)),
                    "papers_by_phase": papers_by_phase,
                    "papers_by_topic": papers_by_topic,
                    "year_range": year_range,
                }
            except Exception:
                return None

        if not self.collection:
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "papers_by_phase": {},
                "papers_by_topic": {},
                "year_range": {"min": 0, "max": 0},
                "embedding_provider": embed_info.get("provider", "unknown"),
                "embedding_model": embed_info.get("model", "unknown"),
                "embedding_dimension": embed_info.get("dimension", 768),
                "ready": False
            }

        try:
            # Prefer cached metadata for fast stats if available.
            cached = _stats_from_cache()
            if cached is not None:
                return {
                    **cached,
                    "embedding_provider": embed_info.get("provider", "unknown"),
                    "embedding_model": embed_info.get("model", "unknown"),
                    "embedding_dimension": embed_info.get("dimension", 768),
                    "ready": True,
                }

            # Get total count
            count = self.collection.count()

            # Avoid expensive full scans for very large collections when cache is missing.
            if count > 5000:
                return {
                    "total_chunks": count,
                    "total_papers": 0,
                    "papers_by_phase": {},
                    "papers_by_topic": {},
                    "year_range": {"min": 0, "max": 0},
                    "embedding_provider": embed_info.get("provider", "unknown"),
                    "embedding_model": embed_info.get("model", "unknown"),
                    "embedding_dimension": embed_info.get("dimension", 768),
                    "ready": True,
                }

            # Get all metadata for statistics
            all_metadata = self.collection.get(
                include=["metadatas"]
            )

            # Calculate statistics
            papers_by_phase = {}
            papers_by_topic = {}
            doc_metadata = {}
            years: List[int] = []

            for meta in all_metadata["metadatas"]:
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in doc_metadata:
                    doc_metadata[doc_id] = meta

            for meta in doc_metadata.values():
                phase = meta.get("phase", "Unknown")
                papers_by_phase[phase] = papers_by_phase.get(phase, 0) + 1

                topic = meta.get("topic_category", "Unknown")
                papers_by_topic[topic] = papers_by_topic.get(topic, 0) + 1
                year = meta.get("year")
                if isinstance(year, int) and year > 1900:
                    years.append(year)

            year_range = {
                "min": min(years) if years else 0,
                "max": max(years) if years else 0,
            }

            return {
                "total_chunks": count,
                "total_papers": len(doc_metadata),
                "papers_by_phase": papers_by_phase,
                "papers_by_topic": papers_by_topic,
                "year_range": year_range,
                "embedding_provider": embed_info.get("provider", "unknown"),
                "embedding_model": embed_info.get("model", "unknown"),
                "embedding_dimension": embed_info.get("dimension", 768),
                "ready": True
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "embedding_provider": embed_info.get("provider", "unknown"),
                "embedding_model": embed_info.get("model", "unknown"),
                "year_range": {"min": 0, "max": 0},
                "ready": False,
                "error": str(e)
            }

    def is_ready(self) -> bool:
        """Check if system is ready to serve queries."""
        return self.collection is not None and self.collection.count() > 0

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings_list: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Add chunks to the collection incrementally.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            embeddings_list: Optional pre-computed embeddings (if None, will compute)

        Returns:
            Dictionary with results:
                - chunks_added: Number of chunks added
                - success: Boolean indicating success
                - error: Error message if failed
        """
        if not self.collection:
            return {
                "chunks_added": 0,
                "success": False,
                "error": "No collection loaded"
            }

        if not chunks:
            return {
                "chunks_added": 0,
                "success": True,
                "error": None
            }

        try:
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Generate IDs
            ids = [
                meta.get('chunk_id', f"{meta.get('doc_id', 'unknown')}_chunk_{meta.get('chunk_index', i)}")
                for i, meta in enumerate(metadatas)
            ]

            # Compute embeddings if not provided
            if embeddings_list is None:
                embeddings_list = self.embeddings.embed_documents(texts)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas
            )

            # Sync BM25 index if hybrid search is enabled
            bm25_retriever = self._get_bm25_retriever()
            if bm25_retriever:
                bm25_retriever.add_chunks(chunks)

            logger.info(f"Added {len(chunks)} chunks to collection")

            return {
                "chunks_added": len(chunks),
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return {
                "chunks_added": 0,
                "success": False,
                "error": str(e)
            }

    def delete_by_doc_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a document from the collection.

        Args:
            doc_id: Document identifier to delete

        Returns:
            Dictionary with results:
                - doc_id: Document identifier
                - chunks_deleted: Number of chunks removed
                - success: Boolean indicating success
                - error: Error message if failed
        """
        if not self.collection:
            return {
                "doc_id": doc_id,
                "chunks_deleted": 0,
                "success": False,
                "error": "No collection loaded"
            }

        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=[]  # Only need IDs
            )

            if not results or not results.get("ids"):
                return {
                    "doc_id": doc_id,
                    "chunks_deleted": 0,
                    "success": False,
                    "error": f"Document not found: {doc_id}"
                }

            chunk_ids = results["ids"]
            chunks_count = len(chunk_ids)

            # Delete all chunks from ChromaDB
            self.collection.delete(ids=chunk_ids)

            # Also remove from BM25 index if hybrid search is enabled
            bm25_retriever = self._get_bm25_retriever()
            if bm25_retriever:
                bm25_retriever.remove_by_doc_id(doc_id)

            logger.info(f"Deleted {chunks_count} chunks for document {doc_id}")

            return {
                "doc_id": doc_id,
                "chunks_deleted": chunks_count,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "chunks_deleted": 0,
                "success": False,
                "error": str(e)
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.

        Returns:
            List of document metadata dictionaries
        """
        if not self.collection:
            return []

        try:
            # Get all metadata
            results = self.collection.get(include=["metadatas"])

            if not results or not results.get("metadatas"):
                return []

            # Extract unique documents
            unique_docs = {}
            for meta in results["metadatas"]:
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        "doc_id": doc_id,
                        "title": meta.get("title"),
                        "authors": meta.get("authors"),
                        "year": meta.get("year"),
                        "phase": meta.get("phase"),
                        "topic_category": meta.get("topic_category"),
                        "filename": meta.get("filename"),
                        "total_pages": meta.get("total_pages"),
                        "doi": meta.get("doi"),
                        "abstract": meta.get("abstract")
                    }

            return list(unique_docs.values())

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
