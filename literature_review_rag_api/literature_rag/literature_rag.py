"""Literature Review RAG System - Simple, Reliable, Accurate

Adapted from personality RAG system (100% MBTI accuracy, 15ms queries).
Key innovation: Explicit academic term normalization for improved search relevance.
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class LiteratureReviewRAG:
    """RAG system for academic literature review with term normalization."""

    def __init__(self, chroma_path: str, config: dict = None, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize Literature Review RAG system.

        Args:
            chroma_path: Path to ChromaDB persistence directory
            config: Configuration dictionary from literature_config.yaml
            embedding_model: Embedding model name (default: BGE-base)
        """
        self.config = config or {}

        # Device selection
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing Literature Review RAG on device: {device}")

        # Initialize embeddings (same as personality RAG)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Get or create collection
        collection_name = self.config.get("collection_name", "literature_review_chunks")
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            logger.warning(f"Collection {collection_name} not found. Will be created during indexing.")
            self.collection = None

        # Load academic term normalization maps
        self.term_maps = self._load_term_maps(self.config.get("term_maps", {}))

        logger.info("Literature Review RAG initialized successfully")

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

        # Embed query
        query_embedding = self.embeddings.embed_query(expanded_query)

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            logger.debug(f"Query returned {len(results['documents'][0])} results")
            return self._postprocess_results(results, n_results)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

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
        if not self.collection:
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "papers_by_phase": {},
                "papers_by_topic": {},
                "embedding_model": self.embeddings.model_name,
                "ready": False
            }

        try:
            # Get total count
            count = self.collection.count()

            # Get all metadata for statistics
            all_metadata = self.collection.get(
                include=["metadatas"]
            )

            # Calculate statistics
            papers_by_phase = {}
            papers_by_topic = {}
            unique_papers = set()

            for meta in all_metadata["metadatas"]:
                # Track unique papers
                doc_id = meta.get("doc_id")
                if doc_id:
                    unique_papers.add(doc_id)

                # Count by phase
                phase = meta.get("phase", "Unknown")
                papers_by_phase[phase] = papers_by_phase.get(phase, 0) + 1

                # Count by topic
                topic = meta.get("topic_category", "Unknown")
                papers_by_topic[topic] = papers_by_topic.get(topic, 0) + 1

            return {
                "total_chunks": count,
                "total_papers": len(unique_papers),
                "papers_by_phase": papers_by_phase,
                "papers_by_topic": papers_by_topic,
                "embedding_model": self.embeddings.model_name,
                "embedding_dimension": 768,
                "ready": True
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "total_papers": 0,
                "embedding_model": self.embeddings.model_name,
                "ready": False,
                "error": str(e)
            }

    def is_ready(self) -> bool:
        """Check if system is ready to serve queries."""
        return self.collection is not None and self.collection.count() > 0
