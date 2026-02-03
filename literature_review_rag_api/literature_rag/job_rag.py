"""Job Collection RAG Wrapper

Wraps a job's ChromaDB collection to provide the same interface
as LiteratureReviewRAG, enabling use with the agentic pipeline.
"""

import logging
from typing import Dict, List, Optional, Any

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from .config import load_config

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
        embedding_model: str = None
    ):
        """
        Initialize the job collection RAG.

        Args:
            collection: ChromaDB collection for the job
            embedding_model: Embedding model name (default from config)
        """
        self.collection = collection
        self.config = load_config()

        # Initialize embeddings
        embedding_model = embedding_model or self.config.embedding.model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        logger.info(f"JobCollectionRAG initialized for collection: {collection.name}")

    def is_ready(self) -> bool:
        """Check if the collection is ready."""
        return self.collection is not None and self.collection.count() > 0

    def query(
        self,
        question: str,
        n_results: int = 5,
        phase_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the job collection.

        Returns results in the same format as LiteratureReviewRAG.query()
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(question)

        # Build filters
        where_filter = None
        conditions = []

        if phase_filter:
            conditions.append({"phase": phase_filter})
        if topic_filter:
            conditions.append({"topic_category": topic_filter})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        return results

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
                "embedding_model": self.config.embedding.model,
                "embedding_dimension": self.config.embedding.dimension
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
