"""BM25 Sparse Retriever for Hybrid Search

Provides BM25 sparse retrieval to complement dense embeddings for improved
retrieval quality. Combines keyword matching (good for author names, specific terms)
with semantic similarity (good for concepts).
"""

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is required for hybrid search. Install with: pip install rank-bm25")

logger = logging.getLogger(__name__)


@dataclass
class BM25Config:
    """Configuration for BM25 retriever."""
    index_path: str = "./indices/bm25_index.pkl"
    use_stemming: bool = True
    min_token_length: int = 2
    # BM25 parameters (Okapi defaults are generally good)
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization


class BM25Retriever:
    """
    BM25 sparse retriever for keyword-based document retrieval.

    Features:
    - Simple word tokenization with optional Porter stemming
    - Incremental index updates (add/remove chunks)
    - Pickle-based persistence
    - Fast query time (<50ms for 14K chunks)
    """

    def __init__(self, config: Optional[BM25Config] = None):
        """
        Initialize BM25 retriever.

        Args:
            config: BM25Config object with settings
        """
        self.config = config or BM25Config()
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []  # Tokenized documents
        self._chunk_ids: List[str] = []  # Parallel list of chunk IDs
        self._chunk_texts: List[str] = []  # Original texts for reference

        # Initialize stemmer if enabled
        self._stemmer = None
        if self.config.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not available, using simple tokenization without stemming")

        logger.info(f"BM25Retriever initialized (stemming: {self.config.use_stemming})")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing/querying.

        Uses simple word splitting with optional Porter stemming.
        Good for academic text with technical terms.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and extract word tokens
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        # Apply stemming if available
        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    def build_index(self, chunks: List[Dict[str, Any]], save: bool = True) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            save: Whether to save the index after building
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        self._corpus = []
        self._chunk_ids = []
        self._chunk_texts = []

        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            # Generate chunk ID
            chunk_id = metadata.get(
                "chunk_id",
                f"{metadata.get('doc_id', 'unknown')}_chunk_{metadata.get('chunk_index', 0)}"
            )

            # Tokenize and add to corpus
            tokens = self._tokenize(text)
            if tokens:  # Only add non-empty documents
                self._corpus.append(tokens)
                self._chunk_ids.append(chunk_id)
                self._chunk_texts.append(text)

        # Build BM25 index
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=self.config.k1, b=self.config.b)
            logger.info(f"BM25 index built with {len(self._corpus)} documents")
        else:
            logger.warning("No valid documents for BM25 index")
            self._bm25 = None

        if save:
            self.save_index()

    def add_chunks(self, chunks: List[Dict[str, Any]], save: bool = True) -> int:
        """
        Add chunks to existing BM25 index.

        Note: BM25Okapi doesn't support incremental updates natively,
        so we rebuild the index with the new chunks added.

        Args:
            chunks: List of chunk dictionaries to add
            save: Whether to save the index after updating

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        added_count = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            chunk_id = metadata.get(
                "chunk_id",
                f"{metadata.get('doc_id', 'unknown')}_chunk_{metadata.get('chunk_index', 0)}"
            )

            # Skip if already indexed
            if chunk_id in self._chunk_ids:
                continue

            tokens = self._tokenize(text)
            if tokens:
                self._corpus.append(tokens)
                self._chunk_ids.append(chunk_id)
                self._chunk_texts.append(text)
                added_count += 1

        # Rebuild BM25 index
        if added_count > 0 and self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=self.config.k1, b=self.config.b)
            logger.info(f"Added {added_count} chunks, BM25 index rebuilt ({len(self._corpus)} total)")

            if save:
                self.save_index()

        return added_count

    def remove_chunks(self, chunk_ids: List[str], save: bool = True) -> int:
        """
        Remove chunks from BM25 index by chunk ID.

        Args:
            chunk_ids: List of chunk IDs to remove
            save: Whether to save the index after updating

        Returns:
            Number of chunks removed
        """
        if not chunk_ids:
            return 0

        chunk_ids_set = set(chunk_ids)
        indices_to_remove = [
            i for i, cid in enumerate(self._chunk_ids) if cid in chunk_ids_set
        ]

        if not indices_to_remove:
            return 0

        # Remove in reverse order to maintain correct indices
        for i in sorted(indices_to_remove, reverse=True):
            self._corpus.pop(i)
            self._chunk_ids.pop(i)
            self._chunk_texts.pop(i)

        # Rebuild BM25 index
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=self.config.k1, b=self.config.b)
        else:
            self._bm25 = None

        removed_count = len(indices_to_remove)
        logger.info(f"Removed {removed_count} chunks, BM25 index rebuilt ({len(self._corpus)} total)")

        if save:
            self.save_index()

        return removed_count

    def remove_by_doc_id(self, doc_id: str, save: bool = True) -> int:
        """
        Remove all chunks for a document from the BM25 index.

        Args:
            doc_id: Document ID to remove
            save: Whether to save after updating

        Returns:
            Number of chunks removed
        """
        # Find chunk IDs that start with the doc_id
        chunks_to_remove = [
            cid for cid in self._chunk_ids
            if cid.startswith(f"{doc_id}_")
        ]
        return self.remove_chunks(chunks_to_remove, save=save)

    def query(self, text: str, n_results: int = 50) -> List[Tuple[str, float]]:
        """
        Query the BM25 index.

        Args:
            text: Query text
            n_results: Maximum number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not self._bm25 or not self._corpus:
            logger.warning("BM25 index not built or empty")
            return []

        # Tokenize query
        query_tokens = self._tokenize(text)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Create (index, score) pairs and sort by score
        scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Return top n_results
        results = [
            (self._chunk_ids[i], score)
            for i, score in scored_indices[:n_results]
        ]

        return results

    def save_index(self) -> None:
        """Save BM25 index to disk."""
        index_path = Path(self.config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": self._corpus,
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "config": {
                "use_stemming": self.config.use_stemming,
                "min_token_length": self.config.min_token_length,
                "k1": self.config.k1,
                "b": self.config.b
            }
        }

        with open(index_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {index_path} ({len(self._corpus)} documents)")

    def load_index(self) -> bool:
        """
        Load BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = Path(self.config.index_path)

        if not index_path.exists():
            logger.info(f"BM25 index not found at {index_path}")
            return False

        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)

            self._corpus = data["corpus"]
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data.get("chunk_texts", [""] * len(self._chunk_ids))

            # Rebuild BM25 from loaded corpus
            if self._corpus:
                saved_config = data.get("config", {})
                self._bm25 = BM25Okapi(
                    self._corpus,
                    k1=saved_config.get("k1", self.config.k1),
                    b=saved_config.get("b", self.config.b)
                )
            else:
                self._bm25 = None

            logger.info(f"BM25 index loaded from {index_path} ({len(self._corpus)} documents)")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if BM25 index is ready for queries."""
        return self._bm25 is not None and len(self._corpus) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics."""
        return {
            "total_documents": len(self._corpus),
            "total_tokens": sum(len(doc) for doc in self._corpus),
            "avg_doc_length": sum(len(doc) for doc in self._corpus) / max(len(self._corpus), 1),
            "use_stemming": self.config.use_stemming,
            "index_path": self.config.index_path,
            "ready": self.is_ready()
        }


class HybridScorer:
    """
    Combines BM25 and dense retrieval scores using Reciprocal Rank Fusion (RRF)
    or weighted combination.

    RRF is preferred as it's robust to score scale differences between
    sparse and dense retrievers.
    """

    def __init__(
        self,
        method: str = "rrf",
        dense_weight: float = 0.7,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid scorer.

        Args:
            method: Fusion method - "rrf" (Reciprocal Rank Fusion) or "weighted"
            dense_weight: Weight for dense scores (if method="weighted")
            rrf_k: RRF constant (typically 60)
        """
        self.method = method
        self.dense_weight = dense_weight
        self.bm25_weight = 1.0 - dense_weight
        self.rrf_k = rrf_k

        logger.info(f"HybridScorer initialized (method: {method})")

    def combine_scores(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Combine BM25 and dense retrieval scores.

        Args:
            bm25_results: List of (chunk_id, bm25_score) from BM25 retriever
            dense_results: List of (chunk_id, distance) from dense retriever
                          Note: Dense results use distance (lower is better)
            n_results: Number of results to return

        Returns:
            List of (chunk_id, combined_score) tuples, sorted by score descending
        """
        if self.method == "rrf":
            return self._rrf_fusion(bm25_results, dense_results, n_results)
        else:
            return self._weighted_fusion(bm25_results, dense_results, n_results)

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF) - robust score combination.

        RRF score = sum(1 / (k + rank)) for each result list

        This method is robust to score scale differences and works well
        when combining different retrieval methods.
        """
        scores: Dict[str, float] = {}

        # Add BM25 RRF scores (already sorted by score desc)
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

        # Add dense RRF scores (sorted by distance asc = best first)
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

        # Sort by combined RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:n_results]

    def _weighted_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Weighted score combination.

        Normalizes scores to [0, 1] range and combines with weights.
        Less robust than RRF but allows explicit control over method importance.
        """
        scores: Dict[str, float] = {}

        # Normalize and add BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1

            for chunk_id, score in bm25_results:
                normalized = (score - min_bm25) / bm25_range
                scores[chunk_id] = self.bm25_weight * normalized

        # Normalize and add dense scores (convert distance to similarity)
        if dense_results:
            max_dist = max(dist for _, dist in dense_results)
            min_dist = min(dist for _, dist in dense_results)
            dist_range = max_dist - min_dist if max_dist != min_dist else 1

            for chunk_id, distance in dense_results:
                # Convert distance to similarity (1 - normalized_distance)
                normalized_sim = 1 - ((distance - min_dist) / dist_range)
                scores[chunk_id] = scores.get(chunk_id, 0) + self.dense_weight * normalized_sim

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:n_results]
