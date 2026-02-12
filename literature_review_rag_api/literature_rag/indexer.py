"""Document Indexer for Literature RAG

Provides reusable indexing functionality for both batch indexing and
runtime PDF uploads. Extracted from build_index.py patterns.

Supports:
- OpenAI (API) embeddings
- Multiple document types (academic, business, generic) via extractor factory
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from .pdf_extractor import AcademicPDFExtractor, extract_keywords_from_text
from .extractors import create_extractor, BaseExtractor

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Reusable document indexer for single PDF uploads.

    Supports incremental indexing without rebuilding the entire collection.
    Uses the same chunking and embedding strategies as build_index.py.
    """

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        collection: chromadb.Collection,
        embeddings: Embeddings,
        config: Optional[dict] = None,
        bm25_retriever=None,
        extractor_type: Optional[str] = None
    ):
        """
        Initialize document indexer.

        Args:
            chroma_client: Existing ChromaDB client instance
            collection: Existing ChromaDB collection to add documents to
            embeddings: Embeddings instance (OpenAI)
            config: Configuration dictionary (optional)
            bm25_retriever: Optional BM25Retriever instance for hybrid search sync
            extractor_type: Override extractor type ("academic", "business", "generic", "auto")
        """
        self.client = chroma_client
        self.collection = collection
        self.embeddings = embeddings
        self.config = config or {}
        self._bm25_retriever = bm25_retriever

        # Initialize PDF extractor based on type
        extraction_config = self.config.get("extraction", {})
        self._extractor_type = extractor_type or extraction_config.get("extractor_type", "academic")

        # Use the extractor factory for flexibility
        if self._extractor_type == "academic":
            # Use the full AcademicPDFExtractor for backward compatibility
            self.pdf_extractor = AcademicPDFExtractor(config=extraction_config)
        else:
            # Use the new extractor factory for other types
            self.pdf_extractor = create_extractor(self._extractor_type, extraction_config)

        # Initialize text splitters
        self._init_text_splitters()

        logger.info(f"DocumentIndexer initialized (extractor: {self._extractor_type})")

    def _init_text_splitters(self):
        """Initialize text splitters for chunking."""
        chunking_config = self.config.get("chunking", {})

        # Standard fixed-size splitter
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get("fixed_chunk_size", 1000),
            chunk_overlap=chunking_config.get("fixed_chunk_overlap", 200),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Hierarchical chunking splitters (parent + child)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def index_pdf(
        self,
        pdf_path: Path,
        phase: str,
        phase_name: str,
        topic_category: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index a single PDF file into the collection.

        Args:
            pdf_path: Path to the PDF file
            phase: Phase identifier (e.g., "Phase 1")
            phase_name: Phase name (e.g., "Theoretical Foundation")
            topic_category: Topic category (e.g., "Business Formation")
            additional_metadata: Extra metadata to include in chunks

        Returns:
            Dictionary with indexing results:
                - doc_id: Document identifier
                - chunks_indexed: Number of chunks created
                - metadata: Extracted document metadata
                - success: Boolean indicating success
                - error: Error message if failed
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return {
                "doc_id": None,
                "chunks_indexed": 0,
                "metadata": None,
                "success": False,
                "error": f"File not found: {pdf_path}"
            }

        try:
            logger.info(f"Indexing PDF: {pdf_path.name}")

            # Extract PDF content and metadata
            sections, metadata = self.pdf_extractor.extract_pdf(
                pdf_path,
                phase_info={
                    "phase": phase,
                    "phase_name": phase_name,
                    "topic_category": topic_category
                }
            )

            # Prepare base metadata for chunks
            chunk_base_metadata = self._prepare_chunk_metadata(
                metadata,
                additional_metadata
            )

            # Chunk the document
            chunks = self._chunk_document(pdf_path, sections, metadata, chunk_base_metadata)

            if not chunks:
                return {
                    "doc_id": metadata.doc_id,
                    "chunks_indexed": 0,
                    "metadata": vars(metadata),
                    "success": False,
                    "error": "No text extracted from PDF"
                }

            # Embed and add to collection
            self._add_chunks_to_collection(chunks)

            logger.info(f"Successfully indexed {len(chunks)} chunks for {metadata.doc_id}")

            return {
                "doc_id": metadata.doc_id,
                "chunks_indexed": len(chunks),
                "metadata": {
                    "title": metadata.title,
                    "authors": metadata.authors,
                    "year": metadata.year,
                    "doi": metadata.doi,
                    "phase": metadata.phase,
                    "topic_category": metadata.topic_category,
                    "total_pages": metadata.total_pages,
                    "filename": metadata.filename,
                    "language": metadata.language
                },
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Failed to index {pdf_path.name}: {e}")
            return {
                "doc_id": None,
                "chunks_indexed": 0,
                "metadata": None,
                "success": False,
                "error": str(e)
            }

    def _prepare_chunk_metadata(
        self,
        metadata,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare base metadata for chunks."""
        normalized_authors = None
        if metadata.authors:
            if isinstance(metadata.authors, list):
                normalized_authors = ", ".join(a for a in metadata.authors if a)
            else:
                normalized_authors = str(metadata.authors).strip()

        normalized_year = metadata.year
        try:
            if normalized_year is not None:
                normalized_year = int(normalized_year)
        except (TypeError, ValueError):
            normalized_year = None

        normalized_doi = metadata.doi.strip().lower() if metadata.doi else None

        chunk_base_metadata = {
            "doc_id": metadata.doc_id,
            "title": " ".join(metadata.title.split()) if metadata.title else metadata.title,
            "authors": normalized_authors or "",
            "year": normalized_year,
            "doi": normalized_doi,
            "phase": metadata.phase,
            "phase_name": metadata.phase_name,
            "topic_category": metadata.topic_category,
            "filename": metadata.filename,
            "file_path": metadata.file_path,
            "abstract": metadata.abstract,
            "total_pages": metadata.total_pages,
            "language": metadata.language
        }

        # Extract keywords
        if metadata.abstract:
            keywords = extract_keywords_from_text(metadata.abstract)
        elif metadata.title:
            keywords = extract_keywords_from_text(metadata.title)
        else:
            keywords = []
        chunk_base_metadata["keywords"] = ", ".join(keywords) if keywords else ""

        # Guess research type and geographic focus
        chunk_base_metadata["research_type"] = self._guess_research_type(
            metadata.topic_category
        )
        geographic_focus = self._guess_geographic_focus(
            metadata.filename,
            metadata.title or ""
        )
        chunk_base_metadata["geographic_focus"] = ", ".join(geographic_focus) if geographic_focus else ""

        # Add any additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                if key not in chunk_base_metadata:
                    chunk_base_metadata[key] = value

        # Sanitize for ChromaDB compatibility
        return self._sanitize_metadata(chunk_base_metadata)

    def _chunk_document(
        self,
        pdf_path: Path,
        sections,
        metadata,
        chunk_base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk document using appropriate strategy."""
        if sections and metadata.extraction_method == "section_aware":
            chunks = self._chunk_sections(sections, chunk_base_metadata)
            logger.info(f"Section-aware chunking: {len(chunks)} chunks")
        else:
            full_text = self.pdf_extractor.extract_full_text(pdf_path)
            if not full_text:
                return []
            chunks = self._chunk_hierarchical(full_text, chunk_base_metadata)
            logger.info(f"Hierarchical chunking: {len(chunks)} chunks")

        return chunks

    def _chunk_hierarchical(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """Create hierarchical chunks (parent + child) with deduplication."""
        chunk_dicts = []
        doc_id = metadata.get("doc_id", "unknown")
        seen_chunks = set()  # Track chunk fingerprints for deduplication

        parent_chunks = self.parent_splitter.split_text(text)
        global_child_index = 0

        for parent_idx, parent_text in enumerate(parent_chunks):
            # Skip very short parent chunks
            if len(parent_text.strip()) < 100:
                continue

            # Check for near-duplicate content
            parent_fingerprint = self._get_chunk_fingerprint(parent_text)
            if parent_fingerprint in seen_chunks:
                continue
            seen_chunks.add(parent_fingerprint)

            parent_id = f"{doc_id}_parent_{parent_idx}"

            # Create parent chunk
            parent_meta = metadata.copy()
            parent_meta["chunk_type"] = "hierarchical_parent"
            parent_meta["chunk_index"] = parent_idx
            parent_meta["chunk_id"] = parent_id
            parent_meta["hierarchy_level"] = "parent"
            parent_meta["section_type"] = "full_text"

            chunk_dicts.append({
                "text": parent_text,
                "metadata": parent_meta
            })

            # Create child chunks from this parent
            child_chunks = self.child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_chunks):
                # Skip very short child chunks
                if len(child_text.strip()) < 50:
                    continue

                # Check for near-duplicate content
                child_fingerprint = self._get_chunk_fingerprint(child_text)
                if child_fingerprint in seen_chunks:
                    continue
                seen_chunks.add(child_fingerprint)

                child_meta = metadata.copy()
                child_meta["chunk_type"] = "hierarchical_child"
                child_meta["chunk_index"] = global_child_index
                child_meta["chunk_id"] = f"{doc_id}_child_{global_child_index}"
                child_meta["parent_id"] = parent_id
                child_meta["hierarchy_level"] = "child"
                child_meta["child_position"] = child_idx
                child_meta["section_type"] = "full_text"

                chunk_dicts.append({
                    "text": child_text,
                    "metadata": child_meta
                })
                global_child_index += 1

        return chunk_dicts

    def _get_chunk_fingerprint(self, text: str) -> str:
        """
        Create a fingerprint for deduplication.
        Uses first 100 chars normalized to detect near-duplicates.
        """
        import re
        # Normalize: lowercase, remove extra whitespace, take first 100 chars
        normalized = re.sub(r'\s+', ' ', text.lower().strip())[:100]
        return normalized

    def _chunk_sections(self, sections, metadata: dict) -> List[Dict[str, Any]]:
        """Chunk extracted sections with section-aware sizes."""
        chunk_dicts = []
        chunk_index = 0

        chunking_config = self.config.get("chunking", {})
        section_sizes = chunking_config.get("section_sizes", {})
        section_overlap = chunking_config.get("section_overlap", 300)
        default_size = chunking_config.get("fixed_chunk_size", 1000)

        for section in sections:
            section_size = section_sizes.get(section.section_type, default_size)

            section_splitter = RecursiveCharacterTextSplitter(
                chunk_size=section_size,
                chunk_overlap=section_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            section_chunks = section_splitter.split_text(section.content)

            for chunk_text in section_chunks:
                chunk_meta = metadata.copy()
                chunk_meta["chunk_type"] = "section_aware"
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["section_type"] = section.section_type
                chunk_meta["page_range"] = f"{section.page_start}-{section.page_end}"

                chunk_dicts.append({
                    "text": chunk_text,
                    "metadata": chunk_meta
                })
                chunk_index += 1

        return chunk_dicts

    def _add_chunks_to_collection(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and add to ChromaDB collection."""
        if not chunks:
            return

        batch_size = self.config.get("embedding", {}).get("batch_size", 32)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            # Generate IDs
            ids = [
                meta.get('chunk_id', f"{meta['doc_id']}_chunk_{meta['chunk_index']}")
                for meta in metadatas
            ]

            # Embed texts
            embeddings = self.embeddings.embed_documents(texts)

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        # Sync BM25 index if available
        if self._bm25_retriever:
            self._bm25_retriever.add_chunks(chunks)

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document and all its chunks from the collection.

        Args:
            doc_id: Document identifier to delete

        Returns:
            Dictionary with deletion results:
                - doc_id: Document identifier
                - chunks_deleted: Number of chunks removed
                - success: Boolean indicating success
                - error: Error message if failed
        """
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

            # Also remove from BM25 index if available
            if self._bm25_retriever:
                self._bm25_retriever.remove_by_doc_id(doc_id)

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

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an indexed document.

        Args:
            doc_id: Document identifier

        Returns:
            Document metadata or None if not found
        """
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
                limit=1
            )

            if results and results.get("metadatas"):
                return results["metadatas"][0]
            return None

        except Exception as e:
            logger.error(f"Error getting document info for {doc_id}: {e}")
            return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the collection.

        Returns:
            List of document metadata dictionaries
        """
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
                        "total_pages": meta.get("total_pages")
                    }

            return list(unique_docs.values())

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are ChromaDB-compatible."""
        cleaned: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value if v is not None)
            elif not isinstance(value, (str, int, float, bool)):
                value = str(value)
            cleaned[key] = value
        return cleaned

    def _guess_research_type(self, topic_category: str) -> str:
        """Guess research type from topic category."""
        topic_lower = (topic_category or "").lower()

        if "method" in topic_lower or "data" in topic_lower:
            return "methodology"
        elif "case stud" in topic_lower:
            return "case_study"
        elif "theoretical" in topic_lower or "framework" in topic_lower:
            return "theoretical"
        else:
            return "empirical"

    def _guess_geographic_focus(self, filename: str, title: str) -> List[str]:
        """Guess geographic focus from filename and title."""
        text = (filename + " " + title).lower()
        focus = []

        if any(term in text for term in ["ruhr", "ruhrgebiet"]):
            focus.append("Ruhr Valley")
        if any(term in text for term in ["nrw", "north rhine", "nordrhein"]):
            focus.append("North Rhine-Westphalia")
        if "german" in text or "deutschland" in text:
            focus.append("Germany")
        if any(term in text for term in ["europe", "eu", "european"]):
            focus.append("Europe")

        return focus if focus else ["Germany"]


def create_indexer_from_rag(rag_instance, config: Optional[dict] = None) -> DocumentIndexer:
    """
    Create a DocumentIndexer from an existing LiteratureReviewRAG instance.

    Args:
        rag_instance: Initialized LiteratureReviewRAG instance
        config: Additional configuration (optional)

    Returns:
        DocumentIndexer instance
    """
    # Get BM25 retriever if hybrid search is enabled
    bm25_retriever = None
    if hasattr(rag_instance, '_get_bm25_retriever'):
        bm25_retriever = rag_instance._get_bm25_retriever()

    return DocumentIndexer(
        chroma_client=rag_instance.client,
        collection=rag_instance.collection,
        embeddings=rag_instance.embeddings,
        config=config or rag_instance.config,
        bm25_retriever=bm25_retriever
    )
