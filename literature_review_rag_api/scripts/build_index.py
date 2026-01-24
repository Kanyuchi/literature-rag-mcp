"""Build Literature Review RAG Index

Processes 86 academic PDFs, extracts metadata and content, chunks documents,
embeds chunks, and stores in ChromaDB.

Adapted from personality RAG index building patterns.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from literature_rag.config import load_config
from literature_rag.pdf_extractor import AcademicPDFExtractor, extract_keywords_from_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiteratureIndexBuilder:
    """Build literature review index from academic PDFs."""

    def __init__(self, config):
        """
        Initialize index builder.

        Args:
            config: LiteratureRAGConfig object
        """
        self.config = config

        # Initialize PDF extractor
        self.pdf_extractor = AcademicPDFExtractor(config=vars(config.extraction))

        # Initialize text splitter
        self.init_text_splitter()

        # Initialize embeddings
        self.init_embeddings()

        # Initialize ChromaDB
        self.init_chroma_db()

    def init_text_splitter(self):
        """Initialize text splitters for chunking (including hierarchical)."""
        # Standard fixed-size splitter
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.fixed_chunk_size,
            chunk_overlap=self.config.chunking.fixed_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Hierarchical chunking splitters (like LlamaIndex)
        # Parent chunks: larger for context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Child chunks: smaller for precise retrieval
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(f"Text splitters initialized:")
        logger.info(f"  - Fixed: {self.config.chunking.fixed_chunk_size} chars")
        logger.info(f"  - Hierarchical: Parent 2048 chars, Child 1024 chars")

    def init_embeddings(self):
        """Initialize embedding model."""
        device = self.config.embedding.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding.model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": self.config.embedding.normalize}
        )
        logger.info(f"Embeddings initialized: {self.config.embedding.model} on {device}")

    def init_chroma_db(self):
        """Initialize ChromaDB client and collection."""
        # Create indices directory if it doesn't exist
        indices_path = Path(self.config.storage.indices_path)
        indices_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(indices_path))

        # Delete existing collection if it exists (fresh start)
        collection_name = self.config.storage.collection_name
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Literature review chunks with academic metadata"}
        )
        logger.info(f"Created collection: {collection_name}")

    def find_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Find all PDFs in the data folder with phase information.

        Returns:
            List of dicts with: path, phase, phase_name, topic_category
        """
        pdf_path = Path(self.config.data.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF path not found: {pdf_path}")

        pdfs = []

        # Walk through Phase folders
        for phase_folder in sorted(pdf_path.iterdir()):
            if not phase_folder.is_dir():
                continue

            # Extract phase info from folder name (e.g., "Phase 1 - Theoretical Foundation")
            folder_name = phase_folder.name
            phase_match = folder_name.split(" - ")

            if len(phase_match) >= 2:
                phase = phase_match[0].strip()  # "Phase 1"
                phase_name = phase_match[1].strip()  # "Theoretical Foundation"
            else:
                phase = folder_name
                phase_name = folder_name

            logger.info(f"Processing {phase}: {phase_name}")

            # Walk through topic subfolders
            for topic_folder in sorted(phase_folder.iterdir()):
                if not topic_folder.is_dir():
                    continue

                topic_category = topic_folder.name

                # Find PDFs in this topic
                for pdf_file in topic_folder.glob("*.pdf"):
                    pdfs.append({
                        "path": pdf_file,
                        "phase": phase,
                        "phase_name": phase_name,
                        "topic_category": topic_category
                    })

        logger.info(f"Found {len(pdfs)} PDFs across {len(set(p['phase'] for p in pdfs))} phases")
        return pdfs

    def chunk_text(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """
        Chunk text using fixed-size splitter.

        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk

        Returns:
            List of chunks with metadata
        """
        chunks = self.fixed_splitter.split_text(text)

        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_type"] = "fixed_size"
            chunk_meta["chunk_index"] = i
            chunk_meta["section_type"] = "full_text"  # Since we're using fallback

            chunk_dicts.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })

        return chunk_dicts

    def chunk_hierarchical(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks (parent + child) for better retrieval.

        This mimics LlamaIndex's HierarchicalNodeParser approach:
        - Parent chunks (2048 chars): Provide broader context
        - Child chunks (1024 chars): Enable precise retrieval

        Each child chunk links to its parent via parent_id metadata.

        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk

        Returns:
            List of chunks with hierarchical metadata
        """
        chunk_dicts = []
        doc_id = metadata.get("doc_id", "unknown")

        # First create parent chunks
        parent_chunks = self.parent_splitter.split_text(text)

        global_child_index = 0

        for parent_idx, parent_text in enumerate(parent_chunks):
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

        logger.debug(f"Created {len(parent_chunks)} parent chunks and {global_child_index} child chunks")
        return chunk_dicts

    def chunk_sections(self, sections, metadata: dict) -> List[Dict[str, Any]]:
        """
        Chunk extracted sections with section-aware sizes.

        Args:
            sections: List of ExtractedSection objects
            metadata: Base metadata

        Returns:
            List of chunks with metadata
        """
        chunk_dicts = []
        chunk_index = 0

        for section in sections:
            # Get section-specific chunk size
            section_size = self.config.chunking.section_sizes.get(
                section.section_type,
                self.config.chunking.fixed_chunk_size
            )

            # Create splitter for this section
            section_splitter = RecursiveCharacterTextSplitter(
                chunk_size=section_size,
                chunk_overlap=self.config.chunking.section_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            # Chunk the section
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

    def process_pdf(self, pdf_info: dict) -> List[Dict[str, Any]]:
        """
        Process a single PDF: extract, chunk, prepare for indexing.

        Args:
            pdf_info: Dict with path, phase, phase_name, topic_category

        Returns:
            List of chunks with metadata
        """
        pdf_path = pdf_info["path"]
        logger.info(f"Processing: {pdf_path.name}")

        # Extract PDF
        sections, metadata = self.pdf_extractor.extract_pdf(
            pdf_path,
            phase_info={
                "phase": pdf_info["phase"],
                "phase_name": pdf_info["phase_name"],
                "topic_category": pdf_info["topic_category"]
            }
        )

        # Prepare base metadata for chunks
        # Note: ChromaDB only accepts str, int, float, bool - convert lists to strings
        chunk_base_metadata = {
            "doc_id": metadata.doc_id,
            "title": metadata.title,
            "authors": ", ".join(metadata.authors) if metadata.authors else "",
            "year": metadata.year,
            "doi": metadata.doi,
            "phase": metadata.phase,
            "phase_name": metadata.phase_name,
            "topic_category": metadata.topic_category,
            "filename": metadata.filename,
            "file_path": metadata.file_path,
            "abstract": metadata.abstract,
            "total_pages": metadata.total_pages
        }

        # Extract keywords if we have abstract or title
        if metadata.abstract:
            keywords = extract_keywords_from_text(metadata.abstract)
        elif metadata.title:
            keywords = extract_keywords_from_text(metadata.title)
        else:
            keywords = []

        # Convert keywords list to comma-separated string
        chunk_base_metadata["keywords"] = ", ".join(keywords) if keywords else ""

        # Guess research type from topic category
        research_type = self._guess_research_type(pdf_info["topic_category"])
        chunk_base_metadata["research_type"] = research_type

        # Guess geographic focus (convert list to comma-separated string)
        geographic_focus = self._guess_geographic_focus(pdf_path.name, metadata.title or "")
        chunk_base_metadata["geographic_focus"] = ", ".join(geographic_focus) if geographic_focus else ""

        # Remove None values and coerce metadata to Chroma-compatible types
        chunk_base_metadata = self._sanitize_metadata(chunk_base_metadata)

        # Chunk the document
        if sections and metadata.extraction_method == "section_aware":
            # Section-aware chunking
            chunks = self.chunk_sections(sections, chunk_base_metadata)
            logger.info(f"  → Section-aware chunking: {len(chunks)} chunks from {len(sections)} sections")
        else:
            # Fallback: hierarchical chunking (like LlamaIndex)
            full_text = self.pdf_extractor.extract_full_text(pdf_path)
            if not full_text:
                logger.warning(f"  → No text extracted from {pdf_path.name}")
                return []

            # Use hierarchical chunking for better retrieval
            chunks = self.chunk_hierarchical(full_text, chunk_base_metadata)
            parent_count = sum(1 for c in chunks if c["metadata"].get("hierarchy_level") == "parent")
            child_count = sum(1 for c in chunks if c["metadata"].get("hierarchy_level") == "child")
            logger.info(f"  → Hierarchical chunking: {parent_count} parents + {child_count} children = {len(chunks)} total chunks")

        return chunks

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are ChromaDB-compatible (no None values)."""
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
        topic_lower = topic_category.lower()

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

        return focus if focus else ["Germany"]  # Default

    def build_index(self):
        """Build the complete index from all PDFs."""
        logger.info("=" * 80)
        logger.info("BUILDING LITERATURE REVIEW INDEX")
        logger.info("=" * 80)

        # Find all PDFs
        pdfs = self.find_all_pdfs()
        logger.info(f"\nTotal PDFs to process: {len(pdfs)}")

        # Process all PDFs
        all_chunks = []
        failed_pdfs = []

        for pdf_info in tqdm(pdfs, desc="Processing PDFs"):
            try:
                chunks = self.process_pdf(pdf_info)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_info['path'].name}: {e}")
                failed_pdfs.append(pdf_info['path'].name)

        logger.info(f"\n\nTotal chunks created: {len(all_chunks)}")
        logger.info(f"Failed PDFs: {len(failed_pdfs)}")
        if failed_pdfs:
            logger.warning(f"Failed: {', '.join(failed_pdfs)}")

        # Embed and index chunks
        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING AND INDEXING CHUNKS")
        logger.info("=" * 80)

        batch_size = self.config.embedding.batch_size
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches", total=total_batches):
            batch = all_chunks[i:i + batch_size]

            # Extract texts and metadata
            texts = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            # Generate IDs - use chunk_id if available (for hierarchical chunks), otherwise fallback
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

        # Save metadata cache
        if self.config.storage.cache_metadata:
            self._cache_metadata(all_chunks)

        logger.info("\n" + "=" * 80)
        logger.info("INDEX BUILD COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Total chunks indexed: {self.collection.count()}")
        logger.info(f"Index location: {self.config.storage.indices_path}")
        logger.info(f"Collection name: {self.config.storage.collection_name}")

    def _cache_metadata(self, all_chunks: List[Dict]):
        """Cache metadata for faster system startup."""
        cache_path = Path(self.config.storage.metadata_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract unique papers
        unique_papers = {}
        for chunk in all_chunks:
            meta = chunk["metadata"]
            doc_id = meta["doc_id"]
            if doc_id not in unique_papers:
                unique_papers[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "year": meta.get("year"),
                    "phase": meta.get("phase"),
                    "topic_category": meta.get("topic_category"),
                    "filename": meta.get("filename")
                }

        cache_data = {
            "total_papers": len(unique_papers),
            "total_chunks": len(all_chunks),
            "papers": list(unique_papers.values())
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Metadata cached: {cache_path}")


def main():
    """Main entry point for index building."""
    logger.info("Literature Review RAG - Index Builder")
    logger.info("=" * 80)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    # Create index builder
    builder = LiteratureIndexBuilder(config)

    # Build index
    builder.build_index()

    logger.info("\n✅ Index building complete! You can now start the API server.")
    logger.info(f"Run: uvicorn literature_rag.api:app --host {config.api.host} --port {config.api.port}")


if __name__ == "__main__":
    main()
