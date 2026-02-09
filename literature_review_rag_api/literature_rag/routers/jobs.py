"""Jobs Router for Literature RAG API

Provides endpoints for managing knowledge base jobs.
Each job represents an isolated knowledge base with its own ChromaDB collection.
Each job has its own BM25 index for hybrid search.
"""

import logging
from typing import Optional
from pathlib import Path

import chromadb
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session

from ..database import (
    get_db, User, Job, Document, JobCRUD, DocumentCRUD,
    JobStatus, DocumentStatus
)
from ..auth import get_current_user
from ..models import (
    JobCreateRequest, JobResponse, JobListResponse
)
from ..config import load_config
from ..storage import get_storage, get_storage_auto, is_s3_configured
from ..indexer import DocumentIndexer
from ..bm25_retriever import BM25Retriever, BM25Config, HybridScorer
from ..isolation import (
    get_tenant_context, TenantContext, verify_job_access,
    TenantScopedQuery
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

# Load config
config = load_config()

# Cache for job BM25 retrievers (lazy-loaded)
_job_bm25_cache: dict[int, BM25Retriever] = {}


def get_job_collection(job: Job) -> tuple[chromadb.ClientAPI, chromadb.Collection]:
    """Get or create ChromaDB collection for a job."""
    client = chromadb.PersistentClient(path=config.storage.indices_path)

    try:
        collection = client.get_collection(job.collection_name)
    except Exception:
        collection = client.create_collection(
            name=job.collection_name,
            metadata={"job_id": job.id, "job_name": job.name}
        )

    return client, collection


def get_job_bm25_path(job_id: int) -> str:
    """Get the BM25 index path for a job."""
    indices_path = Path(config.storage.indices_path)
    return str(indices_path / f"bm25_job_{job_id}.pkl")


def get_job_bm25_retriever(job_id: int, collection=None) -> Optional[BM25Retriever]:
    """
    Get or create BM25 retriever for a job.

    Uses caching to avoid reloading the same index multiple times.
    If the index doesn't exist but the collection has documents, builds it on-demand.
    """
    global _job_bm25_cache

    # Check if hybrid search is enabled
    if not config.retrieval.use_hybrid:
        return None

    # Check cache first
    if job_id in _job_bm25_cache:
        return _job_bm25_cache[job_id]

    # Create BM25 config for this job
    bm25_config = BM25Config(
        index_path=get_job_bm25_path(job_id),
        use_stemming=config.retrieval.bm25_use_stemming,
        min_token_length=config.retrieval.bm25_min_token_length
    )

    bm25 = BM25Retriever(bm25_config)

    # Try to load existing index
    if bm25.load_index():
        logger.info(f"Loaded BM25 index for job {job_id} ({bm25.get_stats()['total_documents']} docs)")
        _job_bm25_cache[job_id] = bm25
        return bm25

    # Index doesn't exist - build it from ChromaDB if collection has documents
    if collection is not None and collection.count() > 0:
        logger.info(f"Building BM25 index for job {job_id} from existing ChromaDB data...")
        _build_bm25_from_collection(bm25, collection)
        _job_bm25_cache[job_id] = bm25
        return bm25

    # Empty job - create empty index for future uploads
    _job_bm25_cache[job_id] = bm25
    return bm25


def _build_bm25_from_collection(bm25: BM25Retriever, collection) -> None:
    """Build BM25 index from existing ChromaDB collection data."""
    try:
        # Fetch all documents from collection
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data or not all_data.get("documents"):
            return

        chunks = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            chunks.append({
                "text": doc,
                "metadata": meta
            })

        bm25.build_index(chunks, save=True)
        logger.info(f"Built BM25 index with {len(chunks)} chunks from ChromaDB")

    except Exception as e:
        logger.error(f"Failed to build BM25 from collection: {e}")


def clear_job_bm25_cache(job_id: int) -> None:
    """Remove a job's BM25 retriever from cache."""
    global _job_bm25_cache
    if job_id in _job_bm25_cache:
        del _job_bm25_cache[job_id]


def delete_job_bm25_index(job_id: int) -> None:
    """Delete a job's BM25 index file and clear from cache."""
    clear_job_bm25_cache(job_id)
    bm25_path = Path(get_job_bm25_path(job_id))
    if bm25_path.exists():
        bm25_path.unlink()
        logger.info(f"Deleted BM25 index for job {job_id}")


def require_s3_storage() -> None:
    """Ensure S3 storage is available when required."""
    if getattr(config.upload, "s3_only", False):
        try:
            get_storage()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"S3 storage is required but not configured: {e}"
            )


def build_job_indexer(
    client: chromadb.ClientAPI,
    collection: chromadb.Collection,
    job_id: int
) -> DocumentIndexer:
    """Create a DocumentIndexer for a job collection using shared config."""
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding.model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    indexer_config = {
        "extraction": vars(config.extraction) if hasattr(config.extraction, "__dict__") else {},
        "chunking": vars(config.chunking) if hasattr(config.chunking, "__dict__") else {},
        "embedding": vars(config.embedding) if hasattr(config.embedding, "__dict__") else {}
    }

    # Get BM25 retriever for this job (for hybrid search sync)
    bm25_retriever = get_job_bm25_retriever(job_id) if config.retrieval.use_hybrid else None

    return DocumentIndexer(
        chroma_client=client,
        collection=collection,
        embeddings=embeddings,
        config=indexer_config,
        bm25_retriever=bm25_retriever
    )


def job_to_response(job: Job) -> JobResponse:
    """Convert Job model to JobResponse."""
    term_maps_value = None
    if job.term_maps:
        import json
        try:
            term_maps_value = json.loads(job.term_maps)
        except Exception:
            term_maps_value = None
    return JobResponse(
        id=job.id,
        name=job.name,
        description=job.description,
        term_maps=term_maps_value,
        collection_name=job.collection_name,
        status=job.status,
        document_count=job.document_count,
        chunk_count=job.chunk_count,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )


# ============================================================================
# JOB ENDPOINTS
# ============================================================================

@router.post("", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new knowledge base job.

    Each job has its own isolated ChromaDB collection for documents.
    """
    # Create job in database
    job = JobCRUD.create(
        db=db,
        user_id=current_user.id,
        name=request.name,
        description=request.description
    )

    if request.term_maps:
        import json
        job.term_maps = json.dumps(request.term_maps)
        db.commit()

    # Create ChromaDB collection
    try:
        get_job_collection(job)
        logger.info(f"Created job {job.id} with collection {job.collection_name}")
    except Exception as e:
        # Rollback job creation if collection fails
        db.delete(job)
        db.commit()
        logger.error(f"Failed to create collection for job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create knowledge base collection"
        )

    return job_to_response(job)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    include_archived: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all jobs for the current user.
    """
    jobs = JobCRUD.get_user_jobs(db, current_user.id, include_archived)

    return JobListResponse(
        total=len(jobs),
        jobs=[job_to_response(job) for job in jobs]
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    ctx: TenantContext = Depends(get_tenant_context)
):
    """
    Get a specific job by ID.

    Automatically verifies ownership - users can only access their own jobs.
    """
    # Verify ownership using isolation module
    job = verify_job_access(job_id, ctx)
    return job_to_response(job)


@router.get("/{job_id}/term-maps")
async def get_job_term_maps(
    job_id: int,
    ctx: TenantContext = Depends(get_tenant_context)
):
    """
    Get term normalization maps for a job.

    Automatically verifies ownership - users can only access their own jobs.
    """
    job = verify_job_access(job_id, ctx)

    if not job.term_maps:
        return {"term_maps": {}}

    import json
    try:
        return {"term_maps": json.loads(job.term_maps)}
    except Exception:
        return {"term_maps": {}}


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a job's name or description.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if name:
        job.name = name
    if description is not None:
        job.description = description
    db.commit()
    db.refresh(job)

    return job_to_response(job)


@router.patch("/{job_id}/term-maps", response_model=JobResponse)
async def update_job_term_maps(
    job_id: int,
    term_maps: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update term normalization maps for a job.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    import json
    job.term_maps = json.dumps(term_maps)
    db.commit()
    db.refresh(job)
    return job_to_response(job)


@router.delete("/{job_id}")
async def delete_job(
    job_id: int,
    hard_delete: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job and all its documents.

    By default, this is a soft delete (job is marked as deleted but data is preserved).
    Use ?hard_delete=true to permanently delete the job, its ChromaDB collection,
    and all associated documents.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if hard_delete:
        # Hard delete: remove ChromaDB collection, BM25 index, storage files, and database records
        try:
            # Delete ChromaDB collection
            client = chromadb.PersistentClient(path=config.storage.indices_path)
            try:
                client.delete_collection(job.collection_name)
                logger.info(f"Deleted ChromaDB collection: {job.collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection {job.collection_name}: {e}")

            # Delete BM25 index for this job
            delete_job_bm25_index(job_id)

            # Delete all documents from storage
            documents = DocumentCRUD.get_job_documents(db, job_id)
            storage = get_storage_auto()
            for doc in documents:
                if doc.storage_key:
                    try:
                        storage.delete_pdf(doc.storage_key)
                    except Exception as e:
                        logger.warning(f"Could not delete file {doc.storage_key}: {e}")
                db.delete(doc)

            # Delete the job record
            db.delete(job)
            db.commit()

            logger.info(f"Hard deleted job {job_id} with all data")
            return {"message": "Job permanently deleted", "hard_delete": True}

        except Exception as e:
            logger.error(f"Hard delete failed for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete job: {str(e)}"
            )
    else:
        # Soft delete
        JobCRUD.delete(db, job)
        logger.info(f"Soft deleted job {job_id}")
        return {"message": "Job deleted successfully", "hard_delete": False}


@router.post("/{job_id}/clear")
async def clear_job_documents(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Clear all documents from a job's knowledge base.

    This removes all indexed documents and chunks but keeps the job itself,
    allowing you to start fresh with new uploads.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    try:
        documents_deleted = 0
        chunks_deleted = 0

        # Get the ChromaDB collection
        client = chromadb.PersistentClient(path=config.storage.indices_path)

        # Delete and recreate the collection (fastest way to clear all data)
        try:
            client.delete_collection(job.collection_name)
            # Recreate empty collection
            client.create_collection(
                name=job.collection_name,
                metadata={"job_id": job.id, "job_name": job.name}
            )
            logger.info(f"Cleared ChromaDB collection: {job.collection_name}")
        except Exception as e:
            logger.warning(f"Could not clear collection {job.collection_name}: {e}")

        # Clear BM25 index for this job (delete and let it be recreated on next upload)
        delete_job_bm25_index(job_id)

        # Delete all documents from storage and database
        documents = DocumentCRUD.get_job_documents(db, job_id)
        storage = get_storage_auto()

        for doc in documents:
            chunks_deleted += doc.chunk_count or 0
            if doc.storage_key:
                try:
                    storage.delete_pdf(doc.storage_key)
                except Exception as e:
                    logger.warning(f"Could not delete file {doc.storage_key}: {e}")
            db.delete(doc)
            documents_deleted += 1

        # Reset job stats
        job.document_count = 0
        job.chunk_count = 0
        db.commit()

        logger.info(f"Cleared job {job_id}: {documents_deleted} documents, {chunks_deleted} chunks")

        return {
            "message": "Knowledge base cleared successfully",
            "documents_deleted": documents_deleted,
            "chunks_deleted": chunks_deleted
        }

    except Exception as e:
        logger.error(f"Clear failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.get("/{job_id}/stats")
async def get_job_stats(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get statistics for a job.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Get collection stats
    try:
        _, collection = get_job_collection(job)
        chunk_count = collection.count()
    except Exception:
        chunk_count = 0

    # Get document stats from DB
    documents = DocumentCRUD.get_job_documents(db, job_id)
    indexed_docs = [d for d in documents if d.status == DocumentStatus.INDEXED.value]

    # Get metadata distribution
    phases = {}
    topics = {}
    years = []

    for doc in indexed_docs:
        if doc.phase:
            phases[doc.phase] = phases.get(doc.phase, 0) + 1
        if doc.topic_category:
            topics[doc.topic_category] = topics.get(doc.topic_category, 0) + 1
        if doc.year:
            years.append(doc.year)

    return {
        "job_id": job_id,
        "document_count": len(indexed_docs),
        "chunk_count": chunk_count,
        "phases": phases,
        "topics": topics,
        "year_range": {
            "min": min(years) if years else None,
            "max": max(years) if years else None
        }
    }


@router.get("/{job_id}/documents")
async def list_job_documents(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents in a job.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    documents = DocumentCRUD.get_job_documents(db, job_id)

    return {
        "total": len(documents),
        "documents": [
            {
                "id": doc.id,
                "doc_id": doc.doc_id,
                "filename": doc.original_filename,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "phase": doc.phase,
                "topic_category": doc.topic_category,
                "doi": doc.doi,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "total_pages": doc.total_pages,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ]
    }


@router.get("/{job_id}/documents/{doc_id}/download")
async def get_job_document_download_url(
    job_id: int,
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a presigned URL to download a job document.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    document = DocumentCRUD.get_by_doc_id(db, job_id, doc_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    if not document.storage_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document storage key not available"
        )

    storage = get_storage_auto()
    url = storage.get_presigned_url(document.storage_key)
    return {"doc_id": doc_id, "download_url": url}


@router.get("/{job_id}/query")
async def query_job(
    job_id: int,
    question: str,
    n_sources: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Query documents in a specific job's knowledge base (raw search results).

    Uses hybrid BM25 + dense search when enabled for improved retrieval.
    For LLM-powered answers with citations, use GET /{job_id}/chat instead.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    try:
        # Get collection
        _, collection = get_job_collection(job)

        if collection.count() == 0:
            return {
                "question": question,
                "results": [],
                "message": "No documents in this knowledge base yet"
            }

        # Initialize embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding.model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

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

        # Get BM25 retriever (builds from ChromaDB if index doesn't exist)
        bm25_retriever = get_job_bm25_retriever(job.id, collection=collection)
        use_hybrid = config.retrieval.use_hybrid and bm25_retriever and bm25_retriever.is_ready()

        if use_hybrid:
            # Hybrid search: combine BM25 + dense
            formatted_results = _hybrid_query_job(
                collection, embeddings, bm25_retriever, question, n_sources, where_filter
            )
        else:
            # Dense-only search
            formatted_results = _dense_query_job(
                collection, embeddings, question, n_sources, where_filter
            )

        return {
            "question": question,
            "results": formatted_results,
            "hybrid_search": use_hybrid
        }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


def _dense_query_job(collection, embeddings, question: str, n_sources: int, where_filter) -> list:
    """Execute dense-only query for a job collection."""
    query_embedding = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_sources,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score = 1 - distance

        formatted_results.append({
            "content": results["documents"][0][i],
            "metadata": {
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "phase": metadata.get("phase"),
                "topic_category": metadata.get("topic_category")
            },
            "score": round(score, 4)
        })

    return formatted_results


def _hybrid_query_job(
    collection, embeddings, bm25_retriever, question: str, n_sources: int, where_filter
) -> list:
    """Execute hybrid BM25 + dense query for a job collection."""
    # 1. Get BM25 candidates
    bm25_candidates = config.retrieval.bm25_candidates
    bm25_results = bm25_retriever.query(question, n_results=bm25_candidates)

    # 2. Get dense candidates
    query_embedding = embeddings.embed_query(question)
    dense_k = max(bm25_candidates, n_sources * 3)

    dense_results_raw = collection.query(
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

    # 3. Combine with RRF
    hybrid_scorer = HybridScorer(
        method=config.retrieval.hybrid_method,
        dense_weight=config.retrieval.hybrid_weight
    )
    fusion_k = n_sources * 2
    hybrid_results = hybrid_scorer.combine_scores(bm25_results, dense_results, n_results=fusion_k)

    # 4. Fetch full results by chunk IDs
    hybrid_chunk_ids = [chunk_id for chunk_id, _ in hybrid_results]
    score_lookup = {chunk_id: score for chunk_id, score in hybrid_results}

    formatted_results = []
    if hybrid_chunk_ids:
        try:
            fetched = collection.get(
                ids=hybrid_chunk_ids,
                include=["documents", "metadatas"]
            )

            if fetched and fetched.get("ids"):
                id_to_idx = {id_: i for i, id_ in enumerate(fetched["ids"])}

                for chunk_id in hybrid_chunk_ids:
                    if chunk_id in id_to_idx:
                        idx = id_to_idx[chunk_id]
                        meta = fetched["metadatas"][idx]

                        # Apply where_filter manually if present
                        if where_filter and not _matches_filter(meta, where_filter):
                            continue

                        formatted_results.append({
                            "content": fetched["documents"][idx],
                            "metadata": {
                                "doc_id": meta.get("doc_id"),
                                "title": meta.get("title"),
                                "authors": meta.get("authors"),
                                "year": meta.get("year"),
                                "phase": meta.get("phase"),
                                "topic_category": meta.get("topic_category")
                            },
                            "score": round(score_lookup.get(chunk_id, 0.5), 4)
                        })

                        if len(formatted_results) >= n_sources:
                            break
        except Exception as e:
            logger.warning(f"Hybrid fetch failed, falling back to dense: {e}")
            return _dense_query_job(collection, None, "", n_sources, where_filter)

    return formatted_results


def _matches_filter(metadata: dict, where_filter: dict) -> bool:
    """Check if metadata matches a ChromaDB-style where filter."""
    if not where_filter:
        return True

    if "$and" in where_filter:
        return all(_matches_filter(metadata, cond) for cond in where_filter["$and"])

    if "$or" in where_filter:
        return any(_matches_filter(metadata, cond) for cond in where_filter["$or"])

    for key, condition in where_filter.items():
        if key.startswith("$"):
            continue

        value = metadata.get(key)

        if isinstance(condition, dict):
            for op, op_val in condition.items():
                if op == "$gte" and (value is None or value < op_val):
                    return False
                if op == "$lte" and (value is None or value > op_val):
                    return False
                if op == "$eq" and value != op_val:
                    return False
                if op == "$contains" and (value is None or op_val not in str(value)):
                    return False
        else:
            if value != condition:
                return False

    return True


@router.get("/{job_id}/chat")
async def chat_with_job(
    job_id: int,
    question: str,
    n_sources: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    deep_analysis: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Chat with a job's knowledge base using the agentic RAG pipeline.

    Routes queries based on complexity:
    - Simple queries: Fast path (~2s)
    - Medium queries: Standard RAG (~4s)
    - Complex queries: Full agentic pipeline with planning,
      evaluation, and validation (~6-10s)

    Parameters:
    - question: Your research question
    - n_sources: Number of sources to retrieve (default 5)
    - phase_filter: Filter by phase
    - topic_filter: Filter by topic category
    - deep_analysis: Force complex pipeline for thorough analysis

    Returns:
    - answer: Generated response with citations
    - sources: List of cited sources
    - complexity: Query complexity classification
    - pipeline_stats: Execution statistics
    """
    import os
    from groq import Groq

    from ..job_rag import JobCollectionRAG
    from ..agentic import AgenticRAGPipeline
    from ..models import AgenticChatResponse, PipelineStatsResponse

    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Check for Groq API key
    groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured. Set GROQ_API_KEY in environment."
        )

    try:
        # Get collection
        _, collection = get_job_collection(job)

        if collection.count() == 0:
            return {
                "question": question,
                "answer": "This knowledge base has no documents yet. Please upload some documents first.",
                "sources": [],
                "complexity": "simple",
                "pipeline_stats": {
                    "llm_calls": 0,
                    "retrieval_attempts": 0,
                    "validation_passed": None,
                    "total_time_ms": 0,
                    "evaluation_scores": None,
                    "retries": {"retrieval": 0, "generation": 0}
                },
                "model": config.llm.model,
                "filters_applied": {}
            }

        # Create RAG wrapper for the job collection
        term_maps = None
        if job.term_maps:
            try:
                import json
                term_maps = json.loads(job.term_maps)
            except Exception:
                term_maps = None
        job_rag = JobCollectionRAG(collection, term_maps=term_maps)

        # Initialize Groq client
        groq_client = Groq(api_key=groq_api_key)

        # Build agentic config
        agentic_config = {
            "classification": {
                "simple_max_words": config.agentic.classification.simple_max_words,
                "complex_min_topics": config.agentic.classification.complex_min_topics,
                "complex_min_words": config.agentic.classification.complex_min_words,
            },
            "thresholds": {
                "evaluation_sufficient": config.agentic.thresholds.evaluation_sufficient,
                "citation_accuracy_min": config.agentic.thresholds.citation_accuracy_min,
                "max_retrieval_retries": config.agentic.thresholds.max_retrieval_retries,
                "max_regeneration_retries": config.agentic.thresholds.max_regeneration_retries,
            },
            "agents": config.agentic.agents,
        }

        # Create pipeline with job RAG
        pipeline = AgenticRAGPipeline(job_rag, groq_client, agentic_config)

        # Build filters dict
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter

        # Run pipeline
        result = pipeline.run(
            question=question,
            n_sources=n_sources,
            phase_filter=phase_filter,
            topic_filter=topic_filter,
            deep_analysis=deep_analysis
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "complexity": result["complexity"],
            "pipeline_stats": {
                "llm_calls": result["pipeline_stats"]["llm_calls"],
                "retrieval_attempts": result["pipeline_stats"]["retrieval_attempts"],
                "validation_passed": result["pipeline_stats"]["validation_passed"],
                "total_time_ms": result["pipeline_stats"]["total_time_ms"],
                "evaluation_scores": result["pipeline_stats"].get("evaluation_scores"),
                "retries": result["pipeline_stats"]["retries"]
            },
            "model": config.llm.model,
            "filters_applied": filters
        }

    except Exception as e:
        logger.error(f"Chat error for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


# ============================================================================
# JOB-SCOPED UPLOAD
# ============================================================================

@router.post("/{job_id}/upload")
async def upload_to_job(
    job_id: int,
    file: UploadFile = File(..., description="PDF file to upload"),
    phase: str = Form(..., description="Phase (e.g., 'Phase 1')"),
    topic: str = Form(..., description="Topic category"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and index a PDF to a specific job's knowledge base.
    """
    import uuid
    from pathlib import Path

    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    require_s3_storage()

    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    # Check file size
    contents = await file.read()
    max_size = getattr(config.upload, "max_file_size", 52428800)
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )

    # Save to temp file
    upload_id = str(uuid.uuid4())[:8]
    temp_dir = Path(config.upload.temp_path)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{upload_id}_{file.filename}"

    try:
        with open(temp_file, "wb") as f:
            f.write(contents)

        logger.info(f"Processing upload for job {job_id}: {file.filename}")

        # Get phase name from config
        phases_config = config.data.phases if hasattr(config.data, 'phases') else []
        phase_names = {p.get('name', ''): p.get('full_name', '') for p in phases_config}
        phase_name = phase_names.get(phase, phase)

        # Get job collection + indexer (with BM25 for hybrid search)
        client, collection = get_job_collection(job)
        indexer = build_job_indexer(client, collection, job.id)

        # Index PDF using shared pipeline (section-aware, normalization, etc.)
        result = indexer.index_pdf(
            pdf_path=temp_file,
            phase=phase,
            phase_name=phase_name,
            topic_category=topic
        )

        if not result["success"]:
            if temp_file.exists():
                temp_file.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to index document")
            )

        # Upload to storage (S3 if configured, otherwise local)
        storage = get_storage_auto()
        try:
            with open(temp_file, "rb") as f:
                storage_key = storage.upload_pdf(
                    job_id=job_id,
                    phase=phase,
                    topic=topic,
                    filename=file.filename,
                    file_content=f
                )
        except Exception as e:
            # Roll back chunks if storage fails
            if result.get("doc_id"):
                existing = collection.get(where={"doc_id": result["doc_id"]}, include=[])
                if existing and existing.get("ids"):
                    collection.delete(ids=existing["ids"])
            if temp_file.exists():
                temp_file.unlink()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage upload failed: {e}"
            )

        if temp_file.exists():
            temp_file.unlink()

        # Create document record in database
        authors_value = None
        if result.get("metadata"):
            authors_value = result["metadata"].get("authors")
            if isinstance(authors_value, list):
                authors_value = ", ".join(authors_value)

        document = DocumentCRUD.create(
            db=db,
            job_id=job_id,
            doc_id=result["doc_id"],
            filename=f"{upload_id}_{file.filename}",
            original_filename=file.filename,
            title=result["metadata"].get("title") if result.get("metadata") else None,
            authors=authors_value,
            year=result["metadata"].get("year") if result.get("metadata") else None,
            phase=phase,
            topic_category=topic,
            doi=result["metadata"].get("doi") if result.get("metadata") else None,
            file_size=len(contents),
            storage_key=storage_key,
            total_pages=result["metadata"].get("total_pages") if result.get("metadata") else None
        )

        # Update document status
        DocumentCRUD.update_status(
            db, document,
            status=DocumentStatus.INDEXED.value,
            chunk_count=result["chunks_indexed"]
        )

        # Update job stats
        job.document_count += 1
        job.chunk_count += result["chunks_indexed"]
        db.commit()

        logger.info(f"Successfully indexed {result['chunks_indexed']} chunks for job {job_id}")

        return {
            "success": True,
            "doc_id": result["doc_id"],
            "filename": file.filename,
            "chunks_indexed": result["chunks_indexed"],
            "metadata": result.get("metadata")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{job_id}/documents/{doc_id}")
async def delete_job_document(
    job_id: int,
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document from a job's knowledge base.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Get document from database
    document = DocumentCRUD.get_by_doc_id(db, job_id, doc_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        # Delete from ChromaDB collection
        _, collection = get_job_collection(job)

        results = collection.get(
            where={"doc_id": doc_id},
            include=[]
        )

        if results and results.get("ids"):
            collection.delete(ids=results["ids"])
            chunks_deleted = len(results["ids"])

            # Also remove from BM25 index (don't pass collection - no need to build if missing)
            bm25_retriever = get_job_bm25_retriever(job_id)
            if bm25_retriever:
                bm25_retriever.remove_by_doc_id(doc_id)
        else:
            chunks_deleted = 0

        # Try to delete from storage if storage key exists
        if document.storage_key:
            try:
                storage = get_storage_auto()
                storage.delete_pdf(document.storage_key)
                logger.info(f"Deleted PDF from storage: {document.storage_key}")
            except Exception as e:
                logger.warning(f"Failed to delete from storage: {e}")

        # Delete document record first
        db.delete(document)

        # Update job stats
        job.document_count = max(0, job.document_count - 1)
        job.chunk_count = max(0, job.chunk_count - chunks_deleted)

        # Commit all changes
        db.commit()

        logger.info(f"Deleted document {doc_id} from job {job_id}")

        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_deleted": chunks_deleted
        }

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
