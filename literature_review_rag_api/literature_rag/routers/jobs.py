"""Jobs Router for Literature RAG API

Provides endpoints for managing knowledge base jobs.
Each job represents an isolated knowledge base with its own ChromaDB collection.
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
from ..storage import get_storage
from ..indexer import DocumentIndexer
from ..isolation import (
    get_tenant_context, TenantContext, verify_job_access,
    TenantScopedQuery
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

# Load config
config = load_config()


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


def build_job_indexer(client: chromadb.ClientAPI, collection: chromadb.Collection) -> DocumentIndexer:
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

    return DocumentIndexer(
        chroma_client=client,
        collection=collection,
        embeddings=embeddings,
        config=indexer_config
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job and all its documents.

    This is a soft delete - the job is marked as deleted but data is preserved.
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

    # Soft delete
    JobCRUD.delete(db, job)

    logger.info(f"Deleted job {job_id}")

    return {"message": "Job deleted successfully"}


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

    storage = get_storage()
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

        # Embed query
        query_embedding = embeddings.embed_query(question)

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
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_sources,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance  # Convert distance to similarity

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

        return {
            "question": question,
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


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

        # Get job collection + indexer
        client, collection = get_job_collection(job)
        indexer = build_job_indexer(client, collection)

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

        # Upload to S3 (single storage backend)
        storage = get_storage()
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
                detail=f"S3 upload failed: {e}"
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
        else:
            chunks_deleted = 0

        # Try to delete from S3 if storage key exists
        if document.storage_key:
            try:
                storage = get_storage()
                storage.delete_pdf(document.storage_key)
                logger.info(f"Deleted PDF from S3: {document.storage_key}")
            except Exception as e:
                logger.warning(f"Failed to delete from S3: {e}")

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
