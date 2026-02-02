"""Jobs Router for Literature RAG API

Provides endpoints for managing knowledge base jobs.
Each job represents an isolated knowledge base with its own ChromaDB collection.
"""

import logging
from typing import Optional, List
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

# Load config
config = load_config()


def get_job_collection(job: Job) -> chromadb.Collection:
    """Get or create ChromaDB collection for a job."""
    client = chromadb.PersistentClient(path=config.storage.indices_path)

    try:
        collection = client.get_collection(job.collection_name)
    except Exception:
        collection = client.create_collection(
            name=job.collection_name,
            metadata={"job_id": job.id, "job_name": job.name}
        )

    return collection


def job_to_response(job: Job) -> JobResponse:
    """Convert Job model to JobResponse."""
    return JobResponse(
        id=job.id,
        name=job.name,
        description=job.description,
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific job by ID.
    """
    job = JobCRUD.get_by_id(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check ownership
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return job_to_response(job)


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
        collection = get_job_collection(job)
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
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ]
    }


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
    Query documents in a specific job's knowledge base.
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
        collection = get_job_collection(job)

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
    import shutil
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import torch

    from ..pdf_extractor import AcademicPDFExtractor, extract_keywords_from_text

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

    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    # Check file size (50MB limit)
    contents = await file.read()
    if len(contents) > 52428800:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size: 50MB"
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

        # Extract PDF content
        extractor = AcademicPDFExtractor()

        # Get phase name from config
        phases_config = config.data.phases if hasattr(config.data, 'phases') else []
        phase_names = {p.get('name', ''): p.get('full_name', '') for p in phases_config}
        phase_name = phase_names.get(phase, phase)

        sections, metadata = extractor.extract_pdf(
            temp_file,
            phase_info={
                "phase": phase,
                "phase_name": phase_name,
                "topic_category": topic
            }
        )

        # Extract full text for chunking
        full_text = extractor.extract_full_text(temp_file)

        if not full_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from PDF"
            )

        # Create chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(full_text)

        # Prepare metadata for chunks
        doc_id = metadata.doc_id

        # Extract keywords
        keywords = []
        if metadata.abstract:
            keywords = extract_keywords_from_text(metadata.abstract)
        elif metadata.title:
            keywords = extract_keywords_from_text(metadata.title)

        base_metadata = {
            "doc_id": doc_id,
            "title": metadata.title or "",
            "authors": ", ".join(metadata.authors) if metadata.authors else "",
            "year": metadata.year or 0,
            "doi": metadata.doi or "",
            "phase": phase,
            "phase_name": phase_name,
            "topic_category": topic,
            "filename": file.filename,
            "keywords": ", ".join(keywords) if keywords else "",
            "total_pages": metadata.total_pages or 0
        }

        # Initialize embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding.model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Get job collection
        collection = get_job_collection(job)

        # Prepare chunk data
        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []

        for i, chunk_text in enumerate(chunks):
            chunk_meta = base_metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_metadatas.append(chunk_meta)
            chunk_texts.append(chunk_text)
            chunk_ids.append(f"{doc_id}_chunk_{i}")

        # Embed and add to collection
        chunk_embeddings = embeddings.embed_documents(chunk_texts)

        collection.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )

        # Create document record in database
        document = DocumentCRUD.create(
            db=db,
            job_id=job_id,
            doc_id=doc_id,
            filename=f"{upload_id}_{file.filename}",
            original_filename=file.filename,
            title=metadata.title,
            authors=", ".join(metadata.authors) if metadata.authors else None,
            year=metadata.year,
            phase=phase,
            topic_category=topic,
            file_size=len(contents)
        )

        # Update document status
        DocumentCRUD.update_status(
            db, document,
            status=DocumentStatus.INDEXED.value,
            chunk_count=len(chunks)
        )

        # Update job stats
        job.document_count += 1
        job.chunk_count += len(chunks)
        db.commit()

        # Move file to permanent storage
        storage_dir = Path(config.upload.storage_path) / job.collection_name
        storage_dir.mkdir(parents=True, exist_ok=True)
        permanent_file = storage_dir / f"{upload_id}_{file.filename}"
        shutil.move(str(temp_file), str(permanent_file))

        logger.info(f"Successfully indexed {len(chunks)} chunks for job {job_id}")

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "metadata": {
                "title": metadata.title,
                "authors": metadata.authors,
                "year": metadata.year,
                "phase": phase,
                "topic_category": topic
            }
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
        collection = get_job_collection(job)

        results = collection.get(
            where={"doc_id": doc_id},
            include=[]
        )

        if results and results.get("ids"):
            collection.delete(ids=results["ids"])
            chunks_deleted = len(results["ids"])
        else:
            chunks_deleted = 0

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
