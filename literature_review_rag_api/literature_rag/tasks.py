"""Background Task Management for Literature RAG

Provides task tracking and async processing for PDF uploads.
Uses database-backed storage for durability across restarts.
"""

import logging
import uuid
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadTask:
    """Represents an upload task."""
    task_id: str
    filename: str
    phase: str
    topic: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = "Waiting to start..."
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # File paths
    temp_file_path: Optional[str] = None
    storage_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "phase": self.phase,
            "topic": self.topic,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error
        }


class TaskStore:
    """
    Database-backed task store for tracking upload jobs.
    """

    def __init__(self, max_tasks: int = 100):
        self._max_tasks = max_tasks

    def create_task(
        self,
        filename: str,
        phase: str,
        topic: str,
        temp_file_path: Optional[str] = None
    ) -> UploadTask:
        """
        Create a new upload task.

        Args:
            filename: Original filename
            phase: Target phase
            topic: Target topic
            temp_file_path: Path to temporary file

        Returns:
            Created UploadTask
        """
        task_id = str(uuid.uuid4())[:12]

        task = UploadTask(
            task_id=task_id,
            filename=filename,
            phase=phase,
            topic=topic,
            temp_file_path=temp_file_path
        )

        from .database import get_db_session, UploadTaskCRUD
        db = get_db_session()
        try:
            UploadTaskCRUD.create(
                db=db,
                task_id=task.task_id,
                filename=task.filename,
                phase=task.phase,
                topic=task.topic,
                status=task.status.value,
                progress=task.progress,
                message=task.message,
                temp_file_path=task.temp_file_path
            )
        finally:
            db.close()

        logger.info(f"Created upload task {task_id} for {filename}")
        return task

    def get_task(self, task_id: str) -> Optional[UploadTask]:
        """Get task by ID."""
        from .database import get_db_session, UploadTaskCRUD
        db = get_db_session()
        try:
            record = UploadTaskCRUD.get_by_task_id(db, task_id)
            return self._record_to_task(record) if record else None
        finally:
            db.close()

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[UploadTask]:
        """
        Update task status and progress.

        Args:
            task_id: Task identifier
            status: New status
            progress: Progress percentage (0-100)
            message: Status message
            result: Result data (on completion)
            error: Error message (on failure)

        Returns:
            Updated task or None if not found
        """
        from .database import get_db_session, UploadTaskCRUD
        db = get_db_session()
        try:
            record = UploadTaskCRUD.get_by_task_id(db, task_id)
            if not record:
                return None

            updates: Dict[str, Any] = {}
            if status:
                updates["status"] = status.value
                if status == TaskStatus.PROCESSING and record.started_at is None:
                    updates["started_at"] = datetime.now()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    updates["completed_at"] = datetime.now()

            if progress is not None:
                updates["progress"] = min(100, max(0, progress))

            if message:
                updates["message"] = message

            if result:
                updates["result_json"] = json.dumps(result)

            if error:
                updates["error"] = error

            record = UploadTaskCRUD.update(db, record, **updates)
            return self._record_to_task(record)
        finally:
            db.close()

    def list_tasks(self, limit: int = 20) -> list[UploadTask]:
        """List recent tasks."""
        from .database import get_db_session, UploadTaskCRUD
        db = get_db_session()
        try:
            records = UploadTaskCRUD.list_recent(db, limit=limit)
            return [self._record_to_task(r) for r in records]
        finally:
            db.close()

    def _record_to_task(self, record) -> UploadTask:
        """Convert DB record to UploadTask."""
        result = None
        if record.result_json:
            try:
                result = json.loads(record.result_json)
            except Exception:
                result = None

        return UploadTask(
            task_id=record.task_id,
            filename=record.filename,
            phase=record.phase,
            topic=record.topic,
            status=TaskStatus(record.status),
            progress=record.progress or 0,
            message=record.message or "",
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            result=result,
            error=record.error,
            temp_file_path=record.temp_file_path,
            storage_file_path=record.storage_file_path
        )


# Global task store instance
task_store = TaskStore()


async def process_pdf_task(
    task_id: str,
    indexer,
    temp_file_path: Path,
    phase: str,
    phase_name: str,
    topic: str,
    storage_path: Optional[Path],
    filename: str,
    owner_id: Optional[str] = "default"
):
    """
    Background task to process and index a PDF.

    Args:
        task_id: Task identifier
        indexer: DocumentIndexer instance
        temp_file_path: Path to temporary uploaded file
        phase: Phase identifier
        phase_name: Phase display name
        topic: Topic category
        storage_path: Path to permanent storage
        filename: Original filename
    """
    try:
        # Update status to processing
        task_store.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            progress=10,
            message="Starting PDF extraction..."
        )

        # Small delay to allow status to be read
        await asyncio.sleep(0.1)

        # Update progress - extracting
        task_store.update_task(
            task_id,
            progress=20,
            message="Extracting text and metadata..."
        )

        # Index the PDF (this is the heavy operation)
        result = indexer.index_pdf(
            pdf_path=temp_file_path,
            phase=phase,
            phase_name=phase_name,
            topic_category=topic
        )

        # Update progress - embedding
        task_store.update_task(
            task_id,
            progress=70,
            message="Generating embeddings..."
        )

        await asyncio.sleep(0.1)

        if result["success"]:
            # Upload to S3 (single storage backend)
            task_store.update_task(
                task_id,
                progress=90,
                message="Uploading to S3..."
            )

            from .storage import get_storage
            from .database import get_db_session, DefaultDocumentCRUD
            storage = get_storage()
            try:
                with open(temp_file_path, "rb") as f:
                    storage_key = storage.upload_pdf(
                        job_id=owner_id,
                        phase=phase,
                        topic=topic,
                        filename=filename,
                        file_content=f
                    )
            except Exception as e:
                # Roll back indexed chunks if storage fails
                if result.get("doc_id") and hasattr(indexer, "collection") and indexer.collection:
                    existing = indexer.collection.get(where={"doc_id": result["doc_id"]}, include=[])
                    if existing and existing.get("ids"):
                        indexer.collection.delete(ids=existing["ids"])
                raise e

            if temp_file_path.exists():
                temp_file_path.unlink()

            if str(owner_id).lower() == "default":
                db = get_db_session()
                try:
                    metadata = result.get("metadata") or {}
                    authors_value = metadata.get("authors")
                    if isinstance(authors_value, list):
                        authors_value = ", ".join(authors_value)
                    DefaultDocumentCRUD.create(
                        db=db,
                        doc_id=result["doc_id"],
                        filename=filename,
                        storage_key=storage_key,
                        file_size=None,
                        title=metadata.get("title"),
                        authors=authors_value,
                        year=metadata.get("year"),
                        phase=phase,
                        topic_category=topic,
                        doi=metadata.get("doi"),
                        total_pages=metadata.get("total_pages")
                    )
                finally:
                    db.close()

            # Mark as completed
            task_store.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                message=f"Successfully indexed {result['chunks_indexed']} chunks",
                result={
                    "doc_id": result["doc_id"],
                    "filename": filename,
                    "chunks_indexed": result["chunks_indexed"],
                    "metadata": result["metadata"],
                    "storage_key": storage_key
                }
            )

            logger.info(f"Task {task_id} completed: {result['chunks_indexed']} chunks indexed")
        else:
            # Mark as failed
            task_store.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=0,
                message="Indexing failed",
                error=result.get("error", "Unknown error during indexing")
            )

            # Cleanup temp file
            if temp_file_path.exists():
                temp_file_path.unlink()

            logger.error(f"Task {task_id} failed: {result.get('error')}")

    except Exception as e:
        logger.exception(f"Task {task_id} failed with exception")

        task_store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            progress=0,
            message="Processing failed",
            error=str(e)
        )

        # Cleanup temp file
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
        except Exception:
            pass


def run_async_task(coro):
    """
    Run an async coroutine in a sync context.
    Used for BackgroundTasks which expects sync functions.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
