"""Tenant Isolation Module for Literature RAG

Provides resource isolation and access control for multi-tenant deployments.
Ensures users can only access their own resources.
"""

import logging
from typing import Optional, TypeVar, Generic, Callable
from functools import wraps

from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session

from .database import User, Job, Document, get_db
from .auth import get_current_user

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TenantContext:
    """Context for tenant-scoped operations.

    Provides a way to scope database queries and operations to a specific user.
    """

    def __init__(self, user: User, db: Session):
        """Initialize tenant context.

        Args:
            user: Current authenticated user
            db: Database session
        """
        self.user = user
        self.user_id = user.id
        self.db = db

    def __repr__(self):
        return f"<TenantContext user_id={self.user_id}>"


def get_tenant_context(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> TenantContext:
    """FastAPI dependency to get tenant context.

    Returns:
        TenantContext with authenticated user and database session
    """
    return TenantContext(user=current_user, db=db)


class ResourceAccessDenied(HTTPException):
    """Exception raised when access to a resource is denied."""

    def __init__(self, resource_type: str, resource_id: int):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to {resource_type} {resource_id}"
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ResourceNotFound(HTTPException):
    """Exception raised when a resource is not found."""

    def __init__(self, resource_type: str, resource_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_type} {resource_id} not found"
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


def verify_job_access(
    job_id: int,
    ctx: TenantContext,
    require_active: bool = True
) -> Job:
    """Verify user has access to a job.

    Args:
        job_id: Job ID to verify access to
        ctx: Tenant context
        require_active: If True, only allow access to active jobs

    Returns:
        Job object if access is granted

    Raises:
        ResourceNotFound: If job doesn't exist
        ResourceAccessDenied: If user doesn't own the job
    """
    from .database import JobCRUD, JobStatus

    job = JobCRUD.get_by_id(ctx.db, job_id)

    if not job:
        logger.warning(
            "job_not_found",
            extra={
                "event": "job_not_found",
                "user_id": ctx.user_id,
                "job_id": job_id
            }
        )
        raise ResourceNotFound("Job", job_id)

    if job.user_id != ctx.user_id:
        logger.warning(
            "job_access_denied",
            extra={
                "event": "job_access_denied",
                "user_id": ctx.user_id,
                "job_id": job_id,
                "owner_id": job.user_id
            }
        )
        raise ResourceAccessDenied("Job", job_id)

    if require_active and job.status == JobStatus.DELETED.value:
        logger.warning(
            "job_deleted",
            extra={
                "event": "job_deleted_access",
                "user_id": ctx.user_id,
                "job_id": job_id
            }
        )
        raise ResourceNotFound("Job", job_id)

    return job


def verify_document_access(
    document_id: int,
    ctx: TenantContext
) -> Document:
    """Verify user has access to a document.

    Args:
        document_id: Document ID to verify access to
        ctx: Tenant context

    Returns:
        Document object if access is granted

    Raises:
        ResourceNotFound: If document doesn't exist
        ResourceAccessDenied: If user doesn't own the document's job
    """
    from .database import DocumentCRUD

    document = DocumentCRUD.get_by_id(ctx.db, document_id)

    if not document:
        raise ResourceNotFound("Document", document_id)

    # Verify access through job ownership
    verify_job_access(document.job_id, ctx)

    return document


def verify_document_by_doc_id(
    job_id: int,
    doc_id: str,
    ctx: TenantContext
) -> Document:
    """Verify user has access to a document by doc_id within a job.

    Args:
        job_id: Job ID the document belongs to
        doc_id: Document's doc_id (ChromaDB ID)
        ctx: Tenant context

    Returns:
        Document object if access is granted

    Raises:
        ResourceNotFound: If document doesn't exist
        ResourceAccessDenied: If user doesn't own the job
    """
    from .database import DocumentCRUD

    # First verify job access
    verify_job_access(job_id, ctx)

    # Then get document
    document = DocumentCRUD.get_by_doc_id(ctx.db, doc_id)

    if not document or document.job_id != job_id:
        raise ResourceNotFound("Document", doc_id)

    return document


class TenantScopedQuery:
    """Helper for creating tenant-scoped database queries.

    Ensures all queries are scoped to the current user's resources.
    """

    def __init__(self, ctx: TenantContext):
        """Initialize scoped query helper.

        Args:
            ctx: Tenant context
        """
        self.ctx = ctx

    def jobs(self, include_archived: bool = False, include_deleted: bool = False):
        """Get user's jobs.

        Args:
            include_archived: Include archived jobs
            include_deleted: Include deleted jobs

        Returns:
            List of Job objects
        """
        from .database import JobCRUD
        return JobCRUD.get_user_jobs(
            self.ctx.db,
            self.ctx.user_id,
            include_archived=include_archived
        )

    def job_count(self) -> int:
        """Get count of user's active jobs."""
        return len(self.jobs())

    def documents(self, job_id: Optional[int] = None):
        """Get user's documents.

        Args:
            job_id: Optional job ID to filter by

        Returns:
            List of Document objects
        """
        from .database import DocumentCRUD

        if job_id:
            # Verify job access first
            verify_job_access(job_id, self.ctx)
            return DocumentCRUD.get_job_documents(self.ctx.db, job_id)

        # Get all documents across all user's jobs
        jobs = self.jobs()
        documents = []
        for job in jobs:
            docs = DocumentCRUD.get_job_documents(self.ctx.db, job.id)
            documents.extend(docs)
        return documents

    def document_count(self, job_id: Optional[int] = None) -> int:
        """Get count of user's documents."""
        return len(self.documents(job_id))


def tenant_scoped(func: Callable) -> Callable:
    """Decorator to add tenant context to a function.

    The decorated function will receive a TenantContext as the first argument.
    """
    @wraps(func)
    async def wrapper(
        *args,
        ctx: TenantContext = Depends(get_tenant_context),
        **kwargs
    ):
        return await func(ctx, *args, **kwargs)

    return wrapper


# Isolation configuration
class IsolationConfig:
    """Configuration for tenant isolation."""

    # Whether to enforce strict isolation (log and deny) or audit-only (log but allow)
    strict_mode: bool = True

    # Whether to log all access checks (can be verbose)
    verbose_logging: bool = False


# Global isolation config
isolation_config = IsolationConfig()


def configure_isolation(strict_mode: bool = True, verbose_logging: bool = False):
    """Configure isolation behavior.

    Args:
        strict_mode: If True, deny access on violations. If False, just log.
        verbose_logging: If True, log all access checks, not just violations.
    """
    isolation_config.strict_mode = strict_mode
    isolation_config.verbose_logging = verbose_logging
    logger.info(
        f"Isolation configured: strict_mode={strict_mode}, verbose_logging={verbose_logging}"
    )
