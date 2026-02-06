"""Quota Management Module for Literature RAG

Provides per-tenant quota tracking and enforcement.
Supports tiered plans with configurable limits.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any

from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

logger = logging.getLogger(__name__)


class PlanTier(str, Enum):
    """User plan tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class QuotaLimits:
    """Quota limits for a plan tier."""
    max_documents: int  # Maximum documents per knowledge base
    max_knowledge_bases: int  # Maximum knowledge bases per user
    max_storage_bytes: int  # Maximum total storage in bytes
    max_api_calls_per_day: int  # Maximum API calls per day
    max_file_size_bytes: int  # Maximum single file size
    max_chunks_per_document: int  # Maximum chunks per document


# Default quota limits by tier
QUOTA_LIMITS: Dict[PlanTier, QuotaLimits] = {
    PlanTier.FREE: QuotaLimits(
        max_documents=50,
        max_knowledge_bases=3,
        max_storage_bytes=100 * 1024 * 1024,  # 100 MB
        max_api_calls_per_day=500,
        max_file_size_bytes=10 * 1024 * 1024,  # 10 MB
        max_chunks_per_document=500
    ),
    PlanTier.PRO: QuotaLimits(
        max_documents=500,
        max_knowledge_bases=20,
        max_storage_bytes=5 * 1024 * 1024 * 1024,  # 5 GB
        max_api_calls_per_day=10000,
        max_file_size_bytes=50 * 1024 * 1024,  # 50 MB
        max_chunks_per_document=2000
    ),
    PlanTier.ENTERPRISE: QuotaLimits(
        max_documents=-1,  # Unlimited (-1)
        max_knowledge_bases=-1,
        max_storage_bytes=-1,
        max_api_calls_per_day=-1,
        max_file_size_bytes=100 * 1024 * 1024,  # 100 MB
        max_chunks_per_document=-1
    )
}


@dataclass
class QuotaUsage:
    """Current quota usage for a user."""
    user_id: int
    plan_tier: PlanTier
    document_count: int
    knowledge_base_count: int
    storage_bytes: int
    api_calls_today: int
    last_api_call_date: Optional[datetime]

    def get_limits(self) -> QuotaLimits:
        """Get limits for this user's plan."""
        return QUOTA_LIMITS[self.plan_tier]

    def can_upload_document(self, file_size_bytes: int = 0) -> tuple[bool, str]:
        """Check if user can upload a document."""
        limits = self.get_limits()

        # Check document count
        if limits.max_documents != -1 and self.document_count >= limits.max_documents:
            return False, f"Document limit reached ({limits.max_documents}). Upgrade plan for more documents."

        # Check file size
        if limits.max_file_size_bytes != -1 and file_size_bytes > limits.max_file_size_bytes:
            max_mb = limits.max_file_size_bytes / (1024 * 1024)
            return False, f"File too large. Maximum: {max_mb:.0f} MB for your plan."

        # Check storage
        if limits.max_storage_bytes != -1:
            if self.storage_bytes + file_size_bytes > limits.max_storage_bytes:
                max_gb = limits.max_storage_bytes / (1024 * 1024 * 1024)
                return False, f"Storage limit reached ({max_gb:.1f} GB). Upgrade plan for more storage."

        return True, ""

    def can_create_knowledge_base(self) -> tuple[bool, str]:
        """Check if user can create a knowledge base."""
        limits = self.get_limits()

        if limits.max_knowledge_bases != -1 and self.knowledge_base_count >= limits.max_knowledge_bases:
            return False, f"Knowledge base limit reached ({limits.max_knowledge_bases}). Upgrade plan for more."

        return True, ""

    def can_make_api_call(self) -> tuple[bool, str]:
        """Check if user can make an API call."""
        limits = self.get_limits()

        # Reset count if new day
        today = datetime.utcnow().date()
        if self.last_api_call_date and self.last_api_call_date.date() < today:
            self.api_calls_today = 0

        if limits.max_api_calls_per_day != -1 and self.api_calls_today >= limits.max_api_calls_per_day:
            return False, f"Daily API limit reached ({limits.max_api_calls_per_day}). Try again tomorrow or upgrade."

        return True, ""

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary with percentages."""
        limits = self.get_limits()

        def calc_pct(used: int, limit: int) -> Optional[float]:
            if limit == -1:
                return None
            return round((used / limit) * 100, 1) if limit > 0 else 100.0

        return {
            "plan_tier": self.plan_tier.value,
            "documents": {
                "used": self.document_count,
                "limit": limits.max_documents,
                "percent": calc_pct(self.document_count, limits.max_documents)
            },
            "knowledge_bases": {
                "used": self.knowledge_base_count,
                "limit": limits.max_knowledge_bases,
                "percent": calc_pct(self.knowledge_base_count, limits.max_knowledge_bases)
            },
            "storage": {
                "used_bytes": self.storage_bytes,
                "limit_bytes": limits.max_storage_bytes,
                "used_mb": round(self.storage_bytes / (1024 * 1024), 2),
                "limit_mb": limits.max_storage_bytes / (1024 * 1024) if limits.max_storage_bytes != -1 else None,
                "percent": calc_pct(self.storage_bytes, limits.max_storage_bytes)
            },
            "api_calls_today": {
                "used": self.api_calls_today,
                "limit": limits.max_api_calls_per_day,
                "percent": calc_pct(self.api_calls_today, limits.max_api_calls_per_day)
            }
        }


class QuotaService:
    """Service for quota management."""

    def __init__(self, db_session_factory):
        """Initialize quota service.

        Args:
            db_session_factory: Callable that returns a DB session
        """
        self._session_factory = db_session_factory

    def get_usage(self, user_id: int) -> QuotaUsage:
        """Get current quota usage for a user."""
        from .database import User, Job, Document

        db = self._session_factory()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")

            # Get plan tier from user (default to FREE)
            plan_tier_str = getattr(user, 'plan_tier', None) or PlanTier.FREE.value
            plan_tier = PlanTier(plan_tier_str)

            # Count knowledge bases
            kb_count = db.query(Job).filter(Job.user_id == user_id).count()

            # Count documents and storage across all user's jobs
            jobs = db.query(Job).filter(Job.user_id == user_id).all()
            doc_count = 0
            storage_bytes = 0
            for job in jobs:
                docs = db.query(Document).filter(Document.job_id == job.id).all()
                doc_count += len(docs)
                for doc in docs:
                    storage_bytes += getattr(doc, 'file_size', 0) or 0

            # Get API call count (from usage tracking table if exists)
            api_calls_today = getattr(user, 'api_calls_today', 0) or 0
            last_api_call = getattr(user, 'last_api_call_date', None)

            return QuotaUsage(
                user_id=user_id,
                plan_tier=plan_tier,
                document_count=doc_count,
                knowledge_base_count=kb_count,
                storage_bytes=storage_bytes,
                api_calls_today=api_calls_today,
                last_api_call_date=last_api_call
            )
        finally:
            db.close()

    def increment_api_calls(self, user_id: int) -> int:
        """Increment API call count for user. Returns new count."""
        from .database import User

        db = self._session_factory()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return 0

            today = datetime.utcnow().date()
            last_call_date = getattr(user, 'last_api_call_date', None)

            # Reset if new day
            if last_call_date and last_call_date.date() < today:
                user.api_calls_today = 1
            else:
                user.api_calls_today = (getattr(user, 'api_calls_today', 0) or 0) + 1

            user.last_api_call_date = datetime.utcnow()
            db.commit()

            return user.api_calls_today
        finally:
            db.close()

    def check_document_upload(self, user_id: int, file_size_bytes: int) -> tuple[bool, str]:
        """Check if user can upload a document."""
        usage = self.get_usage(user_id)
        return usage.can_upload_document(file_size_bytes)

    def check_knowledge_base_creation(self, user_id: int) -> tuple[bool, str]:
        """Check if user can create a knowledge base."""
        usage = self.get_usage(user_id)
        return usage.can_create_knowledge_base()

    def check_api_call(self, user_id: int) -> tuple[bool, str]:
        """Check if user can make an API call."""
        usage = self.get_usage(user_id)
        return usage.can_make_api_call()


# Global quota service instance (initialized lazily)
_quota_service: Optional[QuotaService] = None


def get_quota_service() -> QuotaService:
    """Get global quota service instance."""
    global _quota_service
    if _quota_service is None:
        from .database import get_db_session
        _quota_service = QuotaService(get_db_session)
    return _quota_service


def check_quota_for_upload(user_id: int, file_size: int):
    """Dependency to check quota before document upload."""
    service = get_quota_service()
    allowed, message = service.check_document_upload(user_id, file_size)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message
        )


def check_quota_for_kb_creation(user_id: int):
    """Dependency to check quota before knowledge base creation."""
    service = get_quota_service()
    allowed, message = service.check_knowledge_base_creation(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message
        )


def check_quota_for_api_call(user_id: int):
    """Dependency to check quota before API call."""
    service = get_quota_service()
    allowed, message = service.check_api_call(user_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message
        )


def get_user_quota_summary(user_id: int) -> Dict[str, Any]:
    """Get quota usage summary for a user."""
    service = get_quota_service()
    usage = service.get_usage(user_id)
    return usage.get_usage_summary()
