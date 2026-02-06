"""Database Models and Setup for Literature RAG

SQLAlchemy models for users, jobs, and documents.
Uses SQLite for development, can be switched to PostgreSQL for production.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from enum import Enum

logger = logging.getLogger(__name__)

# Database URL - defaults to SQLite, can be overridden with DATABASE_URL env var
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./literature_rag.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class OAuthProvider(str, Enum):
    """Supported OAuth providers."""
    LOCAL = "local"  # Email/password
    GOOGLE = "google"
    GITHUB = "github"


class JobStatus(str, Enum):
    """Job status enumeration."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # Null for OAuth users

    # OAuth fields
    oauth_provider = Column(String(50), default=OAuthProvider.LOCAL.value)
    oauth_id = Column(String(255), nullable=True)  # Provider's user ID

    # Profile
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)

    # Plan & Quotas
    plan_tier = Column(String(50), default="free")  # free, pro, enterprise

    # Usage tracking (for daily API limits)
    api_calls_today = Column(Integer, default=0)
    last_api_call_date = Column(DateTime, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("Job", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Job(Base):
    """Knowledge base job model.

    Each job represents an isolated knowledge base with its own:
    - ChromaDB collection
    - Uploaded documents
    - Search scope
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Job info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    term_maps = Column(Text, nullable=True)  # JSON string for per-job term maps

    # ChromaDB collection name (unique per job)
    collection_name = Column(String(255), unique=True, nullable=False)

    # Storage prefix for S3/local storage
    storage_prefix = Column(String(500), nullable=True)

    # Status
    status = Column(String(50), default=JobStatus.ACTIVE.value)

    # Statistics (cached for performance)
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="jobs")
    documents = relationship("Document", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Job(id={self.id}, name={self.name})>"


class Document(Base):
    """Uploaded document model.

    Tracks documents uploaded to a job's knowledge base.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)

    # Document info
    doc_id = Column(String(255), index=True, nullable=False)  # ChromaDB doc_id
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)

    # Metadata
    title = Column(String(500), nullable=True)
    authors = Column(String(1000), nullable=True)
    year = Column(Integer, nullable=True)
    phase = Column(String(100), nullable=True)
    topic_category = Column(String(255), nullable=True)
    doi = Column(String(255), nullable=True)

    # Storage
    storage_key = Column(String(500), nullable=True)  # S3 key or local path
    file_size = Column(Integer, nullable=True)  # Bytes

    # Processing
    status = Column(String(50), default=DocumentStatus.PENDING.value)
    chunk_count = Column(Integer, default=0)
    total_pages = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    job = relationship("Job", back_populates="documents")

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


class DefaultDocument(Base):
    """Default collection document record."""
    __tablename__ = "default_documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(255), unique=True, index=True, nullable=False)
    filename = Column(String(500), nullable=False)
    storage_key = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)

    title = Column(String(500), nullable=True)
    authors = Column(String(1000), nullable=True)
    year = Column(Integer, nullable=True)
    phase = Column(String(100), nullable=True)
    topic_category = Column(String(255), nullable=True)
    total_pages = Column(Integer, nullable=True)
    doi = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DefaultDocument(id={self.id}, doc_id={self.doc_id})>"


class RefreshToken(Base):
    """Refresh token storage for JWT auth."""
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    token_hash = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    # Device/session info (optional)
    device_info = Column(String(500), nullable=True)
    ip_address = Column(String(50), nullable=True)

    # Status
    is_revoked = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id})>"


class UploadTaskRecord(Base):
    """Async upload task record."""
    __tablename__ = "upload_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(50), unique=True, index=True, nullable=False)
    filename = Column(String(500), nullable=False)
    phase = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    progress = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    temp_file_path = Column(String(1000), nullable=True)
    storage_file_path = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<UploadTaskRecord(task_id={self.task_id}, status={self.status})>"


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def init_db():
    """Initialize database tables."""
    logger.info(f"Initializing database: {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)

    # Auto-migrate: add columns that may not exist in older databases
    _run_migrations()

    logger.info("Database tables created successfully")


def _run_migrations():
    """Run simple ALTER TABLE migrations for new columns."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)

    # Check if 'total_pages' column exists in 'documents' table
    if inspector.has_table("documents"):
        columns = [col["name"] for col in inspector.get_columns("documents")]
        if "total_pages" not in columns:
            logger.info("Migrating: adding total_pages column to documents table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE documents ADD COLUMN total_pages INTEGER"))
            logger.info("Migration complete: total_pages added")
        if "doi" not in columns:
            logger.info("Migrating: adding doi column to documents table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE documents ADD COLUMN doi VARCHAR(255)"))
            logger.info("Migration complete: doi added")

    if inspector.has_table("jobs"):
        columns = [col["name"] for col in inspector.get_columns("jobs")]
        if "term_maps" not in columns:
            logger.info("Migrating: adding term_maps column to jobs table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN term_maps TEXT"))
            logger.info("Migration complete: term_maps added")

    if inspector.has_table("default_documents"):
        columns = [col["name"] for col in inspector.get_columns("default_documents")]
        if "doi" not in columns:
            logger.info("Migrating: adding doi column to default_documents table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE default_documents ADD COLUMN doi VARCHAR(255)"))
            logger.info("Migration complete: doi added")


def get_db() -> Session:
    """Get database session (dependency for FastAPI)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session (for non-async contexts)."""
    return SessionLocal()


# ============================================================================
# CRUD OPERATIONS
# ============================================================================

class UserCRUD:
    """CRUD operations for User model."""

    @staticmethod
    def create(db: Session, email: str, password_hash: Optional[str] = None,
               oauth_provider: str = OAuthProvider.LOCAL.value,
               oauth_id: Optional[str] = None, name: Optional[str] = None) -> User:
        """Create a new user."""
        user = User(
            email=email,
            password_hash=password_hash,
            oauth_provider=oauth_provider,
            oauth_id=oauth_id,
            name=name,
            is_verified=oauth_provider != OAuthProvider.LOCAL.value  # OAuth users are auto-verified
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_by_oauth(db: Session, provider: str, oauth_id: str) -> Optional[User]:
        """Get user by OAuth provider and ID."""
        return db.query(User).filter(
            User.oauth_provider == provider,
            User.oauth_id == oauth_id
        ).first()

    @staticmethod
    def update_last_login(db: Session, user: User):
        """Update user's last login timestamp."""
        user.last_login_at = datetime.utcnow()
        db.commit()


class JobCRUD:
    """CRUD operations for Job model."""

    @staticmethod
    def create(db: Session, user_id: int, name: str, description: Optional[str] = None) -> Job:
        """Create a new job."""
        import uuid
        collection_name = f"job_{uuid.uuid4().hex[:12]}"

        job = Job(
            user_id=user_id,
            name=name,
            description=description,
            collection_name=collection_name,
            storage_prefix=f"jobs/{collection_name}"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_by_id(db: Session, job_id: int) -> Optional[Job]:
        """Get job by ID."""
        return db.query(Job).filter(Job.id == job_id).first()

    @staticmethod
    def get_user_jobs(db: Session, user_id: int, include_archived: bool = False) -> List[Job]:
        """Get all jobs for a user."""
        query = db.query(Job).filter(Job.user_id == user_id)
        if not include_archived:
            query = query.filter(Job.status == JobStatus.ACTIVE.value)
        return query.order_by(Job.created_at.desc()).all()

    @staticmethod
    def update_stats(db: Session, job: Job, document_count: int, chunk_count: int):
        """Update job statistics."""
        job.document_count = document_count
        job.chunk_count = chunk_count
        job.updated_at = datetime.utcnow()
        db.commit()

    @staticmethod
    def delete(db: Session, job: Job):
        """Soft delete a job."""
        job.status = JobStatus.DELETED.value
        job.updated_at = datetime.utcnow()
        db.commit()


class DocumentCRUD:
    """CRUD operations for Document model."""

    @staticmethod
    def create(db: Session, job_id: int, doc_id: str, filename: str,
               original_filename: str, **metadata) -> Document:
        """Create a new document record."""
        document = Document(
            job_id=job_id,
            doc_id=doc_id,
            filename=filename,
            original_filename=original_filename,
            **metadata
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document

    @staticmethod
    def get_by_doc_id(db: Session, job_id: int, doc_id: str) -> Optional[Document]:
        """Get document by doc_id within a job."""
        return db.query(Document).filter(
            Document.job_id == job_id,
            Document.doc_id == doc_id
        ).first()

    @staticmethod
    def get_job_documents(db: Session, job_id: int) -> List[Document]:
        """Get all documents for a job."""
        return db.query(Document).filter(
            Document.job_id == job_id,
            Document.status != DocumentStatus.FAILED.value
        ).order_by(Document.created_at.desc()).all()

    @staticmethod
    def update_status(db: Session, document: Document, status: str,
                      chunk_count: int = 0, error_message: Optional[str] = None):
        """Update document processing status."""
        document.status = status
        document.chunk_count = chunk_count
        document.error_message = error_message
        if status == DocumentStatus.INDEXED.value:
            document.processed_at = datetime.utcnow()
        db.commit()

    @staticmethod
    def delete(db: Session, document: Document):
        """Delete a document record."""
        db.delete(document)
        db.commit()


class DefaultDocumentCRUD:
    """CRUD operations for DefaultDocument model."""

    @staticmethod
    def create(db: Session, doc_id: str, filename: str, storage_key: str,
               file_size: Optional[int] = None, **metadata) -> DefaultDocument:
        """Create a new default document record."""
        document = DefaultDocument(
            doc_id=doc_id,
            filename=filename,
            storage_key=storage_key,
            file_size=file_size,
            **metadata
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document

    @staticmethod
    def get_by_doc_id(db: Session, doc_id: str) -> Optional[DefaultDocument]:
        """Get default document by doc_id."""
        return db.query(DefaultDocument).filter(DefaultDocument.doc_id == doc_id).first()

    @staticmethod
    def list_all(db: Session, limit: int = 100) -> List[DefaultDocument]:
        """List default documents."""
        return db.query(DefaultDocument).order_by(DefaultDocument.created_at.desc()).limit(limit).all()

    @staticmethod
    def delete(db: Session, document: DefaultDocument):
        """Delete a default document record."""
        db.delete(document)
        db.commit()


class UploadTaskCRUD:
    """CRUD operations for UploadTaskRecord model."""

    @staticmethod
    def create(db: Session, **fields) -> UploadTaskRecord:
        record = UploadTaskRecord(**fields)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[UploadTaskRecord]:
        return db.query(UploadTaskRecord).filter(UploadTaskRecord.task_id == task_id).first()

    @staticmethod
    def update(db: Session, record: UploadTaskRecord, **fields) -> UploadTaskRecord:
        for key, value in fields.items():
            setattr(record, key, value)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def list_recent(db: Session, limit: int = 20) -> List[UploadTaskRecord]:
        return db.query(UploadTaskRecord).order_by(UploadTaskRecord.created_at.desc()).limit(limit).all()


class RefreshTokenCRUD:
    """CRUD operations for RefreshToken model."""

    @staticmethod
    def create(db: Session, user_id: int, token_hash: str, expires_at: datetime,
               device_info: Optional[str] = None, ip_address: Optional[str] = None) -> RefreshToken:
        """Create a new refresh token."""
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            device_info=device_info,
            ip_address=ip_address
        )
        db.add(token)
        db.commit()
        db.refresh(token)
        return token

    @staticmethod
    def get_by_hash(db: Session, token_hash: str) -> Optional[RefreshToken]:
        """Get refresh token by hash."""
        return db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False
        ).first()

    @staticmethod
    def revoke(db: Session, token: RefreshToken):
        """Revoke a refresh token."""
        token.is_revoked = True
        db.commit()

    @staticmethod
    def revoke_all_for_user(db: Session, user_id: int):
        """Revoke all refresh tokens for a user."""
        db.query(RefreshToken).filter(
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False
        ).update({"is_revoked": True})
        db.commit()

    @staticmethod
    def cleanup_expired(db: Session):
        """Remove expired tokens."""
        db.query(RefreshToken).filter(
            RefreshToken.expires_at < datetime.utcnow()
        ).delete()
        db.commit()
