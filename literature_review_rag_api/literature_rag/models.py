"""Pydantic Models for Literature Review RAG API

Defines request/response models and enums for API endpoints.
Adapted from personality RAG API models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class PhaseEnum(str, Enum):
    """Research phases."""
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"


class ResearchTypeEnum(str, Enum):
    """Types of research papers."""
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    CASE_STUDY = "case_study"
    MIXED_METHODS = "mixed_methods"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"


class GeographicFocusEnum(str, Enum):
    """Geographic focus areas."""
    GERMANY = "Germany"
    RUHR_VALLEY = "Ruhr Valley"
    NORTH_RHINE_WESTPHALIA = "North Rhine-Westphalia"
    EUROPE = "Europe"
    GLOBAL = "Global"
    COMPARATIVE = "Comparative"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., description="Search query", min_length=1)
    n_results: int = Field(default=5, description="Number of results to return", ge=1, le=50)

    # Filters
    phase_filter: Optional[PhaseEnum] = Field(default=None, description="Filter by research phase")
    topic_filter: Optional[str] = Field(default=None, description="Filter by topic category")
    year_min: Optional[int] = Field(default=None, description="Minimum publication year", ge=1950, le=2030)
    year_max: Optional[int] = Field(default=None, description="Maximum publication year", ge=1950, le=2030)
    methodology_filter: Optional[str] = Field(default=None, description="Filter by methodology")
    geographic_filter: Optional[GeographicFocusEnum] = Field(default=None, description="Filter by geographic focus")
    research_type_filter: Optional[ResearchTypeEnum] = Field(default=None, description="Filter by research type")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "business formation in Ruhr Valley",
                "n_results": 5,
                "phase_filter": "Phase 2",
                "topic_filter": "Business Formation",
                "year_min": 2015
            }
        }


class ContextRequest(BaseModel):
    """Request model for /context endpoint (LLM-ready context)."""
    query: str = Field(..., description="Search query", min_length=1)
    n_results: int = Field(default=5, description="Number of results", ge=1, le=20)

    # Filters (same as QueryRequest)
    phase_filter: Optional[PhaseEnum] = None
    topic_filter: Optional[str] = None
    year_min: Optional[int] = Field(default=None, ge=1950, le=2030)
    year_max: Optional[int] = Field(default=None, ge=1950, le=2030)
    methodology_filter: Optional[str] = None
    geographic_filter: Optional[GeographicFocusEnum] = None
    research_type_filter: Optional[ResearchTypeEnum] = None


class SynthesisRequest(BaseModel):
    """Request model for /synthesis endpoint (multi-topic query)."""
    query: str = Field(..., description="Search query", min_length=1)
    topics: List[str] = Field(..., description="List of topic categories to query", min_length=1)
    n_per_topic: int = Field(default=2, description="Number of results per topic", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "regional economic transitions",
                "topics": ["Business Formation", "Deindustrialization & Tertiarization"],
                "n_per_topic": 2
            }
        }


class RelatedRequest(BaseModel):
    """Request model for /related endpoint."""
    paper_id: str = Field(..., description="Document ID of the paper")
    n_results: int = Field(default=5, description="Number of related papers", ge=1, le=20)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PaperMetadata(BaseModel):
    """Metadata for a single paper."""
    doc_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    phase: Optional[str] = None
    phase_name: Optional[str] = None
    topic_category: Optional[str] = None
    filename: str
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    research_type: Optional[str] = None
    methodology: Optional[str] = None
    geographic_focus: Optional[List[str]] = None
    section_type: Optional[str] = None
    total_pages: Optional[int] = None


class DocumentResult(BaseModel):
    """Single document result from query."""
    content: str = Field(..., description="Document chunk content")
    metadata: PaperMetadata = Field(..., description="Paper metadata")
    score: float = Field(..., description="Similarity score (0-1, higher is better)")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    query: str = Field(..., description="Original query")
    n_results: int = Field(..., description="Number of results returned")
    documents: List[DocumentResult] = Field(..., description="Retrieved documents")
    detected_terms: List[str] = Field(default=[], description="Academic terms detected and expanded")
    filters_applied: Dict[str, Any] = Field(default={}, description="Filters that were applied")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "business formation in Ruhr Valley",
                "n_results": 2,
                "documents": [
                    {
                        "content": "Business formation in the Ruhr Valley has shown...",
                        "metadata": {
                            "doc_id": "phase2_business_formation_001",
                            "title": "Entrepreneurship in Post-Industrial Regions",
                            "authors": ["Schmidt", "Mueller"],
                            "year": 2020,
                            "phase": "Phase 2",
                            "topic_category": "Business Formation"
                        },
                        "score": 0.89
                    }
                ],
                "detected_terms": ["ruhr valley"],
                "filters_applied": {"phase": "Phase 2"}
            }
        }


class ContextResponse(BaseModel):
    """Response model for /context endpoint."""
    query: str
    context: str = Field(..., description="Formatted context with citations for LLM")
    n_results: int


class SynthesisResponse(BaseModel):
    """Response model for /synthesis endpoint."""
    query: str
    topics: List[str]
    results: Dict[str, str] = Field(..., description="Map of topic → formatted context")


class RelatedResponse(BaseModel):
    """Response model for /related endpoint."""
    paper_id: str
    n_results: int
    related_papers: List[DocumentResult]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = Field(..., description="System status (healthy/unhealthy)")
    ready: bool = Field(..., description="Whether system is ready to serve queries")
    stats: Dict[str, Any] = Field(..., description="System statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ready": True,
                "stats": {
                    "total_chunks": 8543,
                    "total_papers": 86,
                    "papers_by_phase": {
                        "Phase 1": 23,
                        "Phase 2": 25,
                        "Phase 3": 25,
                        "Phase 4": 13
                    },
                    "embedding_model": "BAAI/bge-base-en-v1.5",
                    "embedding_dimension": 768
                }
            }
        }


class PaperInfo(BaseModel):
    """Detailed paper information."""
    doc_id: str
    title: Optional[str]
    authors: Optional[List[str]]
    year: Optional[int]
    doi: Optional[str]
    phase: Optional[str]
    phase_name: Optional[str]
    topic_category: Optional[str]
    filename: str
    abstract: Optional[str]
    total_pages: Optional[int]
    file_path: Optional[str]


class PapersResponse(BaseModel):
    """Response model for /papers endpoint."""
    total_papers: int
    papers: List[PaperInfo]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


# ============================================================================
# FILTER MODELS (for complex filtering)
# ============================================================================

class PaperFilter(BaseModel):
    """Complex filter for paper queries."""
    phases: Optional[List[PhaseEnum]] = Field(default=None, description="Filter by phases")
    topics: Optional[List[str]] = Field(default=None, description="Filter by topic categories")
    year_range: Optional[tuple[int, int]] = Field(default=None, description="Year range (min, max)")
    research_types: Optional[List[ResearchTypeEnum]] = Field(default=None, description="Research types")
    geographic_focus: Optional[List[GeographicFocusEnum]] = Field(default=None, description="Geographic focuses")
    has_abstract: Optional[bool] = Field(default=None, description="Only papers with abstracts")
    has_doi: Optional[bool] = Field(default=None, description="Only papers with DOI")


# ============================================================================
# STATISTICS MODELS
# ============================================================================

class PhaseStatistics(BaseModel):
    """Statistics for a research phase."""
    phase: str
    phase_name: str
    total_papers: int
    total_chunks: int
    topics: Dict[str, int]  # topic → count


class TopicStatistics(BaseModel):
    """Statistics for a topic category."""
    topic: str
    total_papers: int
    total_chunks: int
    avg_year: Optional[float]
    year_range: Optional[tuple[int, int]]


class SystemStatistics(BaseModel):
    """Complete system statistics."""
    total_papers: int
    total_chunks: int
    phases: List[PhaseStatistics]
    topics: List[TopicStatistics]
    year_distribution: Dict[int, int]  # year → count
    research_type_distribution: Dict[str, int]
    geographic_distribution: Dict[str, int]


# ============================================================================
# UPLOAD MODELS
# ============================================================================

class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    success: bool = Field(..., description="Whether upload was successful")
    doc_id: Optional[str] = Field(default=None, description="Document ID of indexed paper")
    filename: str = Field(..., description="Original filename")
    chunks_indexed: int = Field(default=0, description="Number of chunks created")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extracted metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "doc_id": "phase2_business_formation_new_paper",
                "filename": "2024_Smith_Entrepreneurship.pdf",
                "chunks_indexed": 45,
                "metadata": {
                    "title": "Entrepreneurship in Post-Industrial Regions",
                    "authors": ["Smith, J.", "Jones, M."],
                    "year": 2024,
                    "phase": "Phase 2",
                    "topic_category": "Business Formation"
                },
                "error": None
            }
        }


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    doc_id: str
    title: Optional[str]
    authors: Optional[str]
    year: Optional[int]
    phase: Optional[str]
    topic_category: Optional[str]
    filename: Optional[str]
    total_pages: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    total: int = Field(..., description="Total number of documents")
    documents: List[DocumentInfo] = Field(..., description="List of documents")


class DocumentRelationInfo(BaseModel):
    """Related document info."""
    doc_id: str
    related_doc_id: str
    score: float
    title: Optional[str]
    authors: Optional[str]
    year: Optional[int]
    phase: Optional[str]
    topic_category: Optional[str]


class DocumentRelationListResponse(BaseModel):
    """Response model for related documents."""
    total: int = Field(..., description="Total number of related documents")
    relations: List[DocumentRelationInfo] = Field(..., description="List of related documents")


class KnowledgeGapInfo(BaseModel):
    gap_type: str
    best_score: float
    evidence_count: int
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class KnowledgeClaimInfo(BaseModel):
    id: int
    doc_id: str
    paragraph_index: Optional[int]
    claim_text: str
    gaps: List[KnowledgeGapInfo] = Field(default_factory=list)


class KnowledgeInsightsResponse(BaseModel):
    total_claims: int
    claims: List[KnowledgeClaimInfo]


class KnowledgeInsightsRunResponse(BaseModel):
    documents_processed: int
    claims_extracted: int
    gaps_detected: int


class KnowledgeGraphNode(BaseModel):
    id: int
    name: str
    entity_type: str
    cluster: Optional[str] = None


class KnowledgeGraphEdge(BaseModel):
    source: int
    target: int
    relation_type: str
    weight: float


class KnowledgeGraphResponse(BaseModel):
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]


class KnowledgeGraphRunResponse(BaseModel):
    claims_processed: int
    entities_created: int
    edges_created: int


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    success: bool
    doc_id: str
    chunks_deleted: int = Field(default=0, description="Number of chunks removed")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# AUTH MODELS
# ============================================================================

class RegisterRequest(BaseModel):
    """Request model for user registration."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    name: Optional[str] = Field(default=None, description="User's display name")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "name": "John Doe"
            }
        }


class LoginRequest(BaseModel):
    """Request model for login."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class RefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: Optional[str] = Field(default=None, description="Refresh token")


class UserResponse(BaseModel):
    """Response model for user information."""
    id: int
    email: str
    name: Optional[str]
    avatar_url: Optional[str]
    oauth_provider: str
    is_active: bool
    is_verified: bool
    created_at: str

    class Config:
        from_attributes = True


class OAuthCallbackRequest(BaseModel):
    """Request model for OAuth callback."""
    code: str = Field(..., description="Authorization code from OAuth provider")
    state: Optional[str] = Field(default=None, description="State parameter for CSRF protection")
    redirect_uri: Optional[str] = Field(default=None, description="Redirect URI used in authorization")


# ============================================================================
# JOB MODELS
# ============================================================================

class JobCreateRequest(BaseModel):
    """Request model for creating a job."""
    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    description: Optional[str] = Field(default=None, description="Job description")
    term_maps: Optional[Dict[str, List[List[str]]]] = Field(
        default=None,
        description="Optional term normalization maps for this job"
    )
    extractor_type: Optional[str] = Field(
        default="auto",
        description="Document extractor type: 'academic', 'business', 'generic', or 'auto'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Research Project",
                "description": "Knowledge base for my thesis research",
                "extractor_type": "auto"
            }
        }


class JobResponse(BaseModel):
    """Response model for a job."""
    id: int
    name: str
    description: Optional[str]
    term_maps: Optional[Dict[str, List[List[str]]]] = None
    extractor_type: Optional[str] = "auto"
    collection_name: str
    status: str
    document_count: int
    chunk_count: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    total: int
    jobs: List[JobResponse]


# ============================================================================
# AGENTIC RAG MODELS
# ============================================================================

class AgenticChatRequest(BaseModel):
    """Request model for agentic /api/chat endpoint."""
    question: str = Field(..., description="The question to ask", min_length=1)
    n_sources: int = Field(default=5, description="Number of sources to retrieve", ge=1, le=20)
    phase_filter: Optional[str] = Field(default=None, description="Filter by research phase")
    topic_filter: Optional[str] = Field(default=None, description="Filter by topic category")
    deep_analysis: bool = Field(
        default=False,
        description="Force complex pipeline with full agent reasoning"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Compare business formation patterns in Ruhr Valley with other post-industrial regions",
                "n_sources": 5,
                "phase_filter": None,
                "topic_filter": None,
                "deep_analysis": True
            }
        }


class PipelineStatsResponse(BaseModel):
    """Pipeline execution statistics."""
    llm_calls: int = Field(..., description="Number of LLM calls made")
    retrieval_attempts: int = Field(..., description="Number of retrieval attempts")
    validation_passed: Optional[bool] = Field(
        default=None,
        description="Whether validation passed (null for simple/medium queries)"
    )
    total_time_ms: int = Field(..., description="Total execution time in milliseconds")
    evaluation_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Context evaluation scores (relevance, coverage, diversity, overall)"
    )
    retries: Dict[str, int] = Field(
        default_factory=lambda: {"retrieval": 0, "generation": 0},
        description="Number of retries for each stage"
    )


class AgenticChatResponse(BaseModel):
    """Response model for agentic /api/chat endpoint."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source citations")
    complexity: str = Field(
        ...,
        description="Query complexity classification: simple, medium, or complex"
    )
    pipeline_stats: PipelineStatsResponse = Field(
        ...,
        description="Pipeline execution statistics"
    )
    model: str = Field(..., description="LLM model used")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters that were applied to the query"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is business formation?",
                "answer": "Business formation refers to the process of creating new enterprises...",
                "sources": [
                    {
                        "citation_number": 1,
                        "authors": "Smith, J.",
                        "year": 2020,
                        "title": "Business Formation in Post-Industrial Regions",
                        "doc_id": "phase2_business_001"
                    }
                ],
                "complexity": "simple",
                "pipeline_stats": {
                    "llm_calls": 1,
                    "retrieval_attempts": 1,
                    "validation_passed": None,
                    "total_time_ms": 1850,
                    "evaluation_scores": None,
                    "retries": {"retrieval": 0, "generation": 0}
                },
                "model": "llama-3.3-70b-versatile",
                "filters_applied": {}
            }
        }


# ============================================================================
# CHAT MEMORY MODELS
# ============================================================================

class ChatSessionCreateRequest(BaseModel):
    """Request model for creating a chat session."""
    job_id: int = Field(..., description="Knowledge base (job) id")
    title: Optional[str] = Field(default=None, description="Session title")


class ChatSessionUpdateRequest(BaseModel):
    """Request model for updating a chat session."""
    title: Optional[str] = Field(default=None, description="Session title")


class ChatSessionResponse(BaseModel):
    """Response model for a chat session."""
    id: int
    job_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionListResponse(BaseModel):
    """Response model for listing chat sessions."""
    total: int
    sessions: List[ChatSessionResponse]


class ChatMessageCreateRequest(BaseModel):
    """Request model for adding a chat message."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    citations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional citations array for assistant messages"
    )
    model: Optional[str] = Field(default=None, description="Model used for the message")


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    id: int
    session_id: int
    role: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionDetailResponse(BaseModel):
    """Response model for a chat session with messages."""
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]
