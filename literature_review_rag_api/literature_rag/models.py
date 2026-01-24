"""Pydantic Models for Literature Review RAG API

Defines request/response models and enums for API endpoints.
Adapted from personality RAG API models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
