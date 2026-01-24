"""Literature Review RAG System

Academic literature review system adapted from personality RAG (100% MBTI accuracy, 15ms queries).
"""

__version__ = "1.0.0"

from .literature_rag import LiteratureReviewRAG
from .config import load_config, LiteratureRAGConfig
from .models import (
    QueryRequest,
    QueryResponse,
    ContextRequest,
    ContextResponse,
    SynthesisRequest,
    SynthesisResponse,
    HealthResponse
)

__all__ = [
    "LiteratureReviewRAG",
    "load_config",
    "LiteratureRAGConfig",
    "QueryRequest",
    "QueryResponse",
    "ContextRequest",
    "ContextResponse",
    "SynthesisRequest",
    "SynthesisResponse",
    "HealthResponse",
]
