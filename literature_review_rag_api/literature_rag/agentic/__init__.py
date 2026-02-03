"""Agentic RAG Module

Implements an intelligent RAG system with adaptive query routing.
Simple queries get fast responses; complex queries get multi-agent reasoning
with self-correction.

Pipeline Overview:
    Query → [Classifier] → Simple?  → [Fast RAG] → Response
                        → Medium?  → [Multi-Source RAG] → Response
                        → Complex? → [Planning Agent] → [Smart Retrieval]
                                           ↓
                                   [Evaluation Agent] → Score < 0.7? → Retry
                                           ↓
                                   [Generator] → [Validation Agent]
                                           ↓
                                   Hallucination? → Regenerate (stricter)
                                           ↓
                                   Final Response
"""

from .classifier import QueryClassifier, QueryComplexity
from .state import AgenticState, PipelineStats
from .pipeline import AgenticRAGPipeline

__all__ = [
    "QueryClassifier",
    "QueryComplexity",
    "AgenticState",
    "PipelineStats",
    "AgenticRAGPipeline",
]
