"""State Definitions for Agentic RAG Pipeline

Defines the state types used throughout the agentic RAG workflow.
Uses TypedDict for LangGraph compatibility.
"""

from typing import TypedDict, Optional, List, Dict, Any
from enum import Enum


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single topic, definition, summary - 1 retrieval, 1 LLM call
    MEDIUM = "medium"      # Multi-source needed, moderate depth - 1 retrieval, 1 LLM call
    COMPLEX = "complex"    # Comparative, synthesis, multi-part - Full agentic pipeline


class RetrievalStrategy(str, Enum):
    """Strategies for context retrieval."""
    SINGLE = "single"           # Single query retrieval
    MULTI_QUERY = "multi_query" # Multiple sub-query retrieval
    EXPANDED = "expanded"       # Expanded filters/query


class ChunkMetadata(TypedDict, total=False):
    """Metadata for a retrieved chunk."""
    doc_id: str
    title: str
    authors: str
    year: int
    phase: str
    topic_category: str
    filename: str
    section_type: str


class RetrievedChunk(TypedDict):
    """A single retrieved chunk with metadata."""
    content: str
    metadata: ChunkMetadata
    score: float


class EvaluationScores(TypedDict):
    """Scores from the evaluation agent."""
    relevance: float     # 0-1: How relevant are chunks to the query
    coverage: float      # 0-1: How well do chunks cover the query aspects
    diversity: float     # 0-1: Source diversity (different papers/topics)
    overall: float       # 0-1: Combined score (weighted average)


class ValidationResult(TypedDict):
    """Result from the validation agent."""
    is_valid: bool
    citation_accuracy: float  # 0-1: Percentage of valid citations
    issues: List[str]         # List of detected issues
    hallucination_detected: bool


class PlanningResult(TypedDict):
    """Result from the planning agent."""
    intent_type: str          # e.g., "comparative", "synthesis", "factual"
    sub_queries: List[str]    # Decomposed queries
    retrieval_strategy: str   # RetrievalStrategy value
    topics_identified: List[str]
    reasoning: str            # Planning agent's reasoning


class PipelineStats(TypedDict):
    """Statistics about pipeline execution."""
    llm_calls: int
    retrieval_attempts: int
    validation_passed: Optional[bool]  # None for simple/medium
    total_time_ms: int
    evaluation_scores: Optional[EvaluationScores]
    retries: Dict[str, int]  # {"retrieval": n, "generation": m}


class AgenticState(TypedDict, total=False):
    """
    State object passed through the LangGraph workflow.

    Contains all information needed for the agentic RAG pipeline,
    including query information, retrieved context, generated response,
    and validation results.
    """
    # Input
    question: str
    n_sources: int
    phase_filter: Optional[str]
    topic_filter: Optional[str]
    deep_analysis: bool  # Force complex pipeline

    # Classification
    complexity: str  # QueryComplexity value

    # Planning (complex queries only)
    planning_result: Optional[PlanningResult]

    # Retrieval
    retrieved_chunks: List[RetrievedChunk]
    retrieval_strategy: str
    retrieval_attempt: int

    # Evaluation (complex queries only)
    evaluation_scores: Optional[EvaluationScores]
    evaluation_passed: bool

    # Generation
    generated_answer: str
    generation_attempt: int
    strict_mode: bool  # Regenerate with stricter constraints

    # Validation (complex queries only)
    validation_result: Optional[ValidationResult]
    validation_passed: bool

    # Output
    final_answer: str
    sources: List[Dict[str, Any]]

    # Metadata
    pipeline_stats: PipelineStats
    error: Optional[str]


def create_initial_state(
    question: str,
    n_sources: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    deep_analysis: bool = False
) -> AgenticState:
    """Create an initial state for the pipeline."""
    return AgenticState(
        question=question,
        n_sources=n_sources,
        phase_filter=phase_filter,
        topic_filter=topic_filter,
        deep_analysis=deep_analysis,
        complexity="",
        planning_result=None,
        retrieved_chunks=[],
        retrieval_strategy=RetrievalStrategy.SINGLE.value,
        retrieval_attempt=0,
        evaluation_scores=None,
        evaluation_passed=False,
        generated_answer="",
        generation_attempt=0,
        strict_mode=False,
        validation_result=None,
        validation_passed=False,
        final_answer="",
        sources=[],
        pipeline_stats=PipelineStats(
            llm_calls=0,
            retrieval_attempts=0,
            validation_passed=None,
            total_time_ms=0,
            evaluation_scores=None,
            retries={"retrieval": 0, "generation": 0}
        ),
        error=None
    )
