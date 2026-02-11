"""Agentic RAG Pipeline

Main orchestrator for the adaptive query routing system.
Routes queries based on complexity to appropriate processing paths.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple

from .classifier import QueryClassifier, ClassificationResult
from .state import (
    AgenticState,
    QueryComplexity,
    PipelineStats,
    RetrievedChunk,
    create_initial_state
)
from .agents.planning import PlanningAgent, create_planning_agent
from .agents.evaluation import EvaluationAgent, create_evaluation_agent
from .agents.validation import ValidationAgent, create_validation_agent
from .prompts import (
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT,
    GENERATION_STRICT_SYSTEM_PROMPT,
    GENERATION_STRICT_USER_PROMPT,
    SIMPLE_GENERATION_PROMPT,
    format_context_for_prompt,
    format_citation_guide
)

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """
    Adaptive RAG pipeline with query complexity routing.

    Pipeline Flow:
        Query → Classifier → [Simple|Medium|Complex] path
        - Simple/Medium: Fast single-pass retrieval + generation
        - Complex: Planning → Retrieval → Evaluation → Generation → Validation
    """

    def __init__(
        self,
        rag_system,
        llm_client,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pipeline.

        Args:
            rag_system: LiteratureReviewRAG instance
            llm_client: Groq client for LLM calls
            config: Optional configuration dict
        """
        self.rag = rag_system
        self.llm = llm_client
        self.config = config or {}

        # Initialize classifier
        classification_config = self.config.get("classification", {})
        self.classifier = QueryClassifier(
            simple_max_words=classification_config.get("simple_max_words", 15),
            complex_min_topics=classification_config.get("complex_min_topics", 3),
            complex_min_words=classification_config.get("complex_min_words", 40)
        )

        # Initialize agents (lazy - created when needed)
        self._planning_agent: Optional[PlanningAgent] = None
        self._evaluation_agent: Optional[EvaluationAgent] = None
        self._validation_agent: Optional[ValidationAgent] = None

        # Thresholds
        thresholds = self.config.get("thresholds", {})
        self.evaluation_threshold = thresholds.get("evaluation_sufficient", 0.7)
        self.citation_threshold = thresholds.get("citation_accuracy_min", 0.8)
        self.max_retrieval_retries = thresholds.get("max_retrieval_retries", 2)
        self.max_regeneration_retries = thresholds.get("max_regeneration_retries", 1)

        # Agent configs
        self.agent_configs = self.config.get("agents", {})

    @property
    def planning_agent(self) -> PlanningAgent:
        """Lazy initialization of planning agent."""
        if self._planning_agent is None:
            self._planning_agent = create_planning_agent(
                self.llm,
                self.agent_configs.get("planning", {})
            )
        return self._planning_agent

    @property
    def evaluation_agent(self) -> EvaluationAgent:
        """Lazy initialization of evaluation agent."""
        if self._evaluation_agent is None:
            config = self.agent_configs.get("evaluation", {})
            config["threshold"] = self.evaluation_threshold
            self._evaluation_agent = create_evaluation_agent(self.llm, config)
        return self._evaluation_agent

    @property
    def validation_agent(self) -> ValidationAgent:
        """Lazy initialization of validation agent."""
        if self._validation_agent is None:
            config = self.agent_configs.get("validation", {})
            config["citation_accuracy_min"] = self.citation_threshold
            self._validation_agent = create_validation_agent(self.llm, config)
        return self._validation_agent

    def run(
        self,
        question: str,
        n_sources: int = 5,
        phase_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        deep_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Run the agentic RAG pipeline.

        Args:
            question: User's question
            n_sources: Number of sources to retrieve
            phase_filter: Optional phase filter
            topic_filter: Optional topic filter
            deep_analysis: Force complex pipeline

        Returns:
            Dict with answer, sources, complexity, and pipeline_stats
        """
        start_time = time.time()

        # Classify query
        classification = self.classifier.classify(question, force_complex=deep_analysis)
        logger.info(
            f"Query classified as {classification.complexity.value} "
            f"(confidence={classification.confidence:.2f})"
        )

        # Only rerank when explicitly requested via deep analysis.
        use_reranking = deep_analysis

        # Route to appropriate pipeline
        if classification.complexity == QueryComplexity.SIMPLE:
            result = self._run_simple_pipeline(
                question, n_sources, phase_filter, topic_filter, use_reranking
            )
        elif classification.complexity == QueryComplexity.MEDIUM:
            result = self._run_medium_pipeline(
                question, n_sources, phase_filter, topic_filter, use_reranking
            )
        else:
            result = self._run_complex_pipeline(
                question, n_sources, phase_filter, topic_filter, use_reranking
            )

        # Add timing and classification info
        total_time_ms = int((time.time() - start_time) * 1000)
        result["complexity"] = classification.complexity.value
        result["pipeline_stats"]["total_time_ms"] = total_time_ms
        result["classification_signals"] = classification.signals

        logger.info(
            f"Pipeline complete: complexity={classification.complexity.value}, "
            f"time={total_time_ms}ms, llm_calls={result['pipeline_stats']['llm_calls']}"
        )

        return result

    def _run_simple_pipeline(
        self,
        question: str,
        n_sources: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> Dict[str, Any]:
        """
        Simple pipeline: 1 retrieval, 1 LLM call.
        Target latency: <2s
        """
        stats = PipelineStats(
            llm_calls=0,
            retrieval_attempts=1,
            validation_passed=None,
            total_time_ms=0,
            evaluation_scores=None,
            retries={"retrieval": 0, "generation": 0}
        )

        # Retrieve
        chunks = self._retrieve(question, n_sources, phase_filter, topic_filter, use_reranking)

        if not chunks:
            return {
                "question": question,
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "pipeline_stats": stats,
            }

        # Generate
        answer, sources = self._generate_answer(question, chunks, simple=True)
        stats["llm_calls"] = 1

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "pipeline_stats": stats,
        }

    def _run_medium_pipeline(
        self,
        question: str,
        n_sources: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> Dict[str, Any]:
        """
        Medium pipeline: 1 retrieval, 1 LLM call with fuller context.
        Target latency: <4s
        """
        stats = PipelineStats(
            llm_calls=0,
            retrieval_attempts=1,
            validation_passed=None,
            total_time_ms=0,
            evaluation_scores=None,
            retries={"retrieval": 0, "generation": 0}
        )

        # Retrieve more sources for medium complexity
        effective_n = min(n_sources + 2, 10)
        chunks = self._retrieve(question, effective_n, phase_filter, topic_filter, use_reranking)

        if not chunks:
            return {
                "question": question,
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "pipeline_stats": stats,
            }

        # Generate with full context
        answer, sources = self._generate_answer(question, chunks, simple=False)
        stats["llm_calls"] = 1

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "pipeline_stats": stats,
        }

    def _run_complex_pipeline(
        self,
        question: str,
        n_sources: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> Dict[str, Any]:
        """
        Complex pipeline: Planning → Retrieval → Evaluation → Generation → Validation
        Target latency: <10s (with retries: <15s)
        """
        stats = PipelineStats(
            llm_calls=0,
            retrieval_attempts=0,
            validation_passed=None,
            total_time_ms=0,
            evaluation_scores=None,
            retries={"retrieval": 0, "generation": 0}
        )

        # 1. Planning
        available_topics = self._get_available_topics()
        available_phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]

        plan = self.planning_agent.plan(question, available_topics, available_phases)
        stats["llm_calls"] += 1
        logger.debug(f"Planning result: {plan}")

        # 2. Retrieval with potential retries
        chunks = []
        retrieval_passed = False

        for attempt in range(self.max_retrieval_retries + 1):
            stats["retrieval_attempts"] += 1

            if attempt == 0:
                # Initial retrieval based on plan
                chunks = self._smart_retrieve(
                    question, plan, n_sources, phase_filter, topic_filter, use_reranking
                )
            else:
                # Retry with expanded strategy
                stats["retries"]["retrieval"] += 1
                chunks = self._expanded_retrieve(
                    question, plan, n_sources + 3, phase_filter, topic_filter, use_reranking
                )

            if not chunks:
                continue

            # 3. Evaluation
            scores, passed, suggestions = self.evaluation_agent.evaluate(question, chunks)
            stats["llm_calls"] += 1
            stats["evaluation_scores"] = scores

            if passed:
                retrieval_passed = True
                break

            logger.info(f"Evaluation failed (score={scores['overall']:.2f}), retrying...")

        if not chunks:
            return {
                "question": question,
                "answer": "Unable to find sufficient relevant information to answer your question.",
                "sources": [],
                "pipeline_stats": stats,
            }

        # 4. Generation with potential validation retry
        generation_passed = False
        answer = ""
        sources = []

        for attempt in range(self.max_regeneration_retries + 1):
            strict_mode = attempt > 0
            if attempt > 0:
                stats["retries"]["generation"] += 1

            answer, sources = self._generate_answer(
                question, chunks, simple=False, strict=strict_mode,
                previous_issues=stats.get("validation_issues", [])
            )
            stats["llm_calls"] += 1

            # 5. Validation
            validation_result, passed = self.validation_agent.validate(
                question, answer, chunks
            )
            stats["llm_calls"] += 1
            stats["validation_passed"] = passed

            if passed:
                generation_passed = True
                break

            stats["validation_issues"] = validation_result.get("issues", [])
            logger.info(
                f"Validation failed (issues={validation_result['issues']}), "
                f"regenerating with strict mode..."
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "pipeline_stats": stats,
            "planning_info": {
                "intent_type": plan.get("intent_type"),
                "sub_queries": plan.get("sub_queries", []),
            }
        }

    def _retrieve(
        self,
        question: str,
        n_results: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> List[RetrievedChunk]:
        """Basic retrieval from RAG system."""
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter

        try:
            results = self.rag.query(
                question=question,
                n_results=n_results,
                use_reranking=use_reranking,
                **filters
            )
            return self._convert_results(results)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _smart_retrieve(
        self,
        question: str,
        plan: Dict[str, Any],
        n_results: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> List[RetrievedChunk]:
        """Smart retrieval based on planning result."""
        all_chunks = []
        seen_ids = set()

        sub_queries = plan.get("sub_queries", [question])
        results_per_query = max(n_results // len(sub_queries), 2)

        for sub_query in sub_queries:
            chunks = self._retrieve(sub_query, results_per_query, phase_filter, topic_filter, use_reranking)
            for chunk in chunks:
                chunk_id = chunk.get("metadata", {}).get("doc_id", "") + chunk.get("content", "")[:50]
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_chunks.append(chunk)

        # Sort by score and limit
        all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_chunks[:n_results]

    def _expanded_retrieve(
        self,
        question: str,
        plan: Dict[str, Any],
        n_results: int,
        phase_filter: Optional[str],
        topic_filter: Optional[str],
        use_reranking: bool
    ) -> List[RetrievedChunk]:
        """Expanded retrieval for retry - relaxes filters."""
        all_chunks = []

        # Try without filters first
        chunks = self._retrieve(question, n_results, None, None, use_reranking)
        all_chunks.extend(chunks)

        # Add sub-query results
        for sub_query in plan.get("sub_queries", [])[:3]:
            more_chunks = self._retrieve(sub_query, 3, None, None, use_reranking)
            all_chunks.extend(more_chunks)

        # Deduplicate and sort
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            cid = chunk.get("metadata", {}).get("doc_id", "") + str(chunk.get("score", 0))
            if cid not in seen:
                seen.add(cid)
                unique_chunks.append(chunk)

        unique_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique_chunks[:n_results]

    def _convert_results(self, results: Dict) -> List[RetrievedChunk]:
        """Convert RAG results to RetrievedChunk format."""
        chunks = []
        if not results.get("documents") or not results["documents"][0]:
            return chunks

        for i in range(len(results["documents"][0])):
            chunk = RetrievedChunk(
                content=results["documents"][0][i],
                metadata=dict(results["metadatas"][0][i]),
                score=1 - results["distances"][0][i]  # Convert distance to similarity
            )
            chunks.append(chunk)

        return chunks

    def _generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        simple: bool = False,
        strict: bool = False,
        previous_issues: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate answer from chunks."""
        context = format_context_for_prompt(chunks)

        # Only format citation guide when needed (skip for simple queries)
        citation_guide = None if simple else format_citation_guide(chunks)

        if simple:
            prompt = SIMPLE_GENERATION_PROMPT.format(
                question=question,
                context=context
            )
            messages = [{"role": "user", "content": prompt}]
        elif strict:
            issues_str = "\n".join(f"- {issue}" for issue in (previous_issues or []))
            messages = [
                {"role": "system", "content": GENERATION_STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_STRICT_USER_PROMPT.format(
                    question=question,
                    citation_guide=citation_guide,
                    context=context,
                    issues=issues_str or "None specified"
                )}
            ]
        else:
            messages = [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_USER_PROMPT.format(
                    question=question,
                    citation_guide=citation_guide,
                    context=context
                )}
            ]

        gen_config = self.agent_configs.get("generation", {})
        response = self.llm.chat.completions.create(
            messages=messages,
            model=gen_config.get("model", "llama-3.3-70b-versatile"),
            temperature=gen_config.get("temperature", 0.2),
            max_tokens=gen_config.get("max_tokens", 2048)
        )

        answer = response.choices[0].message.content

        # Format sources
        sources = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            sources.append({
                "citation_number": i,
                "authors": meta.get("authors", "Unknown"),
                "year": meta.get("year", "n.d."),
                "title": meta.get("title", "Untitled"),
                "doc_id": meta.get("doc_id", "")
            })

        return answer, sources

    def _get_available_topics(self) -> List[str]:
        """Get list of available topics from RAG system."""
        try:
            stats = self.rag.get_stats()
            return list(stats.get("papers_by_topic", {}).keys())
        except Exception:
            return []


def create_agentic_pipeline(
    rag_system,
    llm_client,
    config: Optional[Dict[str, Any]] = None
) -> AgenticRAGPipeline:
    """
    Factory function to create the agentic pipeline.

    Args:
        rag_system: LiteratureReviewRAG instance
        llm_client: Groq client
        config: Optional configuration

    Returns:
        Configured AgenticRAGPipeline
    """
    return AgenticRAGPipeline(rag_system, llm_client, config)
