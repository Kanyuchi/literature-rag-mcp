"""Evaluation Agent for Agentic RAG

Assesses the quality of retrieved context to determine
if it's sufficient for generating a good response.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple

from ..state import EvaluationScores, RetrievedChunk
from ..prompts import (
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_PROMPT,
    format_context_for_prompt
)

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    Evaluates the quality of retrieved context.

    Responsibilities:
    - Score relevance, coverage, and diversity
    - Determine if retrieval is sufficient to proceed
    - Suggest improvements when score is low
    """

    def __init__(
        self,
        llm_client,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 300,
        threshold: float = 0.7
    ):
        """
        Initialize the evaluation agent.

        Args:
            llm_client: Groq client instance
            model: LLM model to use
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum response tokens
            threshold: Minimum overall score to pass evaluation
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[EvaluationScores, bool, List[str]]:
        """
        Evaluate retrieved context quality.

        Args:
            question: The original question
            chunks: Retrieved chunks to evaluate

        Returns:
            Tuple of (scores, passed, suggestions)
        """
        if not chunks:
            return (
                EvaluationScores(
                    relevance=0.0,
                    coverage=0.0,
                    diversity=0.0,
                    overall=0.0
                ),
                False,
                ["No chunks retrieved - expand query or relax filters"]
            )

        try:
            context = format_context_for_prompt(chunks)
            user_prompt = EVALUATION_USER_PROMPT.format(
                question=question,
                context=context
            )

            response = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response.choices[0].message.content.strip()
            scores, suggestions = self._parse_response(content, chunks)

            passed = scores["overall"] >= self.threshold
            logger.info(
                f"Evaluation complete: overall={scores['overall']:.2f}, "
                f"passed={passed}"
            )

            return scores, passed, suggestions

        except Exception as e:
            logger.error(f"Evaluation agent error: {e}")
            # Return heuristic-based fallback
            return self._heuristic_evaluation(question, chunks)

    def _parse_response(
        self,
        content: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[EvaluationScores, List[str]]:
        """Parse LLM response into scores and suggestions."""
        try:
            # Handle JSON in code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            # Extract and validate scores
            relevance = self._clamp_score(data.get("relevance", 0.5))
            coverage = self._clamp_score(data.get("coverage", 0.5))
            diversity = self._clamp_score(data.get("diversity", 0.5))

            # Calculate overall if not provided
            overall = data.get("overall")
            if overall is None:
                # Weighted average: relevance most important
                overall = (relevance * 0.5 + coverage * 0.3 + diversity * 0.2)
            else:
                overall = self._clamp_score(overall)

            scores = EvaluationScores(
                relevance=relevance,
                coverage=coverage,
                diversity=diversity,
                overall=overall
            )

            # Extract suggestions
            suggestions = []
            if "missing_aspects" in data:
                for aspect in data["missing_aspects"]:
                    suggestions.append(f"Missing: {aspect}")
            if "suggestions" in data:
                suggestions.append(data["suggestions"])

            return scores, suggestions

        except json.JSONDecodeError:
            logger.warning("Failed to parse evaluation response")
            return self._heuristic_scores(chunks), ["Could not parse LLM evaluation"]

    def _heuristic_evaluation(
        self,
        question: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[EvaluationScores, bool, List[str]]:
        """Fallback heuristic-based evaluation."""
        scores = self._heuristic_scores(chunks)
        passed = scores["overall"] >= self.threshold
        suggestions = []

        if scores["diversity"] < 0.5:
            suggestions.append("Low diversity - try expanding to more topics")
        if scores["relevance"] < 0.7:
            suggestions.append("Low relevance - try more specific query terms")

        return scores, passed, suggestions

    def _heuristic_scores(self, chunks: List[RetrievedChunk]) -> EvaluationScores:
        """Calculate heuristic scores without LLM."""
        if not chunks:
            return EvaluationScores(
                relevance=0.0, coverage=0.0, diversity=0.0, overall=0.0
            )

        # Relevance from similarity scores
        avg_score = sum(c.get("score", 0.5) for c in chunks) / len(chunks)
        relevance = min(avg_score + 0.2, 1.0)  # Boost a bit

        # Diversity from unique sources
        unique_docs = len(set(c.get("metadata", {}).get("doc_id", "") for c in chunks))
        diversity = min(unique_docs / max(len(chunks), 1), 1.0)

        # Coverage estimate based on chunk count
        coverage = min(len(chunks) / 5, 1.0)  # Assume 5 chunks = full coverage

        overall = relevance * 0.5 + coverage * 0.3 + diversity * 0.2

        return EvaluationScores(
            relevance=round(relevance, 2),
            coverage=round(coverage, 2),
            diversity=round(diversity, 2),
            overall=round(overall, 2)
        )

    def _clamp_score(self, value: Any) -> float:
        """Clamp a value to [0, 1] range."""
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.5


def create_evaluation_agent(
    llm_client,
    config: Optional[Dict[str, Any]] = None
) -> EvaluationAgent:
    """
    Factory function to create an evaluation agent.

    Args:
        llm_client: Groq client instance
        config: Optional configuration overrides

    Returns:
        Configured EvaluationAgent instance
    """
    config = config or {}
    return EvaluationAgent(
        llm_client=llm_client,
        model=config.get("model", "llama-3.3-70b-versatile"),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 300),
        threshold=config.get("threshold", 0.7)
    )
