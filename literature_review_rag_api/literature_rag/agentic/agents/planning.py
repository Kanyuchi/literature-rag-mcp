"""Planning Agent for Agentic RAG

Decomposes complex queries into sub-queries and determines
the optimal retrieval strategy.
"""

import json
import logging
from typing import List, Optional, Dict, Any

from ..state import PlanningResult, RetrievalStrategy
from ..prompts import PLANNING_SYSTEM_PROMPT, PLANNING_USER_PROMPT

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Analyzes complex queries and creates retrieval plans.

    Responsibilities:
    - Identify query intent (comparative, synthesis, factual, exploratory)
    - Decompose multi-part questions into sub-queries
    - Determine which topics/phases to prioritize
    - Select retrieval strategy
    """

    def __init__(
        self,
        llm_client,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """
        Initialize the planning agent.

        Args:
            llm_client: Groq client instance
            model: LLM model to use
            temperature: Sampling temperature (lower = more focused)
            max_tokens: Maximum response tokens
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def plan(
        self,
        question: str,
        available_topics: List[str],
        available_phases: List[str]
    ) -> PlanningResult:
        """
        Create a retrieval plan for a complex query.

        Args:
            question: The user's question
            available_topics: List of available topic categories
            available_phases: List of available phases

        Returns:
            PlanningResult with intent, sub-queries, and strategy
        """
        try:
            user_prompt = PLANNING_USER_PROMPT.format(
                question=question,
                topics=", ".join(available_topics) if available_topics else "Not specified",
                phases=", ".join(available_phases) if available_phases else "Phase 1, Phase 2, Phase 3, Phase 4"
            )

            response = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            result = self._parse_response(content, question)
            logger.info(f"Planning complete: intent={result['intent_type']}, sub_queries={len(result['sub_queries'])}")
            return result

        except Exception as e:
            logger.error(f"Planning agent error: {e}")
            # Return fallback plan
            return self._create_fallback_plan(question)

    def _parse_response(self, content: str, original_question: str) -> PlanningResult:
        """Parse LLM response into PlanningResult."""
        try:
            # Try to extract JSON from response
            # Handle case where LLM wraps JSON in markdown code block
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            # Validate and extract fields
            intent_type = data.get("intent_type", "factual")
            if intent_type not in ["comparative", "synthesis", "factual", "exploratory"]:
                intent_type = "factual"

            sub_queries = data.get("sub_queries", [])
            if not sub_queries:
                sub_queries = [original_question]

            retrieval_strategy = data.get("retrieval_strategy", "multi_query")
            if retrieval_strategy not in ["single", "multi_query", "expanded"]:
                retrieval_strategy = "multi_query"

            return PlanningResult(
                intent_type=intent_type,
                sub_queries=sub_queries[:5],  # Limit to 5 sub-queries
                retrieval_strategy=retrieval_strategy,
                topics_identified=data.get("topics_identified", []),
                reasoning=data.get("reasoning", "")
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse planning response: {e}")
            return self._create_fallback_plan(original_question)

    def _create_fallback_plan(self, question: str) -> PlanningResult:
        """Create a fallback plan when parsing fails."""
        return PlanningResult(
            intent_type="factual",
            sub_queries=[question],
            retrieval_strategy=RetrievalStrategy.SINGLE.value,
            topics_identified=[],
            reasoning="Fallback plan due to parsing error"
        )


def create_planning_agent(
    llm_client,
    config: Optional[Dict[str, Any]] = None
) -> PlanningAgent:
    """
    Factory function to create a planning agent.

    Args:
        llm_client: Groq client instance
        config: Optional configuration overrides

    Returns:
        Configured PlanningAgent instance
    """
    config = config or {}
    return PlanningAgent(
        llm_client=llm_client,
        model=config.get("model", "llama-3.3-70b-versatile"),
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 500)
    )
