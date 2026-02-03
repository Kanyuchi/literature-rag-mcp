"""Validation Agent for Agentic RAG

Detects hallucinations and citation errors in generated responses.
Ensures factual accuracy and proper source attribution.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from ..state import ValidationResult, RetrievedChunk
from ..prompts import (
    VALIDATION_SYSTEM_PROMPT,
    VALIDATION_USER_PROMPT,
    format_context_for_prompt
)

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validates generated responses for hallucinations.

    Responsibilities:
    - Verify all citations reference actual sources
    - Detect fabricated facts or statistics
    - Ensure claims are supported by provided context
    - Flag misattributed findings
    """

    def __init__(
        self,
        llm_client,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 500,
        citation_threshold: float = 0.8
    ):
        """
        Initialize the validation agent.

        Args:
            llm_client: Groq client instance
            model: LLM model to use
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum response tokens
            citation_threshold: Minimum citation accuracy to pass
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.citation_threshold = citation_threshold

    def validate(
        self,
        question: str,
        response: str,
        chunks: List[RetrievedChunk]
    ) -> Tuple[ValidationResult, bool]:
        """
        Validate a generated response for hallucinations.

        Args:
            question: The original question
            response: The generated response to validate
            chunks: Source chunks used to generate the response

        Returns:
            Tuple of (ValidationResult, passed)
        """
        if not response or not chunks:
            return (
                ValidationResult(
                    is_valid=False,
                    citation_accuracy=0.0,
                    issues=["Empty response or no sources"],
                    hallucination_detected=True
                ),
                False
            )

        # First, run quick heuristic checks
        heuristic_issues = self._heuristic_validation(response, chunks)

        try:
            context = format_context_for_prompt(chunks)
            user_prompt = VALIDATION_USER_PROMPT.format(
                question=question,
                context=context,
                response=response
            )

            llm_response = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = llm_response.choices[0].message.content.strip()
            result = self._parse_response(content, heuristic_issues)

            passed = (
                result["is_valid"] and
                result["citation_accuracy"] >= self.citation_threshold and
                not result["hallucination_detected"]
            )

            logger.info(
                f"Validation complete: valid={result['is_valid']}, "
                f"accuracy={result['citation_accuracy']:.2f}, passed={passed}"
            )

            return result, passed

        except Exception as e:
            logger.error(f"Validation agent error: {e}")
            # Return heuristic-based fallback
            return self._heuristic_result(heuristic_issues), len(heuristic_issues) == 0

    def _heuristic_validation(
        self,
        response: str,
        chunks: List[RetrievedChunk]
    ) -> List[str]:
        """Run quick heuristic checks for common issues."""
        issues = []

        # Extract author names and years from chunks
        valid_authors = set()
        valid_years = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            authors = meta.get("authors", "")
            if authors:
                # Extract last names
                for author in authors.split(","):
                    parts = author.strip().split()
                    if parts:
                        valid_authors.add(parts[-1].lower())
            year = meta.get("year")
            if year:
                valid_years.add(str(year))

        # Extract citations from response (e.g., "Smith (2020)" or "[1]")
        citation_pattern = r'([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?)\s*\((\d{4})\)'
        citations = re.findall(citation_pattern, response)

        for author, year in citations:
            author_lower = author.split()[0].lower()
            if author_lower not in valid_authors and author_lower not in ["according", "based", "per"]:
                # Check if it's a common word misidentified as author
                if not any(author_lower in va for va in valid_authors):
                    issues.append(f"Potentially fabricated citation: {author} ({year})")

        # Check for suspiciously specific statistics not in sources
        stat_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)'
        stats_in_response = re.findall(stat_pattern, response)
        chunk_text = " ".join(c.get("content", "") for c in chunks)

        for stat in stats_in_response:
            if stat not in chunk_text:
                issues.append(f"Unverified statistic: {stat}%")

        # Check for overgeneralizations
        overgen_patterns = [
            r"all (?:studies|research|papers|authors|evidence) (?:show|indicate|suggest|demonstrate)",
            r"(?:every|all) researcher",
            r"universal(?:ly)? agree",
            r"(?:always|never) the case",
        ]
        for pattern in overgen_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Potential overgeneralization detected")
                break

        return issues

    def _parse_response(
        self,
        content: str,
        heuristic_issues: List[str]
    ) -> ValidationResult:
        """Parse LLM response into ValidationResult."""
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

            # Combine LLM issues with heuristic issues
            all_issues = list(set(data.get("issues", []) + heuristic_issues))

            # Adjust citation accuracy based on heuristic findings
            citation_accuracy = self._clamp_score(data.get("citation_accuracy", 0.8))
            if heuristic_issues:
                # Penalize for heuristic issues
                penalty = len(heuristic_issues) * 0.1
                citation_accuracy = max(0.0, citation_accuracy - penalty)

            hallucination = data.get("hallucination_detected", False)
            if len(heuristic_issues) >= 2:
                hallucination = True

            return ValidationResult(
                is_valid=data.get("is_valid", True) and not hallucination,
                citation_accuracy=citation_accuracy,
                issues=all_issues,
                hallucination_detected=hallucination
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse validation response")
            return self._heuristic_result(heuristic_issues)

    def _heuristic_result(self, issues: List[str]) -> ValidationResult:
        """Create validation result from heuristic issues alone."""
        has_issues = len(issues) > 0
        return ValidationResult(
            is_valid=not has_issues,
            citation_accuracy=0.9 if not has_issues else 0.5,
            issues=issues,
            hallucination_detected=len(issues) >= 2
        )

    def _clamp_score(self, value: Any) -> float:
        """Clamp a value to [0, 1] range."""
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.5


def create_validation_agent(
    llm_client,
    config: Optional[Dict[str, Any]] = None
) -> ValidationAgent:
    """
    Factory function to create a validation agent.

    Args:
        llm_client: Groq client instance
        config: Optional configuration overrides

    Returns:
        Configured ValidationAgent instance
    """
    config = config or {}
    return ValidationAgent(
        llm_client=llm_client,
        model=config.get("model", "llama-3.3-70b-versatile"),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 500),
        citation_threshold=config.get("citation_accuracy_min", 0.8)
    )
