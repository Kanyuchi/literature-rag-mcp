"""Query Classifier for Agentic RAG

Classifies queries by complexity using regex + heuristics (no LLM needed).
Fast classification enables appropriate pipeline routing.

Complexity Levels:
    - Simple: Single topic, definition, summary, <15 words → Fast path
    - Medium: Multi-source needed, moderate depth → Standard RAG
    - Complex: Comparative, synthesis, multi-part, >40 words → Full agentic pipeline
"""

import re
from typing import Tuple, List, Set
from dataclasses import dataclass

from .state import QueryComplexity


@dataclass
class ClassificationResult:
    """Result of query classification."""
    complexity: QueryComplexity
    confidence: float  # 0-1 confidence in classification
    signals: List[str]  # What signals led to this classification
    word_count: int
    topic_count: int


class QueryClassifier:
    """
    Classifies query complexity without using an LLM.

    Uses pattern matching and heuristics to quickly route queries
    to the appropriate pipeline.
    """

    # Patterns indicating SIMPLE queries
    SIMPLE_PATTERNS = [
        r"^what is\b",
        r"^define\b",
        r"^summarize\b",
        r"^summary of\b",
        r"^list\b",
        r"^list papers\b",
        r"^how many\b",
        r"^when did\b",
        r"^who (?:is|are|was|were)\b",
        r"^which (?:paper|author|study)\b",
        r"^explain\b",
        r"^describe briefly\b",
        r"^give (?:me )?(?:a )?(?:brief )?(overview|summary)",
    ]

    # Patterns indicating COMPLEX queries
    COMPLEX_PATTERNS = [
        r"\bcompare\b.*\b(?:and|with|to|versus|vs)\b",
        r"\bcontrast\b",
        r"\bdifferences? between\b",
        r"\bsimilarities? between\b",
        r"\bsynthesize\b",
        r"\bsynthesis\b",
        r"\bcritically (?:analyze|examine|evaluate)\b",
        r"\bevaluate the\b.*\b(?:evidence|literature|research)\b",
        r"\bhow does\b.*\brelate to\b",
        r"\bwhat (?:is|are) the (?:relationship|connection)s? between\b",
        r"\bintegrate\b.*\bfindings\b",
        r"\bacross (?:multiple|different|various)\b",
        r"\bmultiple (?:perspectives|viewpoints|approaches)\b",
        r"\bon one hand\b.*\bon the other\b",
        r"\bpros and cons\b",
        r"\badvantages and disadvantages\b",
        r"\bstrengths and weaknesses\b",
        r"\bover time\b.*\bhow\b",
        r"\bevolution of\b",
        r"\bhistorical (?:development|trajectory|analysis)\b",
    ]

    # Keywords indicating academic/research topics
    TOPIC_KEYWORDS = {
        # Regional/Geographic
        "ruhr", "ruhrgebiet", "ruhr valley", "nrw", "north rhine-westphalia",
        "germany", "german", "european", "europe",
        # Economic concepts
        "deindustrialization", "tertiarization", "structural change",
        "business formation", "entrepreneurship", "startups",
        "economic transition", "regional development", "resilience",
        # Institutional
        "varieties of capitalism", "voc", "institutional", "institutions",
        "lme", "cme", "coordinated market", "liberal market",
        # Methodology
        "spatial panel", "econometrics", "qualitative", "quantitative",
        "mixed methods", "case study", "inkar",
        # Phases
        "phase 1", "phase 2", "phase 3", "phase 4",
        "theoretical", "sectoral", "context", "methodology",
        # COVID/Policy
        "covid", "pandemic", "just transition", "policy",
    }

    def __init__(
        self,
        simple_max_words: int = 15,
        complex_min_topics: int = 3,
        complex_min_words: int = 40
    ):
        """
        Initialize the classifier.

        Args:
            simple_max_words: Max words for simple classification
            complex_min_topics: Min topics for complex classification
            complex_min_words: Min words that suggest complexity
        """
        self.simple_max_words = simple_max_words
        self.complex_min_topics = complex_min_topics
        self.complex_min_words = complex_min_words

        # Compile patterns
        self.simple_regex = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self.complex_regex = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]

    def classify(self, query: str, force_complex: bool = False) -> ClassificationResult:
        """
        Classify query complexity.

        Args:
            query: The user's question
            force_complex: If True, always return COMPLEX (for deep_analysis mode)

        Returns:
            ClassificationResult with complexity level and metadata
        """
        if force_complex:
            return ClassificationResult(
                complexity=QueryComplexity.COMPLEX,
                confidence=1.0,
                signals=["deep_analysis_forced"],
                word_count=len(query.split()),
                topic_count=self._count_topics(query)
            )

        signals: List[str] = []
        query_lower = query.lower().strip()
        word_count = len(query.split())
        topic_count = self._count_topics(query_lower)
        question_count = query.count("?")

        # Check for complex patterns first (higher priority)
        complex_matches = self._match_patterns(query_lower, self.complex_regex)
        if complex_matches:
            signals.extend([f"complex_pattern:{p}" for p in complex_matches])

        # Check for simple patterns
        simple_matches = self._match_patterns(query_lower, self.simple_regex)
        if simple_matches:
            signals.extend([f"simple_pattern:{p}" for p in simple_matches])

        # Word count signals
        if word_count <= self.simple_max_words:
            signals.append(f"short_query:{word_count}_words")
        elif word_count >= self.complex_min_words:
            signals.append(f"long_query:{word_count}_words")

        # Topic count signals
        if topic_count >= self.complex_min_topics:
            signals.append(f"multi_topic:{topic_count}_topics")

        # Multiple questions signal complexity
        if question_count > 1:
            signals.append(f"multiple_questions:{question_count}")

        # Determine complexity
        complexity, confidence = self._determine_complexity(
            signals, word_count, topic_count, question_count,
            bool(complex_matches), bool(simple_matches)
        )

        return ClassificationResult(
            complexity=complexity,
            confidence=confidence,
            signals=signals,
            word_count=word_count,
            topic_count=topic_count
        )

    def _count_topics(self, query: str) -> int:
        """Count distinct academic topics mentioned in query."""
        found_topics: Set[str] = set()
        query_words = set(query.lower().split())

        for keyword in self.TOPIC_KEYWORDS:
            # Check for multi-word keywords
            if " " in keyword:
                if keyword in query.lower():
                    found_topics.add(keyword)
            # Single word keywords
            elif keyword in query_words:
                found_topics.add(keyword)

        return len(found_topics)

    def _match_patterns(
        self,
        query: str,
        patterns: List[re.Pattern]
    ) -> List[str]:
        """Return list of pattern names that matched."""
        matches = []
        for i, pattern in enumerate(patterns):
            if pattern.search(query):
                # Extract a readable pattern name
                pattern_str = pattern.pattern[:30].replace("\\b", "").replace("^", "")
                matches.append(pattern_str)
        return matches

    def _determine_complexity(
        self,
        signals: List[str],
        word_count: int,
        topic_count: int,
        question_count: int,
        has_complex_pattern: bool,
        has_simple_pattern: bool
    ) -> Tuple[QueryComplexity, float]:
        """
        Determine final complexity based on signals.

        Returns:
            Tuple of (complexity, confidence)
        """
        # Strong complex signals
        if has_complex_pattern:
            if topic_count >= 2 or question_count > 1 or word_count >= 30:
                return QueryComplexity.COMPLEX, 0.9
            return QueryComplexity.COMPLEX, 0.75

        # Strong simple signals
        if has_simple_pattern and word_count <= self.simple_max_words:
            if topic_count <= 1:
                return QueryComplexity.SIMPLE, 0.9
            return QueryComplexity.SIMPLE, 0.7

        # Multiple topics suggest medium or complex
        if topic_count >= self.complex_min_topics:
            return QueryComplexity.COMPLEX, 0.7

        # Long queries with multiple questions
        if word_count >= self.complex_min_words and question_count > 1:
            return QueryComplexity.COMPLEX, 0.75

        # Very short queries are simple
        if word_count <= 10:
            return QueryComplexity.SIMPLE, 0.8

        # Short-ish queries with simple patterns
        if word_count <= self.simple_max_words and has_simple_pattern:
            return QueryComplexity.SIMPLE, 0.75

        # Default to medium for ambiguous cases
        if word_count <= 25:
            return QueryComplexity.MEDIUM, 0.6

        # Longer queries default to medium-high complexity
        if word_count <= self.complex_min_words:
            return QueryComplexity.MEDIUM, 0.65

        # Very long queries are likely complex
        return QueryComplexity.COMPLEX, 0.6


# Singleton instance for convenience
_classifier: QueryClassifier = None


def get_classifier(
    simple_max_words: int = 15,
    complex_min_topics: int = 3,
    complex_min_words: int = 40
) -> QueryClassifier:
    """Get or create classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier(
            simple_max_words=simple_max_words,
            complex_min_topics=complex_min_topics,
            complex_min_words=complex_min_words
        )
    return _classifier
