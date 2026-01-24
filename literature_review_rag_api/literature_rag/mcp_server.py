"""MCP Server for Literature Review RAG

Exposes the Literature Review RAG system to Claude and other LLM clients
via the Model Context Protocol (MCP).
"""

from fastmcp import FastMCP
from typing import Optional, List
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Literature Review RAG")

# Initialize RAG system (singleton pattern)
_rag_system = None


def get_rag_system():
    """Get or initialize the RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        from literature_rag.literature_rag import LiteratureReviewRAG
        from literature_rag.config import load_config

        logger.info("Initializing Literature Review RAG system...")
        config = load_config()
        _rag_system = LiteratureReviewRAG(
            chroma_path=config.storage.indices_path,
            config={
                "device": config.embedding.device,
                "collection_name": config.storage.collection_name,
                "expand_queries": config.retrieval.expand_queries,
                "max_expansions": config.retrieval.max_expansions,
                "term_maps": config.normalization.term_maps
            },
            embedding_model=config.embedding.model
        )
        logger.info("Literature Review RAG system initialized successfully")
    return _rag_system


@mcp.tool()
def semantic_search(
    query: str,
    n_results: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> dict:
    """
    Search academic literature on German regional economic transitions.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 5)
        phase_filter: Filter by phase (Phase 1, Phase 2, Phase 3, Phase 4)
        topic_filter: Filter by topic category
        year_min: Minimum publication year
        year_max: Maximum publication year

    Returns:
        Matching paper chunks with metadata and relevance scores
    """
    rag = get_rag_system()

    # Build filter kwargs
    filters = {}
    if phase_filter:
        filters["phase_filter"] = phase_filter
    if topic_filter:
        filters["topic_filter"] = topic_filter
    if year_min:
        filters["year_min"] = year_min
    if year_max:
        filters["year_max"] = year_max

    results = rag.query(question=query, n_results=n_results, **filters)

    # Format results for LLM consumption
    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(1 - results["distances"][0][i], 4)
            })

    return {"query": query, "results": formatted, "total_results": len(formatted)}


@mcp.tool()
def get_context_for_llm(
    query: str,
    n_results: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None
) -> str:
    """
    Get formatted context with citations for answering questions about German regional transitions.

    Returns a pre-formatted string ready for LLM consumption with proper citations.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default 5)
        phase_filter: Filter by phase (Phase 1, Phase 2, Phase 3, Phase 4)
        topic_filter: Filter by topic category

    Returns:
        Formatted context string with citations
    """
    rag = get_rag_system()

    # Build filter kwargs
    filters = {"n_results": n_results}
    if phase_filter:
        filters["phase_filter"] = phase_filter
    if topic_filter:
        filters["topic_filter"] = topic_filter

    return rag.get_context(question=query, **filters)


@mcp.tool()
def list_papers(
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    limit: int = 50
) -> dict:
    """
    List available academic papers in the literature collection.

    Use this to discover what papers are available before searching.

    Args:
        phase_filter: Filter by phase (Phase 1, Phase 2, Phase 3, Phase 4)
        topic_filter: Filter by topic category
        limit: Maximum number of papers to return (default 50)

    Returns:
        Dictionary with total count and list of paper metadata
    """
    rag = get_rag_system()

    if not rag.collection:
        return {"total": 0, "papers": [], "error": "Collection not loaded"}

    all_data = rag.collection.get(include=["metadatas"])

    papers = {}
    for metadata in all_data["metadatas"]:
        doc_id = metadata.get("doc_id")
        if doc_id and doc_id not in papers:
            # Apply filters
            if phase_filter and metadata.get("phase") != phase_filter:
                continue
            if topic_filter and metadata.get("topic_category") != topic_filter:
                continue

            papers[doc_id] = {
                "doc_id": doc_id,
                "title": metadata.get("title"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "phase": metadata.get("phase"),
                "topic": metadata.get("topic_category")
            }
            if len(papers) >= limit:
                break

    return {"total": len(papers), "papers": list(papers.values())}


@mcp.tool()
def find_related_papers(paper_id: str, n_results: int = 5) -> dict:
    """
    Find papers related to a given paper via embedding similarity.

    Args:
        paper_id: The doc_id of the source paper
        n_results: Number of related papers to return (default 5)

    Returns:
        Dictionary with source paper ID and list of related papers
    """
    rag = get_rag_system()
    results = rag.find_related_papers(paper_id=paper_id, n_results=n_results)

    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            content = results["documents"][0][i]
            # Truncate content for readability
            if len(content) > 500:
                content = content[:500] + "..."

            formatted.append({
                "content": content,
                "metadata": results["metadatas"][0][i],
                "similarity": round(1 - results["distances"][0][i], 4)
            })

    return {"source_paper": paper_id, "related": formatted}


@mcp.tool()
def get_collection_stats() -> dict:
    """
    Get statistics about the literature collection.

    Returns paper counts, topic distribution, year range, etc.

    Returns:
        Dictionary with collection statistics
    """
    rag = get_rag_system()
    return rag.get_stats()


@mcp.tool()
def answer_with_citations(
    question: str,
    n_sources: int = 5,
    phase_filter: Optional[str] = None,
    topic_filter: Optional[str] = None
) -> dict:
    """
    Get relevant sources to answer a question with proper academic citations.

    Returns structured data with:
    - Retrieved passages with full citations
    - Suggested answer structure
    - Bibliography in academic format

    Args:
        question: The research question to answer
        n_sources: Number of sources to retrieve (default 5)
        phase_filter: Filter by phase (Phase 1, Phase 2, Phase 3, Phase 4)
        topic_filter: Filter by topic category

    Returns:
        Dictionary with sources, bibliography, and suggested structure
    """
    rag = get_rag_system()

    # Build filter kwargs
    filters = {}
    if phase_filter:
        filters["phase_filter"] = phase_filter
    if topic_filter:
        filters["topic_filter"] = topic_filter

    results = rag.query(question=question, n_results=n_sources, **filters)

    # Format with citations
    sources = []
    bibliography = []

    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            content = results["documents"][0][i]
            score = 1 - results["distances"][0][i]

            # Build citation
            authors = meta.get("authors", "Unknown")
            year = meta.get("year", "n.d.")
            title = meta.get("title", "Untitled")

            citation = f"{authors} ({year})"
            full_ref = f"{authors} ({year}). {title}."
            if meta.get("doi"):
                full_ref += f" https://doi.org/{meta.get('doi')}"

            sources.append({
                "citation": citation,
                "content": content,
                "relevance_score": round(score, 4),
                "metadata": {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "phase": meta.get("phase"),
                    "topic": meta.get("topic_category"),
                    "section": meta.get("section")
                }
            })

            if full_ref not in bibliography:
                bibliography.append(full_ref)

    return {
        "question": question,
        "sources": sources,
        "bibliography": bibliography,
        "suggested_structure": (
            "Use the retrieved sources to answer the question. "
            "Cite each source using (Author, Year) format. "
            "Include all used sources in a References section."
        )
    }


@mcp.tool()
def synthesis_query(
    question: str,
    topics: List[str],
    n_per_topic: int = 2
) -> dict:
    """
    Query multiple topic categories and get synthesized results.

    Useful for comparative analysis or getting perspectives from different research areas.

    Args:
        question: The research question to answer
        topics: List of topic categories to query (e.g., ["Business Formation", "Labor Markets"])
        n_per_topic: Number of results per topic (default 2)

    Returns:
        Dictionary mapping each topic to its context string
    """
    rag = get_rag_system()
    results = rag.synthesis_query(
        question=question,
        topics=topics,
        n_per_topic=n_per_topic
    )
    return {"question": question, "results_by_topic": results}


if __name__ == "__main__":
    mcp.run()
