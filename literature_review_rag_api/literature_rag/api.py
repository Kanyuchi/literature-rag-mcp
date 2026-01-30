"""FastAPI Server for Literature Review RAG

Provides REST API endpoints for querying academic literature.
Adapted from personality RAG API patterns.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .literature_rag import LiteratureReviewRAG
from .config import load_config
from .models import (
    QueryRequest,
    QueryResponse,
    ContextRequest,
    ContextResponse,
    SynthesisRequest,
    SynthesisResponse,
    RelatedRequest,
    RelatedResponse,
    HealthResponse,
    PapersResponse,
    ErrorResponse,
    DocumentResult,
    PaperMetadata,
    PaperInfo
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: LiteratureReviewRAG = None
config: Any = None
groq_client = None  # Groq LLM client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for RAG system initialization."""
    global rag_system, config, groq_client

    logger.info("Initializing Literature Review RAG system...")

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded from {config.storage.indices_path}")

        # Initialize RAG system
        rag_system = LiteratureReviewRAG(
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

        if rag_system.is_ready():
            logger.info(f"RAG system ready! Loaded {rag_system.collection.count()} chunks")
        else:
            logger.warning("RAG system initialized but no index found. Run build_index.py first.")

        # Initialize Groq client if API key is available
        if config.llm.groq_api_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=config.llm.groq_api_key)
                logger.info(f"Groq LLM initialized with model: {config.llm.model}")
            except ImportError:
                logger.warning("Groq package not installed. Run: pip install groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        else:
            logger.info("Groq API key not configured. LLM chat endpoint will be disabled.")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

    finally:
        logger.info("Shutting down Literature Review RAG system")


# Create FastAPI app
config_temp = load_config()  # Load config for app metadata
def _api_key_guard(request: Request) -> None:
    """Enforce API key authentication when enabled in config."""
    active_config = config if config is not None else config_temp
    if not active_config.api.require_api_key:
        return
    expected_key = active_config.api.api_key
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key required but not configured"
        )
    provided_key = request.headers.get("x-api-key")
    if not provided_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided_key = auth_header[7:]
    if provided_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


app = FastAPI(
    title=config_temp.api.title,
    description=config_temp.api.description,
    version=config_temp.api.version,
    lifespan=lifespan,
    dependencies=[Depends(_api_key_guard)]
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key"
    }
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "API Key"
    }
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_temp.api.cors_origins,
    allow_credentials=config_temp.api.cors_credentials,
    allow_methods=config_temp.api.cors_methods,
    allow_headers=config_temp.api.cors_headers
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_rag_ready():
    """Check if RAG system is ready to serve requests."""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    if not rag_system.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG index not found. Run build_index.py to create the index."
        )


def _split_comma_list(value: Any) -> list[str] | None:
    """Normalize comma-separated metadata into a list for API responses."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    return [str(value)]


def convert_to_document_results(results: dict) -> list[DocumentResult]:
    """Convert ChromaDB results to DocumentResult list."""
    documents = []

    for i in range(len(results["documents"][0])):
        content = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        metadata = dict(metadata)
        metadata["authors"] = _split_comma_list(metadata.get("authors"))
        metadata["keywords"] = _split_comma_list(metadata.get("keywords"))
        metadata["geographic_focus"] = _split_comma_list(metadata.get("geographic_focus"))
        distance = results["distances"][0][i]

        # Convert distance to similarity score (1 - distance for cosine)
        score = 1 - distance

        doc_result = DocumentResult(
            content=content,
            metadata=PaperMetadata(**metadata),
            score=round(score, 4)
        )
        documents.append(doc_result)

    return documents


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Literature Review RAG API",
        "version": "1.0.0",
        "description": "Academic literature search system for German regional economic transitions",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/api/stats")
async def get_stats():
    """
    Get collection statistics for the webapp dashboard.

    Returns total papers, chunks, phases, topics, and year range.
    """
    check_rag_ready()

    try:
        stats = rag_system.get_stats()

        return {
            "total_papers": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "phases": stats.get("papers_by_phase", {}),
            "topics": stats.get("papers_by_topic", {}),
            "year_range": {
                "min": stats.get("year_range", {}).get("min", 0),
                "max": stats.get("year_range", {}).get("max", 0)
            }
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )


@app.get("/api/papers")
async def api_list_papers(
    phase_filter: str = None,
    topic_filter: str = None,
    limit: int = 100
):
    """
    List papers for the webapp (simplified format).
    """
    check_rag_ready()

    try:
        all_data = rag_system.collection.get(include=["metadatas"])

        papers_dict = {}
        for metadata in all_data["metadatas"]:
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id not in papers_dict:
                if phase_filter and metadata.get("phase") != phase_filter:
                    continue
                if topic_filter and metadata.get("topic_category") != topic_filter:
                    continue

                papers_dict[doc_id] = {
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Untitled"),
                    "authors": metadata.get("authors"),
                    "year": metadata.get("year"),
                    "phase": metadata.get("phase"),
                    "topic": metadata.get("topic_category"),
                    "source": metadata.get("filename", "")
                }

                if len(papers_dict) >= limit:
                    break

        return {
            "total": len(papers_dict),
            "papers": list(papers_dict.values())
        }
    except Exception as e:
        logger.error(f"API list papers failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/search")
async def api_semantic_search(
    query: str,
    n_results: int = 10,
    phase_filter: str = None,
    topic_filter: str = None,
    year_min: int = None,
    year_max: int = None
):
    """
    Semantic search endpoint for the webapp.
    """
    check_rag_ready()

    try:
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter
        if year_min:
            filters["year_min"] = year_min
        if year_max:
            filters["year_max"] = year_max

        results = rag_system.query(
            question=query,
            n_results=n_results,
            **filters
        )

        search_results = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance

            search_results.append({
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title", "Untitled"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "phase": metadata.get("phase"),
                "topic": metadata.get("topic_category"),
                "chunk_text": results["documents"][0][i],
                "relevance_score": round(score, 4)
            })

        return search_results
    except Exception as e:
        logger.error(f"API search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/answer")
async def api_answer_with_citations(
    question: str,
    n_sources: int = 5,
    phase_filter: str = None,
    topic_filter: str = None
):
    """
    Get answer sources with citations for the chat interface.
    """
    check_rag_ready()

    try:
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter

        results = rag_system.query(
            question=question,
            n_results=n_sources,
            **filters
        )

        sources = []
        bibliography = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]

            citation = {
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title", "Untitled"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "chunk_text": results["documents"][0][i]
            }
            sources.append(citation)

            # Build bibliography entry
            authors = metadata.get("authors", "Unknown")
            year = metadata.get("year", "n.d.")
            title = metadata.get("title", "Untitled")
            bibliography.append(f"{authors} ({year}). {title}")

        return {
            "sources": sources,
            "bibliography": bibliography,
            "suggested_structure": [
                f"Based on {len(sources)} relevant sources from the literature collection:",
                "The research addresses this topic through multiple perspectives.",
                "Key findings from the literature include relevant insights on your question.",
                "For detailed analysis, please review the cited sources below."
            ]
        }
    except Exception as e:
        logger.error(f"API answer failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/chat")
async def api_chat_with_llm(
    question: str,
    n_sources: int = 5,
    phase_filter: str = None,
    topic_filter: str = None
):
    """
    Chat with the literature using Groq LLM (Llama 3.3 70B).

    Retrieves relevant context from the literature and generates
    an AI-powered response with citations.
    """
    check_rag_ready()

    if groq_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured. Set GROQ_API_KEY in .env file."
        )

    try:
        # Build filters
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter

        # Retrieve relevant context
        results = rag_system.query(
            question=question,
            n_results=n_sources,
            **filters
        )

        # Build context string with citations
        context_parts = []
        sources = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            chunk_text = results["documents"][0][i]

            authors = metadata.get("authors", "Unknown")
            year = metadata.get("year", "n.d.")
            title = metadata.get("title", "Untitled")

            citation = f"[{i+1}] {authors} ({year})"
            context_parts.append(f"{citation}:\n{chunk_text}\n")

            sources.append({
                "citation_number": i + 1,
                "authors": authors,
                "year": year,
                "title": title,
                "doc_id": metadata.get("doc_id")
            })

        context = "\n".join(context_parts)

        # Build prompt for LLM
        system_prompt = """You are an expert academic research assistant specializing in German regional economic transitions, institutional economics, and the Ruhr Valley transformation.

Answer questions based ONLY on the provided academic literature context. Always cite sources using the citation numbers provided (e.g., [1], [2]).

Be precise, academic in tone, and synthesize information across multiple sources when relevant. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation."""

        user_prompt = f"""Based on the following academic literature excerpts, please answer this question:

QUESTION: {question}

ACADEMIC LITERATURE CONTEXT:
{context}

Please provide a well-structured answer with citations."""

        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )

        answer = chat_completion.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "model": config.llm.model,
            "filters_applied": filters
        }

    except Exception as e:
        logger.error(f"API chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/related")
async def api_find_related(
    paper_id: str,
    n_results: int = 5
):
    """
    Find related papers by embedding similarity.
    """
    check_rag_ready()

    try:
        results = rag_system.find_related_papers(
            paper_id=paper_id,
            n_results=n_results
        )

        related_papers = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            related_papers.append({
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title", "Untitled"),
                "authors": metadata.get("authors"),
                "year": metadata.get("year"),
                "phase": metadata.get("phase"),
                "topic": metadata.get("topic_category")
            })

        return {
            "source_paper_id": paper_id,
            "related_papers": related_papers
        }
    except Exception as e:
        logger.error(f"API related failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/synthesis")
async def api_synthesis_query(request: SynthesisRequest):
    """
    Multi-topic synthesis query for the webapp.
    """
    check_rag_ready()

    try:
        results = rag_system.synthesis_query(
            question=request.query,
            topics=request.topics,
            n_per_topic=request.n_per_topic
        )

        return results
    except Exception as e:
        logger.error(f"API synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system statistics."""
    try:
        if rag_system is None:
            return HealthResponse(
                status="unhealthy",
                ready=False,
                stats={"error": "RAG system not initialized"}
            )

        stats = rag_system.get_stats()

        return HealthResponse(
            status="healthy" if rag_system.is_ready() else "initializing",
            ready=rag_system.is_ready(),
            stats=stats
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            ready=False,
            stats={"error": str(e)}
        )


@app.post("/query", response_model=QueryResponse)
async def query_literature(request: QueryRequest):
    """
    Query academic literature with filters.

    Returns relevant paper chunks with metadata.
    """
    check_rag_ready()

    try:
        # Extract filters
        filters = {}
        if request.phase_filter:
            filters["phase_filter"] = request.phase_filter.value
        if request.topic_filter:
            filters["topic_filter"] = request.topic_filter
        if request.year_min:
            filters["year_min"] = request.year_min
        if request.year_max:
            filters["year_max"] = request.year_max
        if request.methodology_filter:
            filters["methodology_filter"] = request.methodology_filter
        if request.geographic_filter:
            filters["geographic_filter"] = request.geographic_filter.value
        if request.research_type_filter:
            filters["research_type_filter"] = request.research_type_filter.value

        # Query RAG system
        results = rag_system.query(
            question=request.query,
            n_results=request.n_results,
            **filters
        )

        # Get detected terms
        _, detected_terms = rag_system.normalize_query(request.query)

        # Convert to response format
        documents = convert_to_document_results(results)

        return QueryResponse(
            query=request.query,
            n_results=len(documents),
            documents=documents,
            detected_terms=detected_terms,
            filters_applied=filters
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Get LLM-ready context with citations.

    Returns formatted string with paper citations and content.
    """
    check_rag_ready()

    try:
        # Extract filters
        filters = {}
        if request.phase_filter:
            filters["phase_filter"] = request.phase_filter.value
        if request.topic_filter:
            filters["topic_filter"] = request.topic_filter
        if request.year_min:
            filters["year_min"] = request.year_min
        if request.year_max:
            filters["year_max"] = request.year_max
        if request.methodology_filter:
            filters["methodology_filter"] = request.methodology_filter
        if request.geographic_filter:
            filters["geographic_filter"] = request.geographic_filter.value
        if request.research_type_filter:
            filters["research_type_filter"] = request.research_type_filter.value

        # Get formatted context
        context = rag_system.get_context(
            question=request.query,
            n_results=request.n_results,
            **filters
        )

        return ContextResponse(
            query=request.query,
            context=context,
            n_results=request.n_results
        )

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(e)}"
        )


@app.post("/synthesis", response_model=SynthesisResponse)
async def synthesis_query(request: SynthesisRequest):
    """
    Query multiple topic categories and synthesize results.

    Like the personality RAG's council_query but for academic topics.
    """
    check_rag_ready()

    try:
        # Query multiple topics
        results = rag_system.synthesis_query(
            question=request.query,
            topics=request.topics,
            n_per_topic=request.n_per_topic
        )

        return SynthesisResponse(
            query=request.query,
            topics=request.topics,
            results=results
        )

    except Exception as e:
        logger.error(f"Synthesis query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis query failed: {str(e)}"
        )


@app.post("/related", response_model=RelatedResponse)
async def find_related_papers(request: RelatedRequest):
    """
    Find papers related to a given paper via embedding similarity.
    """
    check_rag_ready()

    try:
        results = rag_system.find_related_papers(
            paper_id=request.paper_id,
            n_results=request.n_results
        )

        # Convert to response format
        related_papers = convert_to_document_results(results)

        return RelatedResponse(
            paper_id=request.paper_id,
            n_results=len(related_papers),
            related_papers=related_papers
        )

    except Exception as e:
        logger.error(f"Related papers query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Related papers query failed: {str(e)}"
        )


@app.get("/papers", response_model=PapersResponse)
async def list_papers(
    phase_filter: str = None,
    topic_filter: str = None,
    year_min: int = None,
    year_max: int = None,
    limit: int = 100
):
    """
    List all papers with optional filters.

    Returns paper metadata without content.

    Parameters:
        phase_filter: Filter by phase (e.g., "Phase 1", "Phase 2")
        topic_filter: Filter by topic category (e.g., "Business Formation")
        year_min: Minimum publication year
        year_max: Maximum publication year
        limit: Maximum number of papers to return (default 100)
    """
    check_rag_ready()

    try:
        # Get all metadata (we'll filter in memory for simplicity)
        all_data = rag_system.collection.get(
            include=["metadatas"]
        )

        # Group by doc_id to get unique papers
        papers_dict = {}
        for metadata in all_data["metadatas"]:
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id not in papers_dict:
                # Apply filters
                if phase_filter and metadata.get("phase") != phase_filter:
                    continue
                if topic_filter and metadata.get("topic_category") != topic_filter:
                    continue
                if year_min and metadata.get("year", 0) < year_min:
                    continue
                if year_max and metadata.get("year", 9999) > year_max:
                    continue

                papers_dict[doc_id] = PaperInfo(
                    doc_id=doc_id,
                    title=metadata.get("title"),
                    authors=_split_comma_list(metadata.get("authors")),
                    year=metadata.get("year"),
                    doi=metadata.get("doi"),
                    phase=metadata.get("phase"),
                    phase_name=metadata.get("phase_name"),
                    topic_category=metadata.get("topic_category"),
                    filename=metadata.get("filename", ""),
                    abstract=metadata.get("abstract"),
                    total_pages=metadata.get("total_pages"),
                    file_path=metadata.get("file_path")
                )

                if len(papers_dict) >= limit:
                    break

        papers = list(papers_dict.values())

        return PapersResponse(
            total_papers=len(papers),
            papers=papers
        )

    except Exception as e:
        logger.error(f"List papers failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List papers failed: {str(e)}"
        )


@app.get("/gaps", response_model=Dict[str, Any])
async def analyze_gaps(focus_area: str = None):
    """
    Identify research gaps in the literature.

    Analyzes topic coverage, methodology diversity, and temporal distribution.
    """
    check_rag_ready()

    try:
        stats = rag_system.get_stats()

        # Simple gap analysis
        gaps = {
            "focus_area": focus_area or "all",
            "analysis": {
                "topic_coverage": {},
                "methodology_gaps": [],
                "temporal_gaps": []
            }
        }

        # Analyze topic coverage
        papers_by_topic = stats.get("papers_by_topic", {})
        min_coverage = config.advanced.gap_analysis.get("min_topic_coverage", 3)

        for topic, count in papers_by_topic.items():
            if count < min_coverage:
                gaps["analysis"]["topic_coverage"][topic] = {
                    "current_papers": count,
                    "recommended_papers": min_coverage,
                    "gap": min_coverage - count
                }

        # Placeholder for more sophisticated gap analysis
        gaps["analysis"]["recommendations"] = [
            "Consider adding more papers on underrepresented topics",
            "Diversify methodological approaches in certain phases",
            "Include more recent publications (post-2020)"
        ]

        return gaps

    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gap analysis failed: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    # Load config for port
    cfg = load_config()

    uvicorn.run(
        "literature_rag.api:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=True
    )
