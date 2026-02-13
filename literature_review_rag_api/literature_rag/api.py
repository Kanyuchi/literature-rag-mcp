"""FastAPI Server for Literature Review RAG

Provides REST API endpoints for querying academic literature.
Adapted from personality RAG API patterns.
"""

import logging
import os
import uuid
import time
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, status, Depends, Request, UploadFile, File, Form, BackgroundTasks
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
    PaperInfo,
    UploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DeleteResponse,
    AgenticChatResponse,
    PipelineStatsResponse,
)
from .indexer import DocumentIndexer, create_indexer_from_rag
from .tasks import task_store, TaskStatus, process_pdf_task, run_async_task
from .storage import get_storage
from .database import get_db_session, DefaultDocumentCRUD, User
from .auth import get_current_user_optional, get_current_user
from .logging_utils import setup_logging, request_id_ctx
from .rate_limiter import create_rate_limiter, RateLimitMiddleware
from .quotas import get_user_quota_summary, get_quota_service
from .pool import get_pool
from .worker import get_worker
from .utils import sanitize_filename

# Setup structured logging (default INFO, overridden after config load)
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: LiteratureReviewRAG = None
document_indexer: DocumentIndexer = None
config: Any = None
groq_client = None  # Groq LLM client
agentic_pipeline = None  # Agentic RAG pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for RAG system initialization."""
    global rag_system, document_indexer, config, groq_client

    logger.info("Initializing Literature Review RAG system...")

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded from {config.storage.indices_path}")
        setup_logging(getattr(config.advanced, "log_level", "INFO"))

        # Initialize RAG system
        rag_system = LiteratureReviewRAG(
            chroma_path=config.storage.indices_path,
            config={
                "collection_name": config.storage.collection_name,
                "cache_metadata": config.storage.cache_metadata,
                "metadata_cache_path": config.storage.metadata_cache_path,
                "expand_queries": config.retrieval.expand_queries,
                "max_expansions": config.retrieval.max_expansions,
                "use_reranking": config.retrieval.use_reranking,
                "reranker_model": config.retrieval.reranker_model,
                "rerank_top_k": config.retrieval.rerank_top_k,
                "normalization_enable": config.normalization.enable,
                "term_maps": config.normalization.term_maps,
                # Hybrid search settings
                "use_hybrid": config.retrieval.use_hybrid,
                "hybrid_method": config.retrieval.hybrid_method,
                "hybrid_weight": config.retrieval.hybrid_weight,
                "bm25_candidates": config.retrieval.bm25_candidates,
                "bm25_use_stemming": config.retrieval.bm25_use_stemming,
                "bm25_min_token_length": config.retrieval.bm25_min_token_length,
                "indices_path": config.storage.indices_path,
                "openai_model": config.embedding.openai_model,
                "language_filter_enabled": config.retrieval.language_filter_enabled,
                "language_filter_fallback": config.retrieval.language_filter_fallback,
            },
            openai_model=config.embedding.openai_model
        )

        if rag_system.is_ready():
            logger.info(f"RAG system ready! Loaded {rag_system.collection.count()} chunks")
        else:
            logger.warning("RAG system initialized but no index found. Run build_index.py first.")

        # Initialize document indexer for uploads
        if rag_system.collection:
            document_indexer = create_indexer_from_rag(rag_system, {
                "extraction": vars(config.extraction) if hasattr(config.extraction, '__dict__') else {},
                "chunking": vars(config.chunking) if hasattr(config.chunking, '__dict__') else {},
                "embedding": vars(config.embedding) if hasattr(config.embedding, '__dict__') else {}
            })
            logger.info("Document indexer initialized for PDF uploads")

        # Create upload directories if they don't exist
        upload_config = getattr(config, 'upload', None)
        if upload_config and hasattr(upload_config, 'temp_path'):
            Path(upload_config.temp_path).mkdir(parents=True, exist_ok=True)
            Path(upload_config.storage_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Upload directories ready: {upload_config.storage_path}")

        # Log auth mode
        if config.auth.require_auth:
            logger.info("🔒 Authentication REQUIRED for API access")
        else:
            logger.warning("⚠️  Authentication DISABLED - set auth.require_auth=true for production!")

        # Initialize Groq client if API key is available
        if config.llm.groq_api_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=config.llm.groq_api_key)
                logger.info(f"Groq LLM initialized with model: {config.llm.model}")

                # Initialize agentic pipeline if enabled
                if config.agentic.enabled:
                    from .agentic import AgenticRAGPipeline
                    agentic_config = {
                        "classification": {
                            "simple_max_words": config.agentic.classification.simple_max_words,
                            "complex_min_topics": config.agentic.classification.complex_min_topics,
                            "complex_min_words": config.agentic.classification.complex_min_words,
                        },
                        "thresholds": {
                            "evaluation_sufficient": config.agentic.thresholds.evaluation_sufficient,
                            "citation_accuracy_min": config.agentic.thresholds.citation_accuracy_min,
                            "max_retrieval_retries": config.agentic.thresholds.max_retrieval_retries,
                            "max_regeneration_retries": config.agentic.thresholds.max_regeneration_retries,
                        },
                        "agents": config.agentic.agents,
                    }
                    agentic_pipeline = AgenticRAGPipeline(rag_system, groq_client, agentic_config)
                    logger.info("Agentic RAG pipeline initialized")

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


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_ctx.set(request_id)
    start_time = time.time()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        status_code = response.status_code if response else 500
        if response is not None:
            response.headers["X-Request-Id"] = request_id
        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms
            }
        )
        request_id_ctx.reset(token)


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
# CSRF PROTECTION (double-submit cookie)
# ============================================================================
_CSRF_COOKIE_NAME = "csrf_token"
_CSRF_HEADER_NAME = "X-CSRF-Token"
_CSRF_EXEMPT_PATHS = {
    # With /api/ prefix (direct backend access)
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/oauth/google/callback",
    "/api/auth/oauth/github/callback",
    # Without /api/ prefix (nginx strips it)
    "/auth/login",
    "/auth/register",
    "/auth/oauth/google/callback",
    "/auth/oauth/github/callback",
}


@app.middleware("http")
async def csrf_protection_middleware(request: Request, call_next):
    # Ensure CSRF cookie exists
    csrf_cookie = request.cookies.get(_CSRF_COOKIE_NAME)
    if not csrf_cookie:
        csrf_cookie = secrets.token_urlsafe(32)

    # Enforce on unsafe methods
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if request.url.path not in _CSRF_EXEMPT_PATHS:
            # Skip CSRF check if Bearer token auth is used (tokens are CSRF-safe)
            auth_header = request.headers.get("authorization", "")
            if not auth_header.lower().startswith("bearer "):
                # Only enforce CSRF for cookie-based auth
                header_token = request.headers.get(_CSRF_HEADER_NAME)
                if not header_token or header_token != csrf_cookie:
                    # Include CORS headers in error response
                    origin = request.headers.get("origin", "")
                    cors_headers = {}
                    if origin in config_temp.api.cors_origins:
                        cors_headers = {
                            "Access-Control-Allow-Origin": origin,
                            "Access-Control-Allow-Credentials": "true",
                        }
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"error": "CSRF validation failed"},
                        headers=cors_headers
                    )

    response = await call_next(request)

    # Set CSRF cookie (non-HttpOnly so JS can read)
    response.set_cookie(
        key=_CSRF_COOKIE_NAME,
        value=csrf_cookie,
        httponly=False,
        secure=os.getenv("AUTH_COOKIE_SECURE", "false").lower() in ("true", "1", "yes"),
        samesite=os.getenv("AUTH_COOKIE_SAMESITE", "lax"),
        domain=os.getenv("AUTH_COOKIE_DOMAIN") or None,
        path="/"
    )
    return response

# Setup rate limiting (if enabled)
_rate_limiter = create_rate_limiter(config_temp.api.rate_limit)
if _rate_limiter:
    app.add_middleware(RateLimitMiddleware, rate_limiter=_rate_limiter)

# Include routers
from .routers.auth import router as auth_router
from .routers.jobs import router as jobs_router
from .routers.chats import router as chats_router
from .routers.insights import router as insights_router
from .routers.graph import router as graph_router
from .routers.data_sources import router as data_sources_router

app.include_router(auth_router)
app.include_router(jobs_router)
app.include_router(chats_router)
app.include_router(insights_router)
app.include_router(graph_router)
app.include_router(data_sources_router)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def require_auth_if_configured(user=Depends(get_current_user_optional)):
    """Enforce auth when configured.

    Auth is REQUIRED by default. To disable, explicitly set:
    - YAML: auth.require_auth: false
    - Or env: AUTH_REQUIRE_AUTH=false
    """
    # Use loaded config if available, otherwise use temp config
    active_config = config if config is not None else config_temp
    require_auth = getattr(active_config.auth, "require_auth", True)  # Default True

    if require_auth and user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


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
        year_range = stats.get("year_range", {"min": 0, "max": 0})

        return {
            "total_papers": stats.get("total_papers", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "phases": stats.get("papers_by_phase", {}),
            "topics": stats.get("papers_by_topic", {}),
            "year_range": year_range
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )


@app.get("/api/quota")
async def get_quota_usage(user: User = Depends(get_current_user)):
    """
    Get current quota usage for the authenticated user.

    Returns usage statistics and limits for:
    - Documents uploaded
    - Knowledge bases created
    - Storage used
    - API calls today

    Requires authentication.
    """
    try:
        summary = get_user_quota_summary(user.id)
        return {
            "user_id": user.id,
            "email": user.email,
            **summary
        }
    except Exception as e:
        logger.error(f"Quota retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quota retrieval failed: {str(e)}"
        )


@app.get("/api/papers", dependencies=[Depends(require_auth_if_configured)])
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


@app.get("/api/search", dependencies=[Depends(require_auth_if_configured)])
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
    start_time = time.time()

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

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "search_metrics",
            extra={
                "event": "search_metrics",
                "query": query,
                "n_results": n_results,
                "returned": len(search_results),
                "phase_filter": phase_filter,
                "topic_filter": topic_filter,
                "duration_ms": duration_ms
            }
        )
        return search_results
    except Exception as e:
        logger.error(f"API search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/answer", dependencies=[Depends(require_auth_if_configured)])
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
                f"Based on {len(sources)} relevant sources from the document collection:",
                "The sources address this topic through multiple perspectives.",
                "Key findings from the sources include relevant insights on your question.",
                "For detailed analysis, please review the cited sources below."
            ]
        }
    except Exception as e:
        logger.error(f"API answer failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/chat", response_model=AgenticChatResponse, dependencies=[Depends(require_auth_if_configured)])
async def api_chat_with_llm(
    question: str,
    n_sources: int = 5,
    phase_filter: str = None,
    topic_filter: str = None,
    deep_analysis: bool = False
):
    """
    Chat with the literature using adaptive agentic RAG.

    Routes queries based on complexity:
    - Simple queries (<15 words, definitions): Fast path, ~2s
    - Medium queries (standard questions): Standard RAG, ~4s
    - Complex queries (comparative, synthesis): Full agentic pipeline with
      planning, evaluation, and validation agents, ~6-10s

    Parameters:
    - question: Your research question
    - n_sources: Number of sources to retrieve (default 5)
    - phase_filter: Filter by research phase (e.g., "Phase 1")
    - topic_filter: Filter by topic category (e.g., "Business Formation")
    - deep_analysis: Force complex pipeline for thorough analysis

    Returns:
    - answer: Generated response with citations
    - sources: List of cited sources
    - complexity: Query complexity classification
    - pipeline_stats: Execution statistics (LLM calls, time, validation status)
    """
    check_rag_ready()

    if groq_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured. Set GROQ_API_KEY in .env file."
        )

    try:
        start_time = time.time()
        filters = {}
        if phase_filter:
            filters["phase_filter"] = phase_filter
        if topic_filter:
            filters["topic_filter"] = topic_filter

        # Use agentic pipeline if available and enabled
        if agentic_pipeline is not None and config.agentic.enabled:
            result = agentic_pipeline.run(
                question=question,
                n_sources=n_sources,
                phase_filter=phase_filter,
                topic_filter=topic_filter,
                deep_analysis=deep_analysis
            )
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "chat_metrics",
                extra={
                    "event": "chat_metrics",
                    "complexity": result.get("complexity"),
                    "llm_calls": result["pipeline_stats"].get("llm_calls"),
                    "retrieval_attempts": result["pipeline_stats"].get("retrieval_attempts"),
                    "duration_ms": duration_ms,
                    "deep_analysis": deep_analysis
                }
            )

            return AgenticChatResponse(
                question=question,
                answer=result["answer"],
                sources=result["sources"],
                complexity=result["complexity"],
                pipeline_stats=PipelineStatsResponse(
                    llm_calls=result["pipeline_stats"]["llm_calls"],
                    retrieval_attempts=result["pipeline_stats"]["retrieval_attempts"],
                    validation_passed=result["pipeline_stats"]["validation_passed"],
                    total_time_ms=result["pipeline_stats"]["total_time_ms"],
                    evaluation_scores=result["pipeline_stats"].get("evaluation_scores"),
                    retries=result["pipeline_stats"]["retries"]
                ),
                model=config.llm.model,
                filters_applied=filters
            )

        # Fallback to original simple chat if agentic pipeline not available
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

        # Build citation reference for LLM
        citation_refs = []
        for s in sources:
            citation_refs.append(f"[{s['citation_number']}] = {s['authors']} ({s['year']}). {s['title']}")
        citation_guide = "\n".join(citation_refs)

        # Build prompt for LLM
        system_prompt = """You are an expert assistant.

Answer questions based ONLY on the provided document context. Respond in the same language as the user's question unless they explicitly request another language. When citing sources, use the author-date format with citation number, like: "According to Author (Year) [1], ..." or "...as noted by Author (Year) [2]".

Be precise, concise, and synthesize information across multiple sources when relevant. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation."""

        user_prompt = f"""Based on the following document excerpts, please answer this question:

QUESTION: {question}

CITATION KEY:
{citation_guide}

DOCUMENT CONTEXT:
{context}

Please provide a well-structured answer using author-date citations (e.g., "According to Smith (2020) [1], ..."). Always include the author name and year when citing."""

        # Call Groq API
        start_time = time.time()

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
        total_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "chat_metrics",
            extra={
                "event": "chat_metrics",
                "complexity": "medium",
                "llm_calls": 1,
                "retrieval_attempts": 1,
                "duration_ms": total_time_ms,
                "deep_analysis": deep_analysis
            }
        )
        return AgenticChatResponse(
            question=question,
            answer=answer,
            sources=sources,
            complexity="medium",  # Fallback always uses medium pipeline
            pipeline_stats=PipelineStatsResponse(
                llm_calls=1,
                retrieval_attempts=1,
                validation_passed=None,
                total_time_ms=total_time_ms,
                evaluation_scores=None,
                retries={"retrieval": 0, "generation": 0}
            ),
            model=config.llm.model,
            filters_applied=filters
        )

    except Exception as e:
        logger.error(f"API chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/related", dependencies=[Depends(require_auth_if_configured)])
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


@app.post("/api/synthesis", dependencies=[Depends(require_auth_if_configured)])
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

        # Add pool statistics
        try:
            pool_stats = get_pool().get_stats()
            stats["pool"] = pool_stats
        except Exception as e:
            stats["pool"] = {"error": str(e)}

        # Add worker statistics
        try:
            worker_stats = get_worker().get_stats()
            stats["worker"] = worker_stats
        except Exception as e:
            stats["worker"] = {"error": str(e)}

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

@app.get("/healthz")
async def healthz():
    """Lightweight health check for container liveness."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_auth_if_configured)])
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


@app.post("/context", response_model=ContextResponse, dependencies=[Depends(require_auth_if_configured)])
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


@app.post("/synthesis", response_model=SynthesisResponse, dependencies=[Depends(require_auth_if_configured)])
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


@app.post("/related", response_model=RelatedResponse, dependencies=[Depends(require_auth_if_configured)])
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


@app.get("/papers", response_model=PapersResponse, dependencies=[Depends(require_auth_if_configured)])
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


@app.get("/gaps", response_model=Dict[str, Any], dependencies=[Depends(require_auth_if_configured)])
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
# UPLOAD ENDPOINTS
# ============================================================================

def check_upload_enabled():
    """Check if upload functionality is enabled."""
    upload_config = getattr(config, 'upload', None)
    if not upload_config or not getattr(upload_config, 'enabled', False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload functionality is not enabled"
        )
    if document_indexer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document indexer not initialized"
        )
        if getattr(upload_config, 's3_only', False):
            try:
                get_storage()
            except Exception as e:
                raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"S3 storage is required but not configured: {e}"
            )


def get_phase_names() -> Dict[str, str]:
    """Get mapping of phase to phase name from config."""
    phases_config = config.data.phases if hasattr(config.data, 'phases') else []
    return {p.get('name', ''): p.get('full_name', '') for p in phases_config}


@app.post("/api/upload", response_model=UploadResponse, dependencies=[Depends(require_auth_if_configured)])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    phase: str = Form(..., description="Phase (e.g., 'Phase 1', 'Phase 2')"),
    topic: str = Form(..., description="Topic category (e.g., 'Business Formation')")
):
    """
    Upload and index a PDF file.

    The PDF will be processed, chunked, embedded, and added to the knowledge base.
    Requires phase and topic selection for proper organization.
    """
    check_rag_ready()
    check_upload_enabled()

    upload_config = config.upload

    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    original_filename = file.filename
    safe_filename = sanitize_filename(original_filename)
    file_ext = Path(safe_filename).suffix.lower()
    allowed_extensions = getattr(upload_config, 'allowed_extensions', ['.pdf'])
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Check file size
    max_size = getattr(upload_config, 'max_file_size', 52428800)  # 50MB default
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )

    # Reset file position
    await file.seek(0)

    # Generate unique ID for this upload
    upload_id = str(uuid.uuid4())[:8]

    # Save to temp directory
    temp_path = Path(upload_config.temp_path)
    temp_file = temp_path / f"{upload_id}_{safe_filename}"

    try:
        # Write file to temp location
        with open(temp_file, "wb") as f:
            f.write(contents)

        logger.info(f"Saved temp file: {temp_file}")

        # Get phase name from config
        phase_names = get_phase_names()
        phase_name = phase_names.get(phase, phase)

        # Index the PDF
        result = document_indexer.index_pdf(
            pdf_path=temp_file,
            phase=phase,
            phase_name=phase_name,
            topic_category=topic
        )

        if result["success"]:
            # Upload to S3 (single storage backend)
            storage = get_storage()
            try:
                with open(temp_file, "rb") as f:
                    storage_key = storage.upload_pdf(
                        job_id="default",
                        phase=phase,
                        topic=topic,
                        filename=safe_filename,
                        file_content=f
                    )
                if temp_file.exists():
                    temp_file.unlink()
                logger.info("Uploaded to S3 and removed temp file")
                # Persist default document record
                db = get_db_session()
                try:
                    metadata = result.get("metadata") or {}
                    authors_value = metadata.get("authors")
                    if isinstance(authors_value, list):
                        authors_value = ", ".join(authors_value)
                    DefaultDocumentCRUD.create(
                        db=db,
                        doc_id=result["doc_id"],
                        filename=original_filename,
                        storage_key=storage_key,
                        file_size=len(contents),
                        title=metadata.get("title"),
                        authors=authors_value,
                        year=metadata.get("year"),
                        phase=phase,
                        topic_category=topic,
                        doi=metadata.get("doi"),
                        total_pages=metadata.get("total_pages")
                    )
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                # Roll back indexed chunks if storage fails
                if result.get("doc_id"):
                    rag_system.delete_by_doc_id(result["doc_id"])
                if temp_file.exists():
                    temp_file.unlink()
                return UploadResponse(
                    success=False,
                    doc_id=result.get("doc_id"),
                    filename=file.filename,
                    chunks_indexed=0,
                    metadata=None,
                    error="Upload failed while storing file in S3"
                )

            return UploadResponse(
                success=True,
                doc_id=result["doc_id"],
                filename=original_filename,
                chunks_indexed=result["chunks_indexed"],
                metadata=result["metadata"],
                error=None
            )
        else:
            # Cleanup temp file on failure
            if temp_file.exists():
                temp_file.unlink()

            return UploadResponse(
                success=False,
                doc_id=None,
                filename=original_filename,
                chunks_indexed=0,
                metadata=None,
                error=result.get("error", "Unknown error during indexing")
            )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/upload/async", dependencies=[Depends(require_auth_if_configured)])
async def upload_pdf_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    phase: str = Form(..., description="Phase (e.g., 'Phase 1', 'Phase 2')"),
    topic: str = Form(..., description="Topic category (e.g., 'Business Formation')")
):
    """
    Upload a PDF file for asynchronous processing.

    Returns a task_id immediately. Use GET /api/upload/{task_id}/status to check progress.
    Recommended for large PDFs where you want to track progress.
    """
    check_rag_ready()
    check_upload_enabled()

    upload_config = config.upload

    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    original_filename = file.filename
    safe_filename = sanitize_filename(original_filename)
    file_ext = Path(safe_filename).suffix.lower()
    allowed_extensions = getattr(upload_config, 'allowed_extensions', ['.pdf'])
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Check file size
    max_size = getattr(upload_config, 'max_file_size', 52428800)
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )

    # Generate unique ID and save to temp
    upload_id = str(uuid.uuid4())[:8]
    temp_path = Path(upload_config.temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    temp_file = temp_path / f"{upload_id}_{safe_filename}"

    try:
        # Write file to temp location
        with open(temp_file, "wb") as f:
            f.write(contents)

        logger.info(f"Saved temp file for async processing: {temp_file}")

        # Create task
        task = task_store.create_task(
            filename=original_filename,
            phase=phase,
            topic=topic,
            temp_file_path=str(temp_file)
        )

        # Get phase name from config
        phase_names = get_phase_names()
        phase_name = phase_names.get(phase, phase)

        # Schedule background processing
        background_tasks.add_task(
            run_async_task,
            process_pdf_task(
                task_id=task.task_id,
                indexer=document_indexer,
                temp_file_path=temp_file,
                phase=phase,
                phase_name=phase_name,
                topic=topic,
                storage_path=None,
                filename=original_filename,
                storage_filename=safe_filename,
                owner_id="default"
            )
        )

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "message": "Upload accepted, processing started",
            "filename": original_filename,
            "phase": phase,
            "topic": topic
        }

    except Exception as e:
        logger.error(f"Async upload failed: {e}")
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/upload/{task_id}/status", dependencies=[Depends(require_auth_if_configured)])
async def get_upload_status(task_id: str):
    """
    Get the status of an async upload task.

    Returns current progress, status, and result (if completed).
    Poll this endpoint to track upload progress.
    """
    task = task_store.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )

    return task.to_dict()


@app.get("/api/upload/tasks", dependencies=[Depends(require_auth_if_configured)])
async def list_upload_tasks(limit: int = 20):
    """
    List recent upload tasks.

    Returns the most recent upload tasks with their status.
    """
    tasks = task_store.list_tasks(limit=limit)
    return {
        "total": len(tasks),
        "tasks": [t.to_dict() for t in tasks]
    }


@app.get("/api/documents", response_model=DocumentListResponse, dependencies=[Depends(require_auth_if_configured)])
async def list_documents(
    phase_filter: str = None,
    topic_filter: str = None,
    limit: int = 100
):
    """
    List all indexed documents.

    Returns document metadata for all papers in the knowledge base.
    """
    check_rag_ready()

    try:
        # Prefer normalized document records from DB
        db = get_db_session()
        try:
            db_docs = DefaultDocumentCRUD.list_all(db, limit=limit)
        finally:
            db.close()

        if db_docs:
            all_docs = [{
                "doc_id": d.doc_id,
                "title": d.title,
                "authors": d.authors,
                "year": d.year,
                "phase": d.phase,
                "topic_category": d.topic_category,
                "filename": d.filename,
                "total_pages": d.total_pages,
                "doi": d.doi,
                "abstract": None
            } for d in db_docs]
        else:
            # Fallback to RAG metadata if DB has no records yet
            all_docs = rag_system.list_documents()

        # Apply filters
        filtered_docs = []
        for doc in all_docs:
            if phase_filter and doc.get("phase") != phase_filter:
                continue
            if topic_filter and doc.get("topic_category") != topic_filter:
                continue

            filtered_docs.append(DocumentInfo(
                doc_id=doc.get("doc_id"),
                title=doc.get("title"),
                authors=doc.get("authors"),
                year=doc.get("year"),
                phase=doc.get("phase"),
                topic_category=doc.get("topic_category"),
                filename=doc.get("filename"),
                total_pages=doc.get("total_pages"),
                doi=doc.get("doi"),
                abstract=doc.get("abstract")
            ))

            if len(filtered_docs) >= limit:
                break

        return DocumentListResponse(
            total=len(filtered_docs),
            documents=filtered_docs
        )

    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/documents/{doc_id}/download")
async def get_document_download_url(
    doc_id: str,
    _user=Depends(require_auth_if_configured)
):
    """
    Get a presigned URL to download a default collection document.
    """
    db = get_db_session()
    try:
        default_doc = DefaultDocumentCRUD.get_by_doc_id(db, doc_id)
        if not default_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        storage = get_storage()
        url = storage.get_presigned_url(default_doc.storage_key)
        return {"doc_id": doc_id, "download_url": url}
    finally:
        db.close()


@app.delete("/api/documents/{doc_id}", response_model=DeleteResponse, dependencies=[Depends(require_auth_if_configured)])
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the knowledge base.

    This permanently removes the document from the index.
    The original PDF file is NOT deleted from storage.
    """
    check_rag_ready()

    try:
        result = rag_system.delete_by_doc_id(doc_id)
        # Remove default document record + S3 object if present
        db = get_db_session()
        try:
            default_doc = DefaultDocumentCRUD.get_by_doc_id(db, doc_id)
            if default_doc:
                try:
                    storage = get_storage()
                    storage.delete_pdf(default_doc.storage_key)
                except Exception as e:
                    logger.warning(f"Failed to delete S3 object: {e}")
                DefaultDocumentCRUD.delete(db, default_doc)
        finally:
            db.close()

        return DeleteResponse(
            success=result["success"],
            doc_id=doc_id,
            chunks_deleted=result.get("chunks_deleted", 0),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/upload/config", dependencies=[Depends(require_auth_if_configured)])
async def get_upload_config():
    """
    Get upload configuration and available options.

    Returns phases, topics, and upload limits for the UI.
    """
    upload_config = getattr(config, 'upload', None)

    # Get phases from config
    phases = []
    if hasattr(config.data, 'phases'):
        for p in config.data.phases:
            phases.append({
                "name": p.get('name'),
                "full_name": p.get('full_name'),
                "description": p.get('description')
            })

    # Get existing topics from the collection
    topics = set()
    if rag_system and rag_system.collection:
        try:
            all_data = rag_system.collection.get(include=["metadatas"])
            for meta in all_data.get("metadatas", []):
                topic = meta.get("topic_category")
                if topic:
                    topics.add(topic)
        except Exception as e:
            logger.warning(f"Could not fetch existing topics: {e}")

    return {
        "enabled": upload_config is not None and getattr(upload_config, 'enabled', False),
        "s3_only": getattr(upload_config, 's3_only', False) if upload_config else False,
        "max_file_size": getattr(upload_config, 'max_file_size', 52428800) if upload_config else 52428800,
        "allowed_extensions": getattr(upload_config, 'allowed_extensions', ['.pdf']) if upload_config else ['.pdf'],
        "phases": phases,
        "existing_topics": sorted(list(topics))
    }


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
