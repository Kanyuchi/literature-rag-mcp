# Literature Review RAG - Claude Context

## Workflow Rules

### Auto-Commit Policy
**IMPORTANT:** After making any file changes (edits, additions, or deletions), Claude MUST:
1. Stage the changed files with `git add <specific-files>`
2. Commit with a descriptive message
3. Push to origin/main

This ensures all changes are tracked in version control automatically without requiring explicit user requests.

---

## Project Overview
Academic literature RAG system for German regional economic transitions research. Contains 13,578 chunks from 83 papers indexed with BAAI/bge-base-en-v1.5 embeddings in ChromaDB.

## Key Paths
- **API Code**: `literature_review_rag_api/literature_rag/`
- **MCP Server**: `literature_review_rag_api/literature_rag/mcp_server.py`
- **Config**: `literature_review_rag_api/config/literature_config.yaml`
- **Indices**: `literature_review_rag_api/indices/`
- **Virtual Env**: `literature_review_rag_api/venv/`

## MCP Server Tools
The `literature-rag` MCP server exposes 7 tools:
- `semantic_search` - Search with filters (phase, topic, year)
- `get_context_for_llm` - Formatted context with citations
- `list_papers` - List available papers
- `find_related_papers` - Find similar papers by embedding
- `get_collection_stats` - Collection statistics
- `answer_with_citations` - Sources with bibliography
- `synthesis_query` - Multi-topic comparative analysis

## Running the MCP Server
```bash
cd /Users/fadzie/Desktop/lit_rag/literature_review_rag_api
source venv/bin/activate
python -m literature_rag.mcp_server
```

## Running the FastAPI Server
```bash
cd /Users/fadzie/Desktop/lit_rag/literature_review_rag_api
source venv/bin/activate
python -m literature_rag.api
```
API runs on http://localhost:8001
