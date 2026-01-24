# literature-rag-mcp

MCP server for academic literature RAG system focused on German regional economic transitions research. Integrates with Claude Desktop and Claude Code.

## Features
- **13,578 indexed chunks** from 83 academic papers
- **Semantic search** using BAAI/bge-base-en-v1.5 embeddings
- **ChromaDB** vector storage
- **FastAPI** REST endpoint (port 8001)
- **MCP Server** for Claude integration

## MCP Server Setup

### Claude Desktop
Config: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "literature-rag": {
      "command": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/venv/bin/python",
      "args": ["-m", "literature_rag.mcp_server"],
      "cwd": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api",
      "env": {
        "PYTHONPATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api",
        "INDICES_PATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/indices",
        "CONFIG_PATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/config/literature_config.yaml"
      }
    }
  }
}
```

### Claude Code
Config: `~/.claude.json` (add to `mcpServers` object)

```json
"literature-rag": {
  "type": "stdio",
  "command": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/venv/bin/python",
  "args": ["-m", "literature_rag.mcp_server"],
  "env": {
    "PYTHONPATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api",
    "INDICES_PATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/indices",
    "CONFIG_PATH": "/Users/fadzie/Desktop/lit_rag/literature_review_rag_api/config/literature_config.yaml"
  }
}
```

### Key Configuration Notes
1. **Absolute paths required** - Relative paths fail in sandboxed environments
2. **PYTHONPATH** - Required for module discovery
3. **INDICES_PATH & CONFIG_PATH** - Override defaults to ensure correct file access
4. **Restart required** - After config changes, restart Claude Desktop/Code

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Search papers with phase/topic/year filters |
| `get_context_for_llm` | Pre-formatted context with citations |
| `list_papers` | Discover available papers in collection |
| `find_related_papers` | Find similar papers via embeddings |
| `get_collection_stats` | Collection statistics and metadata |
| `answer_with_citations` | Structured sources with bibliography |
| `synthesis_query` | Multi-topic comparative analysis |

## Quick Start

```bash
# Install dependencies
cd literature_review_rag_api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fastmcp

# Run MCP server
python -m literature_rag.mcp_server

# Or run FastAPI server
python -m literature_rag.api
```

## Project Structure
```
literature-rag-mcp/
├── literature_review_rag_api/
│   ├── literature_rag/
│   │   ├── mcp_server.py      # MCP server (FastMCP)
│   │   ├── api.py             # FastAPI server
│   │   ├── literature_rag.py  # Core RAG engine
│   │   └── config.py          # Configuration
│   ├── config/
│   │   └── literature_config.yaml
│   ├── indices/               # ChromaDB storage
│   └── venv/
└── README.md
```
