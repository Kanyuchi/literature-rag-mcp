"""Configuration Management for Literature Review RAG

Loads configuration from YAML file and environment variables.
Adapted from personality RAG configuration patterns.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Environment-based settings (overrides config file)."""

    # Data paths
    pdf_path: str = Field(default="/Users/fadzie/Desktop/lit_rag", env="PDF_PATH")
    indices_path: str = Field(default="./indices", env="INDICES_PATH")
    config_path: str = Field(default="./config/literature_config.yaml", env="CONFIG_PATH")

    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8001, env="API_PORT")
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # LLM API Keys
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")

    # Device
    device: str = Field(default="auto", env="DEVICE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database
    database_url: str = Field(default="sqlite:///./literature_rag.db", env="DATABASE_URL")

    # CORS
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000", env="CORS_ORIGINS")

    # AWS S3 Configuration
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    aws_region: Optional[str] = Field(default=None, env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")

    # JWT Authentication
    jwt_secret_key: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # OAuth
    google_client_id: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_SECRET")
    github_client_id: Optional[str] = Field(default=None, env="GITHUB_CLIENT_ID")
    github_client_secret: Optional[str] = Field(default=None, env="GITHUB_CLIENT_SECRET")
    oauth_redirect_base: str = Field(default="http://localhost:8001", env="OAUTH_REDIRECT_BASE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra env vars without error


@dataclass
class DataConfig:
    """Data source configuration."""
    pdf_path: str
    auto_detect_structure: bool = True
    phases: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExtractionConfig:
    """PDF extraction configuration."""
    use_section_detection: bool = True
    section_confidence_threshold: float = 0.7
    extract_sections: List[str] = field(default_factory=lambda: [
        "abstract", "introduction", "methods", "results", "discussion", "conclusion"
    ])
    extract_metadata: List[str] = field(default_factory=lambda: [
        "title", "authors", "year", "keywords", "doi", "abstract"
    ])
    fallback_to_full_text: bool = True
    max_pages_per_pdf: Optional[int] = None
    skip_references: bool = True
    extract_first_n_pages_for_metadata: int = 3


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""
    strategy: str = "section_aware"  # "section_aware", "fixed_size", "hybrid"
    section_sizes: Dict[str, int] = field(default_factory=lambda: {
        "abstract": 1500,
        "introduction": 2000,
        "methods": 2000,
        "results": 2000,
        "discussion": 2000,
        "conclusion": 1500
    })
    section_overlap: int = 300
    fixed_chunk_size: int = 1000
    fixed_chunk_overlap: int = 200
    hybrid_min_confidence: float = 0.7


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    normalize: bool = True
    device: str = "auto"
    batch_size: int = 32
    cache_folder: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    default_n_results: int = 5
    use_hybrid: bool = False
    hybrid_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    hybrid_weight: float = 0.7  # Dense weight when using weighted method
    bm25_candidates: int = 50  # Number of BM25 candidates to retrieve
    bm25_use_stemming: bool = True
    bm25_min_token_length: int = 2
    use_reranking: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 20
    expand_queries: bool = True
    max_expansions: int = 2
    distance_metric: str = "cosine"


@dataclass
class NormalizationConfig:
    """Query normalization configuration."""
    enable: bool = True
    term_maps: Dict[str, List[List[str]]] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Filter configuration."""
    enable_phase_filter: bool = True
    enable_topic_filter: bool = True
    enable_year_filter: bool = True
    enable_methodology_filter: bool = True
    enable_geographic_filter: bool = True
    enable_research_type_filter: bool = True
    valid_research_types: List[str] = field(default_factory=lambda: [
        "empirical", "theoretical", "case_study", "mixed_methods", "literature_review", "methodology"
    ])
    valid_geographic_focus: List[str] = field(default_factory=lambda: [
        "Germany", "Ruhr Valley", "North Rhine-Westphalia", "Europe", "Global", "Comparative"
    ])


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ])
    cors_credentials: bool = False
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_headers: List[str] = field(default_factory=lambda: ["Authorization", "Content-Type", "X-API-Key"])
    require_api_key: bool = False
    api_key: Optional[str] = None
    rate_limit: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "requests": 100,
        "window_seconds": 60
    })
    title: str = "Literature Review RAG API"
    description: str = "Academic literature search system"
    version: str = "1.0.0"


@dataclass
class StorageConfig:
    """Storage configuration."""
    indices_path: str = "./indices"
    collection_name: str = "literature_review_chunks"
    cache_metadata: bool = True
    metadata_cache_path: str = "./indices/metadata_cache.pkl"
    chroma_settings: Dict[str, Any] = field(default_factory=lambda: {
        "anonymized_telemetry": False,
        "allow_reset": True
    })


@dataclass
class LLMConfig:
    """LLM configuration for generating responses."""
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 2048
    groq_api_key: Optional[str] = None


@dataclass
class UploadConfig:
    """Upload configuration."""
    enabled: bool = True
    s3_only: bool = False
    max_file_size: int = 52428800  # 50MB default
    temp_path: str = "./uploads/temp"
    storage_path: str = "./uploads/pdfs"
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    cleanup_temp: bool = True
    processing_timeout: int = 300


@dataclass
class AuthConfig:
    """Authentication configuration.

    Auth is REQUIRED by default for security. To disable auth (e.g., for local dev),
    explicitly set `require_auth: false` in config or AUTH_REQUIRE_AUTH=false env var.
    """
    require_auth: bool = True  # Auth required by default - explicit opt-out needed
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    oauth_redirect_url: str = "http://localhost:5173/auth/callback"


@dataclass
class AdvancedConfig:
    """Advanced features configuration."""
    build_citation_network: bool = False
    enable_gap_analysis: bool = True
    gap_analysis: Dict[str, int] = field(default_factory=lambda: {
        "min_topic_coverage": 3,
        "methodology_diversity_threshold": 2
    })
    use_external_enrichment: bool = False
    semantic_scholar_api_key: Optional[str] = None
    log_level: str = "INFO"
    log_file: str = "./logs/literature_rag.log"
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class AgenticClassificationConfig:
    """Agentic query classification configuration."""
    simple_max_words: int = 15
    complex_min_topics: int = 3
    complex_min_words: int = 40


@dataclass
class AgenticThresholdsConfig:
    """Agentic pipeline thresholds."""
    evaluation_sufficient: float = 0.7
    citation_accuracy_min: float = 0.8
    max_retrieval_retries: int = 2
    max_regeneration_retries: int = 1


@dataclass
class AgenticAgentConfig:
    """Configuration for individual agents."""
    temperature: float = 0.2
    max_tokens: int = 500


@dataclass
class AgenticConfig:
    """Agentic RAG configuration."""
    enabled: bool = True
    classification: AgenticClassificationConfig = field(default_factory=AgenticClassificationConfig)
    thresholds: AgenticThresholdsConfig = field(default_factory=AgenticThresholdsConfig)
    agents: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "planning": {"temperature": 0.3, "max_tokens": 500},
        "evaluation": {"temperature": 0.1, "max_tokens": 300},
        "validation": {"temperature": 0.1, "max_tokens": 500},
        "generation": {"temperature": 0.2, "max_tokens": 2048}
    })


@dataclass
class LiteratureRAGConfig:
    """Complete configuration for Literature Review RAG system."""
    data: DataConfig
    extraction: ExtractionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    normalization: NormalizationConfig
    filters: FilterConfig
    api: APIConfig
    storage: StorageConfig
    upload: UploadConfig
    advanced: AdvancedConfig
    llm: LLMConfig
    auth: AuthConfig = field(default_factory=AuthConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    custom: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> LiteratureRAGConfig:
    """
    Load configuration from YAML file and environment variables.

    Environment variables override YAML settings.

    Args:
        config_path: Path to YAML config file (default: ./config/literature_config.yaml)

    Returns:
        LiteratureRAGConfig object with all settings
    """
    # Load environment settings
    env_settings = Settings()

    # Determine config file path
    if config_path is None:
        config_path = env_settings.config_path

    config_file = Path(config_path)

    # Load YAML config
    if config_file.exists():
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_file}. Using defaults.")
        yaml_config = {}

    # Build configuration objects with YAML + environment overrides
    config = LiteratureRAGConfig(
        data=_load_data_config(yaml_config.get("data", {}), env_settings),
        extraction=_load_extraction_config(yaml_config.get("extraction", {})),
        chunking=_load_chunking_config(yaml_config.get("chunking", {})),
        embedding=_load_embedding_config(yaml_config.get("embedding", {}), env_settings),
        retrieval=_load_retrieval_config(yaml_config.get("retrieval", {})),
        normalization=_load_normalization_config(yaml_config.get("normalization", {})),
        filters=_load_filter_config(yaml_config.get("filters", {})),
        api=_load_api_config(yaml_config.get("api", {}), env_settings),
        storage=_load_storage_config(yaml_config.get("storage", {}), env_settings),
        upload=_load_upload_config(yaml_config.get("upload", {})),
        advanced=_load_advanced_config(yaml_config.get("advanced", {}), env_settings),
        llm=_load_llm_config(yaml_config.get("llm", {}), env_settings),
        auth=_load_auth_config(yaml_config.get("auth", {}), env_settings),
        agentic=_load_agentic_config(yaml_config.get("agentic", {})),
        custom=yaml_config.get("custom", {})
    )

    logger.info("Configuration loaded successfully")
    return config


def _load_data_config(yaml_data: dict, env_settings: Settings) -> DataConfig:
    """Load data configuration with environment overrides."""
    return DataConfig(
        pdf_path=env_settings.pdf_path,  # Environment override
        auto_detect_structure=yaml_data.get("auto_detect_structure", True),
        phases=yaml_data.get("phases", [])
    )


def _load_extraction_config(yaml_extraction: dict) -> ExtractionConfig:
    """Load extraction configuration."""
    return ExtractionConfig(
        use_section_detection=yaml_extraction.get("use_section_detection", True),
        section_confidence_threshold=yaml_extraction.get("section_confidence_threshold", 0.7),
        extract_sections=yaml_extraction.get("extract_sections", [
            "abstract", "introduction", "methods", "results", "discussion", "conclusion"
        ]),
        extract_metadata=yaml_extraction.get("extract_metadata", [
            "title", "authors", "year", "keywords", "doi", "abstract"
        ]),
        fallback_to_full_text=yaml_extraction.get("fallback_to_full_text", True),
        max_pages_per_pdf=yaml_extraction.get("max_pages_per_pdf"),
        skip_references=yaml_extraction.get("skip_references", True),
        extract_first_n_pages_for_metadata=yaml_extraction.get("extract_first_n_pages_for_metadata", 3)
    )


def _load_chunking_config(yaml_chunking: dict) -> ChunkingConfig:
    """Load chunking configuration."""
    return ChunkingConfig(
        strategy=yaml_chunking.get("strategy", "section_aware"),
        section_sizes=yaml_chunking.get("section_sizes", {
            "abstract": 1500,
            "introduction": 2000,
            "methods": 2000,
            "results": 2000,
            "discussion": 2000,
            "conclusion": 1500
        }),
        section_overlap=yaml_chunking.get("section_overlap", 300),
        fixed_chunk_size=yaml_chunking.get("fixed_chunk_size", 1000),
        fixed_chunk_overlap=yaml_chunking.get("fixed_chunk_overlap", 200),
        hybrid_min_confidence=yaml_chunking.get("hybrid_min_confidence", 0.7)
    )


def _load_embedding_config(yaml_embedding: dict, env_settings: Settings) -> EmbeddingConfig:
    """Load embedding configuration with environment overrides."""
    return EmbeddingConfig(
        model=yaml_embedding.get("model", "BAAI/bge-base-en-v1.5"),
        dimension=yaml_embedding.get("dimension", 768),
        normalize=yaml_embedding.get("normalize", True),
        device=env_settings.device,  # Environment override
        batch_size=yaml_embedding.get("batch_size", 32),
        cache_folder=yaml_embedding.get("cache_folder")
    )


def _load_retrieval_config(yaml_retrieval: dict) -> RetrievalConfig:
    """Load retrieval configuration."""
    return RetrievalConfig(
        default_n_results=yaml_retrieval.get("default_n_results", 5),
        use_hybrid=yaml_retrieval.get("use_hybrid", False),
        hybrid_method=yaml_retrieval.get("hybrid_method", "rrf"),
        hybrid_weight=yaml_retrieval.get("hybrid_weight", 0.7),
        bm25_candidates=yaml_retrieval.get("bm25_candidates", 50),
        bm25_use_stemming=yaml_retrieval.get("bm25_use_stemming", True),
        bm25_min_token_length=yaml_retrieval.get("bm25_min_token_length", 2),
        use_reranking=yaml_retrieval.get("use_reranking", False),
        reranker_model=yaml_retrieval.get("reranker_model", "BAAI/bge-reranker-base"),
        rerank_top_k=yaml_retrieval.get("rerank_top_k", 20),
        expand_queries=yaml_retrieval.get("expand_queries", True),
        max_expansions=yaml_retrieval.get("max_expansions", 2),
        distance_metric=yaml_retrieval.get("distance_metric", "cosine")
    )


def _load_normalization_config(yaml_normalization: dict) -> NormalizationConfig:
    """Load normalization configuration."""
    return NormalizationConfig(
        enable=yaml_normalization.get("enable", True),
        term_maps=yaml_normalization.get("term_maps", {})
    )


def _load_filter_config(yaml_filters: dict) -> FilterConfig:
    """Load filter configuration."""
    return FilterConfig(
        enable_phase_filter=yaml_filters.get("enable_phase_filter", True),
        enable_topic_filter=yaml_filters.get("enable_topic_filter", True),
        enable_year_filter=yaml_filters.get("enable_year_filter", True),
        enable_methodology_filter=yaml_filters.get("enable_methodology_filter", True),
        enable_geographic_filter=yaml_filters.get("enable_geographic_filter", True),
        enable_research_type_filter=yaml_filters.get("enable_research_type_filter", True),
        valid_research_types=yaml_filters.get("valid_research_types", [
            "empirical", "theoretical", "case_study", "mixed_methods", "literature_review", "methodology"
        ]),
        valid_geographic_focus=yaml_filters.get("valid_geographic_focus", [
            "Germany", "Ruhr Valley", "North Rhine-Westphalia", "Europe", "Global", "Comparative"
        ])
    )


def _load_api_config(yaml_api: dict, env_settings: Settings) -> APIConfig:
    """Load API configuration with environment overrides."""
    # Parse CORS origins from environment (comma-separated) or use YAML/default
    if env_settings.cors_origins:
        cors_origins = [origin.strip() for origin in env_settings.cors_origins.split(",")]
    else:
        cors_origins = yaml_api.get("cors_origins", ["*"])

    return APIConfig(
        host=env_settings.api_host,  # Environment override
        port=env_settings.api_port,  # Environment override
        cors_origins=cors_origins,
        cors_credentials=yaml_api.get("cors_credentials", True),
        cors_methods=yaml_api.get("cors_methods", ["*"]),
        cors_headers=yaml_api.get("cors_headers", ["*"]),
        require_api_key=yaml_api.get("require_api_key", False),
        api_key=env_settings.api_key,  # Environment override
        rate_limit=yaml_api.get("rate_limit", {
            "enabled": False,
            "requests": 100,
            "window_seconds": 60
        }),
        title=yaml_api.get("title", "Literature Review RAG API"),
        description=yaml_api.get("description", "Academic literature search system"),
        version=yaml_api.get("version", "1.0.0")
    )


def _load_storage_config(yaml_storage: dict, env_settings: Settings) -> StorageConfig:
    """Load storage configuration with environment overrides."""
    return StorageConfig(
        indices_path=env_settings.indices_path,  # Environment override
        collection_name=yaml_storage.get("collection_name", "literature_review_chunks"),
        cache_metadata=yaml_storage.get("cache_metadata", True),
        metadata_cache_path=yaml_storage.get("metadata_cache_path", "./indices/metadata_cache.pkl"),
        chroma_settings=yaml_storage.get("chroma_settings", {
            "anonymized_telemetry": False,
            "allow_reset": True
        })
    )


def _load_advanced_config(yaml_advanced: dict, env_settings: Settings) -> AdvancedConfig:
    """Load advanced configuration with environment overrides."""
    return AdvancedConfig(
        build_citation_network=yaml_advanced.get("build_citation_network", False),
        enable_gap_analysis=yaml_advanced.get("enable_gap_analysis", True),
        gap_analysis=yaml_advanced.get("gap_analysis", {
            "min_topic_coverage": 3,
            "methodology_diversity_threshold": 2
        }),
        use_external_enrichment=yaml_advanced.get("use_external_enrichment", False),
        semantic_scholar_api_key=yaml_advanced.get("semantic_scholar_api_key"),
        log_level=env_settings.log_level,  # Environment override
        log_file=yaml_advanced.get("log_file", "./logs/literature_rag.log"),
        parallel_processing=yaml_advanced.get("parallel_processing", True),
        max_workers=yaml_advanced.get("max_workers", 4)
    )


def _load_llm_config(yaml_llm: dict, env_settings: Settings) -> LLMConfig:
    """Load LLM configuration with environment overrides."""
    return LLMConfig(
        provider=yaml_llm.get("provider", "groq"),
        model=yaml_llm.get("model", "llama-3.3-70b-versatile"),
        temperature=yaml_llm.get("temperature", 0.1),
        max_tokens=yaml_llm.get("max_tokens", 2048),
        groq_api_key=env_settings.groq_api_key  # Environment override
    )


def _load_upload_config(yaml_upload: dict) -> UploadConfig:
    """Load upload configuration."""
    return UploadConfig(
        enabled=yaml_upload.get("enabled", True),
        s3_only=yaml_upload.get("s3_only", False),
        max_file_size=yaml_upload.get("max_file_size", 52428800),
        temp_path=yaml_upload.get("temp_path", "./uploads/temp"),
        storage_path=yaml_upload.get("storage_path", "./uploads/pdfs"),
        allowed_extensions=yaml_upload.get("allowed_extensions", [".pdf"]),
        cleanup_temp=yaml_upload.get("cleanup_temp", True),
        processing_timeout=yaml_upload.get("processing_timeout", 300)
    )


def _load_auth_config(yaml_auth: dict, env_settings: Settings) -> AuthConfig:
    """Load auth configuration with environment overrides.

    Auth is REQUIRED by default. To disable, explicitly set require_auth: false
    in YAML config or AUTH_REQUIRE_AUTH=false environment variable.
    """
    # Check for environment variable override
    require_auth_env = os.getenv("AUTH_REQUIRE_AUTH")
    if require_auth_env is not None:
        require_auth = require_auth_env.lower() in ("true", "1", "yes")
    else:
        # Default to True (auth required) unless explicitly disabled in YAML
        require_auth = yaml_auth.get("require_auth", True)

    return AuthConfig(
        require_auth=require_auth,
        access_token_expire_minutes=env_settings.access_token_expire_minutes,
        refresh_token_expire_days=env_settings.refresh_token_expire_days,
        oauth_redirect_url=yaml_auth.get("oauth_redirect_url", "http://localhost:5173/auth/callback")
    )


def _load_agentic_config(yaml_agentic: dict) -> AgenticConfig:
    """Load agentic RAG configuration."""
    classification_data = yaml_agentic.get("classification", {})
    thresholds_data = yaml_agentic.get("thresholds", {})
    agents_data = yaml_agentic.get("agents", {})

    return AgenticConfig(
        enabled=yaml_agentic.get("enabled", True),
        classification=AgenticClassificationConfig(
            simple_max_words=classification_data.get("simple_max_words", 15),
            complex_min_topics=classification_data.get("complex_min_topics", 3),
            complex_min_words=classification_data.get("complex_min_words", 40)
        ),
        thresholds=AgenticThresholdsConfig(
            evaluation_sufficient=thresholds_data.get("evaluation_sufficient", 0.7),
            citation_accuracy_min=thresholds_data.get("citation_accuracy_min", 0.8),
            max_retrieval_retries=thresholds_data.get("max_retrieval_retries", 2),
            max_regeneration_retries=thresholds_data.get("max_regeneration_retries", 1)
        ),
        agents={
            "planning": agents_data.get("planning", {"temperature": 0.3, "max_tokens": 500}),
            "evaluation": agents_data.get("evaluation", {"temperature": 0.1, "max_tokens": 300}),
            "validation": agents_data.get("validation", {"temperature": 0.1, "max_tokens": 500}),
            "generation": agents_data.get("generation", {"temperature": 0.2, "max_tokens": 2048})
        }
    )
