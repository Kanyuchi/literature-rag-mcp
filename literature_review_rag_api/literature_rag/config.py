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

    # Device
    device: str = Field(default="auto", env="DEVICE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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
    use_reranking: bool = False
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
    advanced: AdvancedConfig
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
        advanced=_load_advanced_config(yaml_config.get("advanced", {}), env_settings),
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
        use_reranking=yaml_retrieval.get("use_reranking", False),
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
    return APIConfig(
        host=env_settings.api_host,  # Environment override
        port=env_settings.api_port,  # Environment override
        cors_origins=yaml_api.get("cors_origins", ["*"]),
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
