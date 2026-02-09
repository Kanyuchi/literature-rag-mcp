"""Unified Embedding Provider Interface

Provides a consistent interface for both HuggingFace (local) and OpenAI (API) embeddings.
Automatically selects the appropriate provider based on configuration and API key availability.

Usage:
    from literature_rag.embeddings import get_embeddings
    from literature_rag.config import load_config

    config = load_config()
    embeddings = get_embeddings(config.embedding)

    # Use for embedding queries
    query_vector = embeddings.embed_query("What is business formation?")

    # Use for embedding documents
    doc_vectors = embeddings.embed_documents(["doc1 text", "doc2 text"])
"""

import logging
import os
from typing import Optional, Union

import torch
from langchain_core.embeddings import Embeddings

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Embedding dimensions by model
EMBEDDING_DIMENSIONS = {
    # HuggingFace models
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dimension(config: EmbeddingConfig) -> int:
    """Get the embedding dimension for the configured provider and model.

    Args:
        config: EmbeddingConfig with provider and model settings

    Returns:
        Embedding dimension as integer
    """
    if config.provider == "openai":
        return EMBEDDING_DIMENSIONS.get(config.openai_model, 1536)
    else:
        return EMBEDDING_DIMENSIONS.get(config.model, config.dimension)


def get_embeddings(
    config: Union[EmbeddingConfig, dict],
    fallback_to_huggingface: bool = True
) -> Embeddings:
    """Get embedding model instance based on configuration.

    Automatically selects between HuggingFace and OpenAI based on:
    1. config.provider setting
    2. OPENAI_API_KEY availability (if provider is "openai")

    Args:
        config: EmbeddingConfig object or dict with embedding settings
        fallback_to_huggingface: If True, fall back to HuggingFace if OpenAI fails

    Returns:
        Embeddings instance (HuggingFaceEmbeddings or OpenAIEmbeddings)

    Raises:
        ValueError: If provider is invalid or required API key is missing
    """
    # Handle dict config (for backward compatibility)
    if isinstance(config, dict):
        config = _dict_to_embedding_config(config)

    provider = config.provider.lower()

    if provider == "openai":
        return _get_openai_embeddings(config, fallback_to_huggingface)
    elif provider == "huggingface":
        return _get_huggingface_embeddings(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'huggingface' or 'openai'.")


def _get_openai_embeddings(
    config: EmbeddingConfig,
    fallback_to_huggingface: bool = True
) -> Embeddings:
    """Get OpenAI embeddings instance.

    Args:
        config: EmbeddingConfig with OpenAI settings
        fallback_to_huggingface: Fall back to HuggingFace if OpenAI unavailable

    Returns:
        OpenAIEmbeddings or HuggingFaceEmbeddings (fallback)
    """
    # Check for API key
    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        if fallback_to_huggingface:
            logger.warning(
                "OpenAI API key not found. Falling back to HuggingFace embeddings. "
                "Set OPENAI_API_KEY environment variable for faster embeddings."
            )
            return _get_huggingface_embeddings(config)
        else:
            raise ValueError(
                "OpenAI API key required for OpenAI embeddings. "
                "Set OPENAI_API_KEY environment variable or config.openai_api_key."
            )

    try:
        from langchain_openai import OpenAIEmbeddings

        model = config.openai_model or "text-embedding-3-small"
        dimension = EMBEDDING_DIMENSIONS.get(model, 1536)

        logger.info(f"Initializing OpenAI embeddings: {model} ({dimension} dimensions)")

        embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            # OpenAI's embedding API handles batching internally
        )

        logger.info("OpenAI embeddings initialized successfully")
        return embeddings

    except ImportError as e:
        if fallback_to_huggingface:
            logger.warning(
                f"langchain-openai not installed: {e}. "
                "Falling back to HuggingFace embeddings. "
                "Install with: pip install langchain-openai"
            )
            return _get_huggingface_embeddings(config)
        else:
            raise ImportError(
                "langchain-openai required for OpenAI embeddings. "
                "Install with: pip install langchain-openai"
            ) from e

    except Exception as e:
        if fallback_to_huggingface:
            logger.warning(
                f"Failed to initialize OpenAI embeddings: {e}. "
                "Falling back to HuggingFace embeddings."
            )
            return _get_huggingface_embeddings(config)
        else:
            raise


def _get_huggingface_embeddings(config: EmbeddingConfig) -> Embeddings:
    """Get HuggingFace embeddings instance.

    Args:
        config: EmbeddingConfig with HuggingFace settings

    Returns:
        HuggingFaceEmbeddings instance
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    # Resolve device
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = config.model or "BAAI/bge-base-en-v1.5"
    dimension = EMBEDDING_DIMENSIONS.get(model_name, config.dimension)

    logger.info(f"Initializing HuggingFace embeddings: {model_name} on {device} ({dimension} dimensions)")

    model_kwargs = {"device": device}
    if config.cache_folder:
        model_kwargs["cache_folder"] = config.cache_folder

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": config.normalize}
    )

    logger.info("HuggingFace embeddings initialized successfully")
    return embeddings


def _dict_to_embedding_config(config_dict: dict) -> EmbeddingConfig:
    """Convert dict to EmbeddingConfig for backward compatibility.

    Args:
        config_dict: Dictionary with embedding configuration

    Returns:
        EmbeddingConfig object
    """
    return EmbeddingConfig(
        provider=config_dict.get("provider", "huggingface"),
        model=config_dict.get("model", "BAAI/bge-base-en-v1.5"),
        dimension=config_dict.get("dimension", 768),
        openai_model=config_dict.get("openai_model", "text-embedding-3-small"),
        openai_api_key=config_dict.get("openai_api_key"),
        normalize=config_dict.get("normalize", True),
        device=config_dict.get("device", "auto"),
        batch_size=config_dict.get("batch_size", 32),
        cache_folder=config_dict.get("cache_folder")
    )


def get_embedding_info(embeddings: Embeddings) -> dict:
    """Get information about an embeddings instance.

    Args:
        embeddings: Embeddings instance

    Returns:
        Dict with provider, model, and dimension info
    """
    # Detect provider type
    class_name = embeddings.__class__.__name__

    if "OpenAI" in class_name:
        model = getattr(embeddings, "model", "text-embedding-3-small")
        return {
            "provider": "openai",
            "model": model,
            "dimension": EMBEDDING_DIMENSIONS.get(model, 1536)
        }
    else:
        model = getattr(embeddings, "model_name", "BAAI/bge-base-en-v1.5")
        return {
            "provider": "huggingface",
            "model": model,
            "dimension": EMBEDDING_DIMENSIONS.get(model, 768)
        }
