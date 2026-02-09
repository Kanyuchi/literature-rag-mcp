"""Connection Pool Module for Literature RAG

Manages shared ChromaDB clients and embedding model instances.
Provides efficient resource sharing across requests and workers.
Supports both HuggingFace (local) and OpenAI (API) embeddings.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from weakref import WeakValueDictionary

import chromadb
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for the connection pool."""
    max_chroma_clients: int = 10  # Max ChromaDB client instances
    embedding_cache_size: int = 5  # Max embedding model instances
    default_device: str = "auto"
    default_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # OpenAI settings
    default_embedding_provider: str = "huggingface"  # "huggingface" or "openai"
    default_openai_model: str = "text-embedding-3-small"


class EmbeddingPool:
    """Pool of embedding model instances.

    Supports both HuggingFace (local) and OpenAI (API) embeddings.
    Embedding models are expensive to load (~500MB+ for BGE-base).
    This pool caches loaded models and shares them across requests.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        max_size: int = 5,
        default_device: str = "auto",
        default_provider: str = "huggingface",
        default_openai_model: str = "text-embedding-3-small"
    ):
        """Initialize embedding pool.

        Args:
            max_size: Maximum number of cached model instances
            default_device: Default device for HuggingFace models
            default_provider: Default provider ("huggingface" or "openai")
            default_openai_model: Default OpenAI model name
        """
        self._cache: Dict[str, Embeddings] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._access_order: list = []  # LRU tracking
        self._default_device = default_device
        self._default_provider = default_provider
        self._default_openai_model = default_openai_model

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _make_cache_key(
        self,
        provider: str,
        model_name: str,
        device: str = None
    ) -> str:
        """Create cache key for model instance."""
        if provider == "openai":
            return f"openai::{model_name}"
        else:
            return f"huggingface::{model_name}::{device}"

    def get_embeddings(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = True,
        provider: str = None,
        openai_model: str = None,
        openai_api_key: str = None
    ) -> Embeddings:
        """Get or create an embedding model instance.

        Args:
            model_name: HuggingFace model identifier (default: BGE-base)
            device: Device to use for HuggingFace (default: auto)
            normalize: Whether to normalize embeddings (HuggingFace only)
            provider: "huggingface" or "openai" (default: auto-detect)
            openai_model: OpenAI model name (default: text-embedding-3-small)
            openai_api_key: OpenAI API key (default: from environment)

        Returns:
            Embeddings instance (HuggingFaceEmbeddings or OpenAIEmbeddings)
        """
        # Determine provider
        if provider is None:
            # Auto-detect: use OpenAI if API key is available
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            provider = "openai" if api_key else self._default_provider

        if provider == "openai":
            return self._get_openai_embeddings(openai_model, openai_api_key)
        else:
            return self._get_huggingface_embeddings(model_name, device, normalize)

    def _get_huggingface_embeddings(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = True
    ) -> HuggingFaceEmbeddings:
        """Get or create a HuggingFace embedding model instance."""
        model_name = model_name or "BAAI/bge-base-en-v1.5"
        device = self._resolve_device(device or self._default_device)
        cache_key = self._make_cache_key("huggingface", model_name, device)

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                logger.debug(f"Embedding pool hit: {cache_key}")
                return self._cache[cache_key]

            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
                logger.info(f"Embedding pool eviction: {lru_key}")

            # Create new instance
            logger.info(f"Loading HuggingFace embedding model: {model_name} on {device}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": normalize}
            )

            self._cache[cache_key] = embeddings
            self._access_order.append(cache_key)
            logger.info(f"HuggingFace embedding model loaded. Pool size: {len(self._cache)}")

            return embeddings

    def _get_openai_embeddings(
        self,
        model_name: str = None,
        api_key: str = None
    ) -> Embeddings:
        """Get or create an OpenAI embedding model instance."""
        model_name = model_name or self._default_openai_model
        cache_key = self._make_cache_key("openai", model_name)

        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found, falling back to HuggingFace")
            return self._get_huggingface_embeddings()

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                logger.debug(f"Embedding pool hit: {cache_key}")
                return self._cache[cache_key]

            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
                logger.info(f"Embedding pool eviction: {lru_key}")

            # Create new instance
            try:
                from langchain_openai import OpenAIEmbeddings

                logger.info(f"Initializing OpenAI embedding model: {model_name}")
                embeddings = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=api_key
                )

                self._cache[cache_key] = embeddings
                self._access_order.append(cache_key)
                logger.info(f"OpenAI embedding model loaded. Pool size: {len(self._cache)}")

                return embeddings

            except ImportError:
                logger.warning("langchain-openai not installed, falling back to HuggingFace")
                return self._get_huggingface_embeddings()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}, falling back to HuggingFace")
                return self._get_huggingface_embeddings()

    def clear(self):
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Embedding pool cleared")

    @property
    def size(self) -> int:
        """Current number of cached models."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "cached_models": len(self._cache),
                "max_size": self._max_size,
                "models": list(self._cache.keys()),
                "default_device": self._default_device,
                "default_provider": self._default_provider
            }


class ChromaPool:
    """Pool of ChromaDB client connections.

    ChromaDB PersistentClient instances can be shared across requests.
    This pool manages client lifecycle and provides efficient reuse.
    """

    def __init__(self, max_clients: int = 10):
        """Initialize ChromaDB pool.

        Args:
            max_clients: Maximum number of client instances
        """
        self._clients: Dict[str, chromadb.PersistentClient] = {}
        self._lock = threading.Lock()
        self._max_clients = max_clients
        self._access_order: list = []

    def get_client(self, path: str) -> chromadb.PersistentClient:
        """Get or create a ChromaDB client for a path.

        Args:
            path: Path to ChromaDB persistence directory

        Returns:
            ChromaDB PersistentClient instance
        """
        with self._lock:
            # Check cache
            if path in self._clients:
                # Update LRU order
                if path in self._access_order:
                    self._access_order.remove(path)
                self._access_order.append(path)
                logger.debug(f"ChromaDB pool hit: {path}")
                return self._clients[path]

            # Evict LRU entry if at capacity
            if len(self._clients) >= self._max_clients:
                lru_path = self._access_order.pop(0)
                # ChromaDB doesn't require explicit close, but we remove reference
                del self._clients[lru_path]
                logger.info(f"ChromaDB pool eviction: {lru_path}")

            # Create new client
            logger.info(f"Creating ChromaDB client for: {path}")
            client = chromadb.PersistentClient(path=path)

            self._clients[path] = client
            self._access_order.append(path)
            logger.info(f"ChromaDB client created. Pool size: {len(self._clients)}")

            return client

    def clear(self):
        """Clear all cached clients."""
        with self._lock:
            self._clients.clear()
            self._access_order.clear()
            logger.info("ChromaDB pool cleared")

    @property
    def size(self) -> int:
        """Current number of cached clients."""
        return len(self._clients)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "cached_clients": len(self._clients),
                "max_clients": self._max_clients,
                "paths": list(self._clients.keys())
            }


class ConnectionPool:
    """Unified connection pool for Literature RAG resources.

    Manages both embedding models and ChromaDB clients.
    Singleton pattern ensures single pool instance per process.
    """

    _instance: Optional["ConnectionPool"] = None
    _lock = threading.Lock()

    def __new__(cls, config: PoolConfig = None):
        """Singleton pattern for connection pool."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: PoolConfig = None):
        """Initialize connection pool.

        Args:
            config: Pool configuration
        """
        if self._initialized:
            return

        config = config or PoolConfig()

        self._embedding_pool = EmbeddingPool(
            max_size=config.embedding_cache_size,
            default_device=config.default_device,
            default_provider=config.default_embedding_provider,
            default_openai_model=config.default_openai_model
        )
        self._chroma_pool = ChromaPool(max_clients=config.max_chroma_clients)
        self._config = config
        self._initialized = True

        logger.info("Connection pool initialized")

    def get_embeddings(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = True,
        provider: str = None,
        openai_model: str = None,
        openai_api_key: str = None
    ) -> Embeddings:
        """Get embedding model from pool."""
        return self._embedding_pool.get_embeddings(
            model_name=model_name or self._config.default_embedding_model,
            device=device or self._config.default_device,
            normalize=normalize,
            provider=provider or self._config.default_embedding_provider,
            openai_model=openai_model or self._config.default_openai_model,
            openai_api_key=openai_api_key
        )

    def get_chroma_client(self, path: str) -> chromadb.PersistentClient:
        """Get ChromaDB client from pool."""
        return self._chroma_pool.get_client(path)

    def clear(self):
        """Clear all pools."""
        self._embedding_pool.clear()
        self._chroma_pool.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "embeddings": self._embedding_pool.get_stats(),
            "chroma": self._chroma_pool.get_stats()
        }


# Global pool instance
_pool: Optional[ConnectionPool] = None


def get_pool(config: PoolConfig = None) -> ConnectionPool:
    """Get global connection pool instance.

    Args:
        config: Optional pool configuration (only used on first call)

    Returns:
        ConnectionPool singleton instance
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config)
    return _pool


def get_pooled_embeddings(
    model_name: str = None,
    device: str = None,
    normalize: bool = True,
    provider: str = None,
    openai_model: str = None,
    openai_api_key: str = None
) -> Embeddings:
    """Convenience function to get embeddings from global pool."""
    return get_pool().get_embeddings(
        model_name=model_name,
        device=device,
        normalize=normalize,
        provider=provider,
        openai_model=openai_model,
        openai_api_key=openai_api_key
    )


def get_pooled_chroma_client(path: str) -> chromadb.PersistentClient:
    """Convenience function to get ChromaDB client from global pool."""
    return get_pool().get_chroma_client(path)
