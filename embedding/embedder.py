"""
Text embedding backends for the distribution pipeline.

Provides pluggable embedding implementations with optional GPU support via
sentence-transformers or a dependency-free hashing fallback.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:
    _SentenceTransformer = None


class TextEmbedder:
    """Abstract base class for text embedding backends."""

    name: str

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding (ignored by some backends).
            show_progress_bar: Whether to show a progress bar (ignored by some backends).

        Returns:
            Array of shape (len(texts), embedding_dim) with dtype float32.
        """
        raise NotImplementedError


class SentenceTransformersEmbedder(TextEmbedder):
    """Embedding backend using sentence-transformers with GPU support."""

    def __init__(self, model_name: str):
        """Initialize with a sentence-transformers model.

        Args:
            model_name: Name of the model (e.g. "all-MiniLM-L6-v2").

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        if _SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is not available in this environment.")
        self.model = _SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts using sentence-transformers.

        Args:
            texts: List of text strings.
            batch_size: Batch size for GPU processing.
            show_progress_bar: Whether to show progress bar.

        Returns:
            Float32 embedding array of shape (len(texts), embedding_dim).
        """
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            ),
            dtype=np.float32,
        )


class HashingEmbedder(TextEmbedder):
    """Fallback embedding backend using BLAKE2b hashing (no dependencies)."""

    TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self, dim: int = 768):
        """Initialize with a dimension.

        Args:
            dim: Embedding dimension (will be clamped to [64, inf)).
        """
        self.dim = max(64, int(dim))
        self.name = f"hash:{self.dim}"

    def _index_and_sign(self, token: str) -> tuple[int, float]:
        """Hash a token to an (index, sign) pair.

        Args:
            token: Token string.

        Returns:
            (index in [0, dim), sign in {-1.0, 1.0}).
        """
        h = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        v = int.from_bytes(h, byteorder="little", signed=False)
        idx = v % self.dim
        sign = 1.0 if ((v >> 63) & 1) == 0 else -1.0
        return int(idx), sign

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts using TF-weighted token hashing.

        Args:
            texts: List of text strings.
            batch_size: Ignored (no batching needed).
            show_progress_bar: Ignored.

        Returns:
            Float32 embedding array of shape (len(texts), self.dim).
        """
        _ = (batch_size, show_progress_bar)
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self.TOKEN_PATTERN.findall(str(text).lower())
            if not tokens:
                continue
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                idx, sign = self._index_and_sign(tok)
                mat[i, idx] += sign * float(1.0 + math.log(cnt))
        return mat


def build_embedder(backend: str, embedding_model: str, hash_dim: int = 768) -> TextEmbedder:
    """Factory function to build an embedder.

    Args:
        backend: Backend choice ("sentence-transformers", "hash", or "auto").
        embedding_model: Model name for sentence-transformers.
        hash_dim: Dimension for hashing embedder.

    Returns:
        Initialized TextEmbedder instance.

    Raises:
        ValueError: If backend is not recognized.
    """
    if backend == "sentence-transformers":
        return SentenceTransformersEmbedder(embedding_model)
    if backend == "hash":
        return HashingEmbedder(dim=hash_dim)
    if backend == "auto":
        # Try sentence-transformers, fall back to hashing
        if _SentenceTransformer is not None:
            return SentenceTransformersEmbedder(embedding_model)
        return HashingEmbedder(dim=hash_dim)
    raise ValueError(f"Unknown embedding backend: {backend}")
