"""Helpers for normalized embedding cache persistence and reuse.

Cache scope is fixed to nomask normalized embeddings:
- input_norm
- output_norm

Files are stored under:
<output-dir>/_temp/embeddings/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from final_metric_refactor.embedding.embedder import TextEmbedder
from final_metric_refactor.shared.geometry import normalize_rows, sanitize_matrix


@dataclass(frozen=True)
class EmbeddingCachePaths:
    input_norm_path: Path
    output_norm_path: Path
    meta_json_path: Path

    @property
    def all_paths(self) -> tuple[Path, Path, Path]:
        return (self.input_norm_path, self.output_norm_path, self.meta_json_path)


@dataclass(frozen=True)
class LoadedEmbeddingCache:
    paths: EmbeddingCachePaths
    input_norm: np.ndarray
    output_norm: np.ndarray
    meta: dict[str, Any]


def build_nomask_cache_paths(output_dir: Path | str, tag: str, stem: str) -> EmbeddingCachePaths:
    root = Path(output_dir).resolve()
    cache_dir = root / "_temp" / "embeddings"
    prefix = f"{str(tag)}_{str(stem)}_nomask"
    return EmbeddingCachePaths(
        input_norm_path=cache_dir / f"{prefix}_input_norm.f32.npy",
        output_norm_path=cache_dir / f"{prefix}_output_norm.f32.npy",
        meta_json_path=cache_dir / f"{prefix}_embedding_cache.json",
    )


def cache_exists(paths: EmbeddingCachePaths) -> bool:
    return all(p.exists() for p in paths.all_paths)


def _to_unique_row_index(row_indices: np.ndarray | list[int], total_rows: int) -> np.ndarray:
    idx = np.asarray(row_indices, dtype=int).ravel()
    if idx.size == 0:
        return idx
    if int(np.min(idx)) < 0 or int(np.max(idx)) >= int(total_rows):
        raise ValueError("row index out of range for embedding cache write")
    if idx.size == np.unique(idx).size:
        return idx
    # Keep first occurrence order.
    seen: set[int] = set()
    out: list[int] = []
    for v in idx.tolist():
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return np.asarray(out, dtype=int)


class NomaskEmbeddingCacheWriter:
    """Incremental memmap writer for nomask normalized embeddings."""

    def __init__(
        self,
        paths: EmbeddingCachePaths,
        total_rows: int,
        *,
        dtype: str = "float32",
    ) -> None:
        self.paths = paths
        self.total_rows = int(total_rows)
        self.dtype = np.dtype(dtype)
        self._input_mm: np.memmap | None = None
        self._output_mm: np.memmap | None = None
        self._dim: int | None = None
        self._written_mask = np.zeros(self.total_rows, dtype=bool)

    def _ensure_open(self, dim: int) -> None:
        if self._input_mm is not None and self._output_mm is not None:
            if self._dim != int(dim):
                raise ValueError(f"embedding dim mismatch: expected {self._dim}, got {dim}")
            return
        self.paths.meta_json_path.parent.mkdir(parents=True, exist_ok=True)
        shape = (self.total_rows, int(dim))
        self._input_mm = open_memmap(
            str(self.paths.input_norm_path),
            mode="w+",
            dtype=self.dtype,
            shape=shape,
        )
        self._output_mm = open_memmap(
            str(self.paths.output_norm_path),
            mode="w+",
            dtype=self.dtype,
            shape=shape,
        )
        self._input_mm[:] = np.nan
        self._output_mm[:] = np.nan
        self._dim = int(dim)

    def write(
        self,
        row_indices: np.ndarray | list[int],
        input_norm: np.ndarray,
        output_norm: np.ndarray,
    ) -> None:
        idx = _to_unique_row_index(row_indices=row_indices, total_rows=self.total_rows)
        if idx.size == 0:
            return
        x = np.asarray(input_norm, dtype=self.dtype)
        y = np.asarray(output_norm, dtype=self.dtype)
        if x.shape != y.shape:
            raise ValueError(f"input/output shape mismatch: {x.shape} vs {y.shape}")
        if x.ndim != 2:
            raise ValueError(f"expected 2D normalized embeddings, got {x.ndim}D")
        if x.shape[0] != idx.size:
            raise ValueError(f"row count mismatch: row_indices={idx.size}, embeddings={x.shape[0]}")
        self._ensure_open(dim=int(x.shape[1]))
        assert self._input_mm is not None and self._output_mm is not None
        self._input_mm[idx, :] = x
        self._output_mm[idx, :] = y
        self._written_mask[idx] = True

    def finalize(self, extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._input_mm is None or self._output_mm is None or self._dim is None:
            return {}
        self._input_mm.flush()
        self._output_mm.flush()
        del self._input_mm
        del self._output_mm
        self._input_mm = None
        self._output_mm = None

        meta: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "nomask",
            "dtype": str(self.dtype),
            "shape": [int(self.total_rows), int(self._dim)],
            "row_alignment": "row_results_index",
            "storage": "numpy_memmap",
            "input_norm_path": str(self.paths.input_norm_path.resolve()),
            "output_norm_path": str(self.paths.output_norm_path.resolve()),
            "valid_rows": int(self._written_mask.sum()),
            "valid_rate": float(self._written_mask.mean()) if self.total_rows > 0 else 0.0,
            "valid_row_indices": np.where(self._written_mask)[0].astype(int).tolist(),
        }
        if extra_meta:
            meta.update(extra_meta)
        self.paths.meta_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta


def _read_meta(meta_json_path: Path) -> dict[str, Any]:
    if not meta_json_path.exists():
        raise FileNotFoundError(f"embedding cache meta not found: {meta_json_path}")
    data = json.loads(meta_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid embedding cache meta format: {meta_json_path}")
    return data


def _paths_from_meta_or_default(
    default_paths: EmbeddingCachePaths,
    meta: dict[str, Any],
) -> EmbeddingCachePaths:
    input_path = Path(str(meta.get("input_norm_path", default_paths.input_norm_path))).resolve()
    output_path = Path(str(meta.get("output_norm_path", default_paths.output_norm_path))).resolve()
    return EmbeddingCachePaths(
        input_norm_path=input_path,
        output_norm_path=output_path,
        meta_json_path=default_paths.meta_json_path.resolve(),
    )


def load_nomask_cache(paths: EmbeddingCachePaths) -> LoadedEmbeddingCache:
    meta = _read_meta(paths.meta_json_path)
    resolved_paths = _paths_from_meta_or_default(default_paths=paths, meta=meta)
    if not resolved_paths.input_norm_path.exists():
        raise FileNotFoundError(f"cache input file not found: {resolved_paths.input_norm_path}")
    if not resolved_paths.output_norm_path.exists():
        raise FileNotFoundError(f"cache output file not found: {resolved_paths.output_norm_path}")
    x = np.load(resolved_paths.input_norm_path, mmap_mode="r")
    y = np.load(resolved_paths.output_norm_path, mmap_mode="r")
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"invalid cache shapes: input={x.shape}, output={y.shape}")
    return LoadedEmbeddingCache(paths=resolved_paths, input_norm=x, output_norm=y, meta=meta)


def resolve_nomask_cache_paths(
    *,
    output_dir: Path | str,
    tag: str,
    stem: str,
    meta_json_path: Path | str | None = None,
) -> EmbeddingCachePaths:
    default_paths = build_nomask_cache_paths(output_dir=output_dir, tag=tag, stem=stem)
    if not meta_json_path:
        return default_paths
    meta_path = Path(meta_json_path).resolve()
    return EmbeddingCachePaths(
        input_norm_path=default_paths.input_norm_path,
        output_norm_path=default_paths.output_norm_path,
        meta_json_path=meta_path,
    )


def rebuild_nomask_cache(
    *,
    paths: EmbeddingCachePaths,
    input_texts: list[str],
    output_texts: list[str],
    embedder: TextEmbedder,
    batch_size: int = 64,
    source: str = "rebuild",
) -> dict[str, Any]:
    if len(input_texts) != len(output_texts):
        raise ValueError("input/output text length mismatch for cache rebuild")
    n_rows = len(input_texts)
    x = sanitize_matrix(embedder.encode(input_texts, batch_size=batch_size))
    y = sanitize_matrix(embedder.encode(output_texts, batch_size=batch_size))
    x_norm = normalize_rows(x).astype(np.float32, copy=False)
    y_norm = normalize_rows(y).astype(np.float32, copy=False)

    writer = NomaskEmbeddingCacheWriter(paths=paths, total_rows=n_rows, dtype="float32")
    writer.write(np.arange(n_rows, dtype=int), x_norm, y_norm)
    return writer.finalize(
        {
            "source": str(source),
            "embedder": str(getattr(embedder, "name", "unknown")),
            "rebuilt": True,
        }
    )


def load_or_rebuild_nomask_cache(
    *,
    paths: EmbeddingCachePaths,
    expected_rows: int,
    input_texts: list[str],
    output_texts: list[str],
    embedder: TextEmbedder,
    batch_size: int = 64,
    allow_rebuild: bool = True,
) -> LoadedEmbeddingCache:
    def _validate_rows(payload: LoadedEmbeddingCache) -> None:
        if int(payload.input_norm.shape[0]) != int(expected_rows):
            raise ValueError(
                f"cache row count mismatch: cache={payload.input_norm.shape[0]}, expected={expected_rows}"
            )

    try:
        loaded = load_nomask_cache(paths=paths)
        _validate_rows(loaded)
        return loaded
    except Exception:
        if not allow_rebuild:
            raise
        rebuild_nomask_cache(
            paths=paths,
            input_texts=input_texts,
            output_texts=output_texts,
            embedder=embedder,
            batch_size=batch_size,
            source="auto_rebuild",
        )
        loaded = load_nomask_cache(paths=paths)
        _validate_rows(loaded)
        return loaded
