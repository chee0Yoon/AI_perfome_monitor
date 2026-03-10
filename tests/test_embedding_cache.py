#!/usr/bin/env python3
"""Unit tests for embedding cache naming and legacy fallback."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR.parent
ROOT_DIR = FINAL_DIR.parent
for p in (ROOT_DIR, FINAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from final_metric_refactor.embedding.cache import (  # noqa: E402
    build_embedding_cache_paths,
    resolve_embedding_cache_paths,
)


class EmbeddingCacheNamingTest(unittest.TestCase):
    def test_new_cache_paths_do_not_use_nomask_suffix(self) -> None:
        paths = build_embedding_cache_paths(output_dir="/tmp/example", tag="tag", stem="stem")
        self.assertNotIn("nomask", paths.input_norm_path.name)
        self.assertNotIn("nomask", paths.output_norm_path.name)
        self.assertNotIn("nomask", paths.meta_json_path.name)

    def test_resolve_embedding_cache_paths_falls_back_to_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_dir = root / "_temp" / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)

            legacy_meta = emb_dir / "tag_stem_nomask_embedding_cache.json"
            legacy_in = emb_dir / "tag_stem_nomask_input_norm.f32.npy"
            legacy_out = emb_dir / "tag_stem_nomask_output_norm.f32.npy"
            legacy_meta.write_text("{}", encoding="utf-8")
            legacy_in.write_bytes(b"x")
            legacy_out.write_bytes(b"y")

            resolved = resolve_embedding_cache_paths(output_dir=root, tag="tag", stem="stem")
            self.assertEqual(resolved.meta_json_path, legacy_meta.resolve())
            self.assertEqual(resolved.input_norm_path, legacy_in.resolve())
            self.assertEqual(resolved.output_norm_path, legacy_out.resolve())


if __name__ == "__main__":
    unittest.main()
