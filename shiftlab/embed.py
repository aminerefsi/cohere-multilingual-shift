"""Text embedders: the real Cohere API, plus an offline fake for tests.

Embeddings are cached on disk, so re-running the experiment costs no API calls.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np


class CachedEmbedder:
    """Base class: handles the disk cache; subclasses implement `_embed_batch`."""

    name = "base"
    batch_size = 96

    def __init__(self, cache_dir: str = "cache"):
        self.cache_path = Path(cache_dir) / f"{self.name.replace('/', '_')}.pkl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            self.cache = pickle.loads(self.cache_path.read_bytes())
        self.api_calls = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            vectors = self._embed_batch(batch)
            self.api_calls += 1
            for text, vec in zip(batch, vectors):
                self.cache[self._key(text)] = np.asarray(vec, dtype=np.float32)
        if missing:
            self.cache_path.write_bytes(pickle.dumps(self.cache))
        return np.stack([self.cache[self._key(t)] for t in texts])


class CohereEmbedder(CachedEmbedder):
    """Cohere Embed API (multilingual). Needs COHERE_API_KEY."""

    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0", cache_dir: str = "cache"):
        import cohere

        self.name = model
        self.model = model
        self.client = cohere.ClientV2(api_key=api_key)
        super().__init__(cache_dir)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(6):
            try:
                resp = self.client.embed(
                    model=self.model,
                    texts=texts,
                    input_type="classification",
                    embedding_types=["float"],
                    truncate="END",
                )
                return resp.embeddings.float_
            except Exception as exc:  # rate limit or transient network error
                wait = 2 ** attempt * 5
                print(f"  embed call failed ({type(exc).__name__}: {exc}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Cohere embed failed 6 times in a row; check your key and network.")


class FakeEmbedder(CachedEmbedder):
    """Deterministic hashed bag-of-words vectors. Offline, for tests only."""

    name = "fake-hash-bow"

    def __init__(self, dim: int = 256, cache_dir: str = "cache"):
        self.dim = dim
        super().__init__(cache_dir)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out
