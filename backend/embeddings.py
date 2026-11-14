"""Encode English text on CPU, with a local model and a content-based cache."""

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np

from backend.config import MODEL_ID


def clean_text(text: str) -> str:
    # Keep punctuation and negations because they can change the meaning.
    return re.sub(r"\s+", " ", text).strip()


def normalise(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


class SentenceEncoder:
    def __init__(self, model_path: Path, cache_path: Path, allow_download: bool = False):
        self.model_path = model_path
        self.cache_path = cache_path
        self.allow_download = allow_download
        self.model = None
        self.lock = threading.RLock()

    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            if not self.model_path.exists() and not self.allow_download:
                raise RuntimeError("The local model is missing. Run python -m scripts.prepare_model first.")
            location = str(self.model_path) if self.model_path.exists() else MODEL_ID
            self.model = SentenceTransformer(
                location, device="cpu", local_files_only=not self.allow_download
            )
            if not self.model_path.exists():
                self.model.save(str(self.model_path))
        return self.model

    def _dimension(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None)
        return getter() if getter else model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        with self.lock:
            model = self._load()
            chunks, owners = [], []
            # Reserve space for special tokens, then split on token boundaries.
            window = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
            for index, text in enumerate(texts):
                token_ids = model.tokenizer.encode(clean_text(text), add_special_tokens=False, verbose=False)
                for start in range(0, len(token_ids), window):
                    chunks.append(model.tokenizer.decode(token_ids[start:start + window]))
                    owners.append(index)
            result = np.zeros((len(texts), self._dimension()), dtype=np.float32)
            if chunks:
                encoded = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
                counts = np.zeros(len(texts), dtype=np.float32)
                for owner, vector in zip(owners, encoded):
                    result[owner] += vector
                    counts[owner] += 1
                result /= np.maximum(counts[:, None], 1)
            return normalise(result)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        # Private user text is never written into this disk cache.
        with self.lock:
            model = self._load()
            model_digest = hashlib.sha256()
            for name in ("config.json", "modules.json", "sentence_bert_config.json",
                         "1_Pooling/config.json", "model.safetensors", "pytorch_model.bin"):
                path = self.model_path / name
                if path.exists():
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            model_digest.update(block)
            metadata = [MODEL_ID, model_digest.hexdigest(), model.max_seq_length, "chunk-mean-v1", texts]
            key = hashlib.sha256(json.dumps(metadata).encode()).hexdigest()
            path = self.cache_path / f"{key}.npy"
            if path.exists():
                try:
                    vectors = np.load(path, allow_pickle=False)
                    if vectors.shape == (len(texts), self._dimension()) and np.isfinite(vectors).all():
                        return vectors
                except (ValueError, OSError):
                    pass
            vectors = self.encode(texts)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors, allow_pickle=False)
            return vectors
