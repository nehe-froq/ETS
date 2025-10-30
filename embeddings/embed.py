from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 128, normalize_embeddings: bool = True) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize_embeddings

    @property
    def dimension(self) -> int:
        # Trigger a small encode to detect dimension once
        emb = self.model.encode(["dim"], batch_size=1, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return int(emb.shape[1])

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).astype(np.float32)

