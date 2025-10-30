from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

from transformers import AutoTokenizer


@dataclass(frozen=True)
class Chunk:
    source: str
    table_name: str | None
    row_id: str | None
    position: int
    text: str


class TokenChunker:
    def __init__(self, model_name: str, max_tokens: int, overlap_tokens: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_records(self, records: Iterable[Tuple[str, str | None, str | None, str]]) -> Iterator[Chunk]:
        position = 0
        for source, table_name, row_id, text in records:
            for chunk_text in self._chunk_text(text):
                yield Chunk(source=source, table_name=table_name, row_id=row_id, position=position, text=chunk_text)
                position += 1

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        enc = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        ids: List[int] = enc["input_ids"]
        if len(ids) <= self.max_tokens:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(ids):
            end = min(start + self.max_tokens, len(ids))
            piece_ids = ids[start:end]
            chunk_text = self.tokenizer.decode(piece_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if chunk_text:
                chunks.append(chunk_text)
            if end == len(ids):
                break
            start = end - self.overlap_tokens
            if start < 0:
                start = 0
            if start >= len(ids):
                break
        return chunks

