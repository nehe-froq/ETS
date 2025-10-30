from __future__ import annotations

import argparse
import os
import sys
from typing import List

import yaml
from rich.progress import Progress

from embeddings.chunker import TokenChunker, Chunk
from embeddings.sql_text_extractor import extract_text_from_sql

print("Starting ingest...")


def load_config(path: str | None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest SQL dump → chunks → embeddings → FAISS index")
    parser.add_argument("--sql", required=True, help="Path to SQL dump file (e.g., damm.sql)")
    parser.add_argument("--out", required=False, default="./embeddings/store", help="Output index directory")
    parser.add_argument("--config", required=False, default=None, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not embed/index")
    parser.add_argument("--persist-every", type=int, default=100000, help="Persist FAISS/DB every N vectors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    model_name: str = model_cfg.get("name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_tokens = int(cfg.get("chunking", {}).get("max_tokens", 512))
    overlap_tokens = int(cfg.get("chunking", {}).get("overlap_tokens", 64))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", True))
    batch_size = int(model_cfg.get("batch_size", 128))
    ingest_cfg = cfg.get("ingest", {})
    min_text_len = int(ingest_cfg.get("min_text_len", 10))
    faiss_cfg = cfg.get("faiss", {})

    # Components
    chunker = TokenChunker(model_name=model_name, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    # DRY RUN MODE
    if args.dry_run:
        with Progress() as progress:
            task = progress.add_task("Parsing SQL...", start=False)
            count = 0
            for _ in extract_text_from_sql(args.sql, min_text_len=min_text_len):
                if count == 0:
                    progress.start_task(task)
                count += 1
                if count % 10000 == 0:
                    progress.update(task, description=f"Parsed {count} text records")
            progress.update(task, description=f"Parsed {count} text records (dry run)")
        return 0

    # Lazy imports (avoid requiring torch for dry-run)
    from embeddings.embed import Embedder  # type: ignore
    from embeddings.index_faiss import FaissSqliteIndex, MetaRow, make_chunk_id  # type: ignore

    embedder = Embedder(model_name=model_name, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
    dim = embedder.dimension
    index = FaissSqliteIndex(
        index_dir=args.out,
        dim=dim,
        factory=str(faiss_cfg.get("factory", "Flat")),
        metric=str(faiss_cfg.get("metric", "ip")),
        nprobe=int(faiss_cfg.get("nprobe", 16)),
        hnsw_ef_search=int(faiss_cfg.get("hnsw_ef_search", 64)),
        hnsw_ef_construction=int(faiss_cfg.get("hnsw_ef_construction", 200)),
    )

    total_added = 0
    chunk_buffer: List[Chunk] = []

    with Progress() as progress:
        parse_task = progress.add_task("Parsing SQL...", start=False)
        embed_task = progress.add_task("Embedding...")
        add_task = progress.add_task("Indexing...")

        for rec in extract_text_from_sql(args.sql, min_text_len=min_text_len):
            if total_added == 0 and not progress.tasks[parse_task].started:
                progress.start_task(parse_task)

            for ch in chunker.chunk_records([(rec.source, rec.table_name, rec.row_id, rec.text)]):
                chunk_buffer.append(ch)

                if len(chunk_buffer) >= 10000:
                    # Process batch
                    texts = [c.text for c in chunk_buffer]
                    embs = embedder.encode(texts)
                    metas = [
                        MetaRow(
                            id=make_chunk_id(c.source, c.table_name, c.row_id, c.position, c.text),
                            source=c.source,
                            table_name=c.table_name,
                            row_id=c.row_id,
                            position=c.position,
                            text=c.text,
                        )
                        for c in chunk_buffer
                    ]
                    progress.update(embed_task, description=f"Embedding {len(chunk_buffer)} chunks")
                    index.add(metas, embs)
                    total_added += len(chunk_buffer)
                    progress.update(add_task, description=f"Indexed total {total_added}")
                    chunk_buffer.clear()

                    if total_added % args.persist_every == 0:
                        index.persist()

        # Flush remainder
        if chunk_buffer:
            texts = [c.text for c in chunk_buffer]
            embs = embedder.encode(texts)
            metas = [
                MetaRow(
                    id=make_chunk_id(c.source, c.table_name, c.row_id, c.position, c.text),
                    source=c.source,
                    table_name=c.table_name,
                    row_id=c.row_id,
                    position=c.position,
                    text=c.text,
                )
                for c in chunk_buffer
            ]
            index.add(metas, embs)
            total_added += len(chunk_buffer)
            chunk_buffer.clear()

        index.persist()
        progress.update(add_task, description=f"Done. Indexed total {total_added}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

print("Ingest completed")
