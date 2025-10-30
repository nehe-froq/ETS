from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import time

import pandas as pd
import yaml
from rich.progress import Progress

from embeddings.csv_utils import (
    build_text,
    compute_row_hash,
    make_row_id,
    pick_key_columns,
    pick_text_columns,
)
from embeddings.embed import Embedder
from embeddings.index_faiss import FaissSqliteIndex, MetaRow, make_chunk_id

start_time = time.time()
print(f'Starting ingest at {start_time}')

def load_config(path: str | None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest CSV → embeddings → FAISS index (batch/incremental)")
    parser.add_argument("--csv", required=False, default="./embeddings/data.csv", help="Path to CSV file")
    parser.add_argument("--out", required=False, default="./embeddings/store", help="Output index directory")
    parser.add_argument("--config", required=False, default=None, help="Path to YAML config")
    parser.add_argument("--incremental", action="store_true", help="Incremental mode (add/update/remove)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Embedding batch size of rows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name: str = cfg.get("model", {}).get("name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    normalize_embeddings = bool(cfg.get("model", {}).get("normalize_embeddings", True))
    st_batch = int(cfg.get("model", {}).get("batch_size", 128))
    min_text_len = int(cfg.get("csv", {}).get("min_text_len", 10))
    csv_path = cfg.get("csv", {}).get("path", args.csv)
    text_cols_cfg = cfg.get("csv", {}).get("text_columns", []) or None
    key_cols_cfg = cfg.get("csv", {}).get("key_columns", []) or None

    # Load CSV
    df = pd.read_csv(csv_path)
    text_columns = pick_text_columns(df, text_cols_cfg)
    key_columns = pick_key_columns(df, key_cols_cfg)

    # Build texts and ids
    df["__text__"] = df.apply(lambda r: build_text(r, text_columns), axis=1)
    df = df[df["__text__"].map(lambda s: isinstance(s, str) and len(s) >= min_text_len)]
    df["__id__"] = df.apply(lambda r: make_row_id(r, key_columns), axis=1)

    embedder = Embedder(model_name=model_name, batch_size=st_batch, normalize_embeddings=normalize_embeddings)
    index = FaissSqliteIndex(index_dir=args.out, dim=embedder.dimension)

    added = 0
    updated = 0
    removed = 0

    with Progress() as progress:
        prep_task = progress.add_task("Preparing rows...", total=len(df))
        embed_task = progress.add_task("Embedding...", total=len(df))
        add_task = progress.add_task("Indexing...", total=len(df))

        # Incremental: compute current ids from SQLite idmap
        current_ids: set[str] = set()
        if args.incremental:
            cur = index._conn.cursor()
            rows = cur.execute("SELECT id FROM idmap").fetchall()
            current_ids = {r[0] for r in rows}

        # Build list of new/changed ids
        to_add: List[Tuple[str, str]] = []  # (id, text)
        seen_ids: set[str] = set()
        for _, row in df.iterrows():
            rid = row["__id__"]
            txt = row["__text__"]
            seen_ids.add(rid)
            if not args.incremental or rid not in current_ids:
                to_add.append((rid, txt))
            progress.advance(prep_task)

        # Removals
        if args.incremental:
            to_remove = list(current_ids - seen_ids)
            if to_remove:
                index.remove_by_str_ids(to_remove)
                removed += len(to_remove)

        # Add/Update in batches
        batch_rows: List[Tuple[str, str]] = []
        for rid, txt in to_add:
            batch_rows.append((rid, txt))
            if len(batch_rows) >= args.batch_size:
                texts = [t for _, t in batch_rows]
                embs = embedder.encode(texts)
                metas = [MetaRow(id=rid, source=os.path.abspath(csv_path), table_name="csv", row_id=rid, position=0, text=txt) for rid, txt in batch_rows]
                index.add(metas, embs)
                added += len(batch_rows)
                progress.advance(add_task, len(batch_rows))
                batch_rows.clear()
                progress.advance(embed_task, len(texts))

        if batch_rows:
            texts = [t for _, t in batch_rows]
            embs = embedder.encode(texts)
            metas = [MetaRow(id=rid, source=os.path.abspath(csv_path), table_name="csv", row_id=rid, position=0, text=txt) for rid, txt in batch_rows]
            index.add(metas, embs)
            added += len(batch_rows)
            progress.advance(add_task, len(batch_rows))
            progress.advance(embed_task, len(texts))

        index.persist()

    print(f"CSV ingest complete. added={added} removed={removed} text_columns={text_columns} key_columns={key_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


end_time = time.time()
print(f'Ingest complete in {end_time - start_time} seconds')
