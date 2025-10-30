from __future__ import annotations

import argparse
import os
import sqlite3
from typing import List

import numpy as np
from rich.table import Table
from rich.console import Console

from embeddings.embed import Embedder
from embeddings.index_faiss import FaissSqliteIndex, MetaRow


def main() -> int:
    parser = argparse.ArgumentParser(description="Search FAISS index with multilingual embeddings")
    parser.add_argument("--index", required=False, default="./embeddings/store", help="Index directory")
    parser.add_argument("--model", required=False, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)
    dim = embedder.dimension
    index = FaissSqliteIndex(index_dir=args.index, dim=dim)

    q = args.query.strip()
    qvec = embedder.encode([q])
    results = index.search(qvec[0], k=args.k)

    table = Table(show_header=True, header_style="bold")
    table.add_column("score", style="cyan")
    table.add_column("table")
    table.add_column("row_id")
    table.add_column("preview")
    for score, meta in results:
        preview = (meta.text[:160] + "…") if len(meta.text) > 160 else meta.text
        table.add_row(f"{score:.4f}", str(meta.table_name or ""), str(meta.row_id or ""), preview)

    Console().print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


