from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable, List, Sequence, Tuple, Optional

import faiss  # type: ignore
import numpy as np
import xxhash  # type: ignore


def to_int64_id(chunk_id: str) -> np.int64:
    """Hash a string to a positive signed int64 suitable for FAISS IDs.

    xxhash64 returns an unsigned 64-bit integer which may exceed int64 max;
    we mask to 0x7FFF... to keep the value within signed int64 range and positive.
    """
    h = xxhash.xxh3_64_intdigest(chunk_id)
    h_signed = h & 0x7FFFFFFFFFFFFFFF
    return np.int64(h_signed)


@dataclass(frozen=True)
class MetaRow:
    id: str
    source: Optional[str]
    table_name: Optional[str]
    row_id: Optional[str]
    position: int
    text: str


def make_chunk_id(source: str, table_name: str | None, row_id: str | None, position: int, text: str) -> str:
    key = f"{source}|{table_name or ''}|{row_id or ''}|{position}|{text[:64]}".encode("utf-8", errors="ignore")
    return sha1(key).hexdigest()


class FaissSqliteIndex:
    def __init__(
        self,
        index_dir: str,
        dim: int,
        *,
        factory: str | None = None,
        metric: str = "ip",
        nprobe: int | None = None,
        hnsw_ef_search: int | None = None,
        hnsw_ef_construction: int | None = None,
    ) -> None:
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.faiss_path = os.path.join(index_dir, "index.faiss")
        self.db_path = os.path.join(index_dir, "meta.sqlite")
        self.dim = dim
        self._metric = metric.lower().strip()
        self._nprobe = nprobe
        self._hnsw_ef_search = hnsw_ef_search
        self._hnsw_ef_construction = hnsw_ef_construction

        # Build base index according to factory/metric
        if self._metric not in ("ip", "l2"):
            raise ValueError("metric must be 'ip' or 'l2'")
        metric_type = faiss.METRIC_INNER_PRODUCT if self._metric == "ip" else faiss.METRIC_L2

        def _build_base() -> faiss.Index:
            if factory and factory.strip().lower() != "flat":
                return faiss.index_factory(dim, factory, metric_type)
            # Default Flat index
            return faiss.IndexFlatIP(dim) if metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)

        base = _build_base()
        # If HNSW, optionally set construction parameter (if building fresh)
        if hasattr(base, "hnsw") and self._hnsw_ef_construction is not None:
            try:
                base.hnsw.efConstruction = int(self._hnsw_ef_construction)
            except Exception:
                pass

        self.index: faiss.Index = faiss.IndexIDMap2(base)
        if os.path.exists(self.faiss_path):
            try:
                loaded = faiss.read_index(self.faiss_path)
                # Ensure we have an ID-mapped index that supports add_with_ids
                if not isinstance(loaded, faiss.IndexIDMap2):
                    loaded = faiss.IndexIDMap2(loaded)
                # Validate dimension to avoid native crashes
                if loaded.d != dim:
                    raise ValueError(
                        f"FAISS index dimension {loaded.d} != expected {dim}. Delete {self.faiss_path} to rebuild with the current model."
                    )
                self.index = loaded
            except Exception:
                self.index = faiss.IndexIDMap2(base)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        finally:
            pass

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              source TEXT,
              table_name TEXT,
              row_id TEXT,
              position INTEGER,
              text TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS idmap (
              idhash INTEGER PRIMARY KEY,
              id TEXT UNIQUE
            );
            """
        )
        self._conn.commit()

    def add(self, metas: Sequence[MetaRow], vectors: np.ndarray) -> None:
        assert len(metas) == vectors.shape[0], "metas and vectors size mismatch"
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must be (N,{self.dim})")

        # map string IDs to stable positive int64 via xxhash (masked)
        ids = np.array([to_int64_id(m.id) for m in metas], dtype=np.int64)
        # Train IVF if needed (train underlying index, not the ID map)
        try:
            base = self.index.index  # type: ignore[attr-defined]
        except Exception:
            base = self.index
        if hasattr(base, "is_trained") and hasattr(base, "train"):
            try:
                # Some index types expose is_trained
                if not base.is_trained:  # type: ignore[attr-defined]
                    base.train(vectors)
            except Exception:
                pass

        # add with ids
        self.index.add_with_ids(vectors, ids)

        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO chunks(id, source, table_name, row_id, position, text) VALUES(?,?,?,?,?,?)",
            [(m.id, m.source, m.table_name, m.row_id, m.position, m.text) for m in metas],
        )
        cur.executemany(
            "INSERT OR REPLACE INTO idmap(idhash, id) VALUES(?,?)",
            [(int(to_int64_id(m.id)), m.id) for m in metas],
        )
        self._conn.commit()

    def persist(self) -> None:
        faiss.write_index(self.index, self.faiss_path)
        self._conn.commit()

    def remove_by_ids(self, metas: Sequence[MetaRow]) -> int:
        """Remove vectors and metadata for the given MetaRow IDs.

        Returns number of removed vectors.
        """
        if not metas:
            return 0
        ids = np.array([to_int64_id(m.id) for m in metas], dtype=np.int64)
        removed = int(self.index.remove_ids(ids))
        cur = self._conn.cursor()
        cur.executemany("DELETE FROM chunks WHERE id = ?", [(m.id,) for m in metas])
        cur.executemany("DELETE FROM idmap WHERE id = ? OR idhash = ?", [(m.id, int(to_int64_id(m.id))) for m in metas])
        self._conn.commit()
        return removed

    def remove_by_str_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        metas = [MetaRow(id=i, source=None, table_name=None, row_id=None, position=0, text="") for i in ids]
        return self.remove_by_ids(metas)

    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[float, MetaRow]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        # No data indexed yet
        if getattr(self.index, "ntotal", 0) == 0:
            return []
        # Set search params on underlying index when available
        try:
            base = self.index.index  # type: ignore[attr-defined]
        except Exception:
            base = self.index
        # IVF parameter
        if self._nprobe is not None and hasattr(base, "nprobe"):
            try:
                base.nprobe = int(self._nprobe)
            except Exception:
                pass
        # HNSW parameter
        if self._hnsw_ef_search is not None and hasattr(base, "hnsw"):
            try:
                base.hnsw.efSearch = int(self._hnsw_ef_search)
            except Exception:
                pass
        D, I = self.index.search(query_vec.astype(np.float32), k)
        results: List[Tuple[float, MetaRow]] = []
        cur = self._conn.cursor()
        for score, int_id in zip(D[0].tolist(), I[0].tolist()):
            if int_id < 0:
                continue
            row = cur.execute(
                "SELECT c.id, c.source, c.table_name, c.row_id, c.position, c.text "
                "FROM chunks c JOIN idmap i ON c.id = i.id WHERE i.idhash = ?",
                (int(int_id),),
            ).fetchone()
            if not row:
                continue
            m = MetaRow(id=row[0], source=row[1], table_name=row[2], row_id=row[3], position=int(row[4] or 0), text=row[5] or "")
            results.append((float(score), m))
        return results

