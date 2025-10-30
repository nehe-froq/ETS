from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def pick_text_columns(df: pd.DataFrame, override: Sequence[str] | None = None) -> List[str]:
    if override:
        return [c for c in override if c in df.columns]
    candidates: List[str] = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            # prefer columns with moderate cardinality and non-trivial text
            uniq = non_null.nunique(dropna=True)
            if 1 <= uniq <= max(2000, int(len(non_null) * 0.9)):
                # ensure average length looks textual
                sample = non_null.astype(str).head(200)
                avg_len = sample.map(len).mean()
                if avg_len >= 5:
                    candidates.append(col)
    preferred = [
        "_source.file_name_text",
        "_source.product_name_text",
        "_source.project_name_text",
        "_source.asset_type_name",
        "_source.customer",
    ]
    picked = [c for c in preferred if c in candidates]
    # add more up to a reasonable cap
    for c in candidates:
        if c not in picked:
            picked.append(c)
        if len(picked) >= 12:
            break
    return picked


def build_text(row: pd.Series, text_columns: Sequence[str], sep: str = " \n") -> str:
    parts: List[str] = []
    for c in text_columns:
        if c in row and pd.notna(row[c]):
            val = str(row[c]).strip()
            if val:
                parts.append(val)
    return sep.join(parts)


def pick_key_columns(df: pd.DataFrame, override: Sequence[str] | None = None) -> List[str]:
    if override:
        return [c for c in override if c in df.columns]
    preferred = ["_id", "_source.assetResourceId", "_source.file_name_text", "_source.filename"]
    return [c for c in preferred if c in df.columns] or [df.columns[0]]


def make_row_id(row: pd.Series, key_columns: Sequence[str]) -> str:
    vals = [str(row[c]) if c in row and pd.notna(row[c]) else "" for c in key_columns]
    base = "|".join(vals)
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


def compute_row_hash(row: pd.Series, text_columns: Sequence[str]) -> str:
    text = build_text(row, text_columns)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


