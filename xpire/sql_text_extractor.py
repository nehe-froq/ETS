from __future__ import annotations

import os
import re
from dataclasses import dataclass
from html import unescape as html_unescape
from io import TextIOWrapper
from typing import Generator, Iterable, Iterator, Optional, Tuple

import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress spurious warning when parsing plain strings that resemble paths
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


_INSERT_RE = re.compile(r"^\s*INSERT\s+INTO\s+`?(?P<table>[\w$]+)`?\s*\((?P<cols>[^)]*)\)\s*VALUES\s*(?P<vals>.*)$",
                        re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class TextRecord:
    source: str
    table_name: Optional[str]
    row_id: Optional[str]
    text: str


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    # HTML decode first
    s = html_unescape(raw)
    # Strip HTML tags if any
    s = BeautifulSoup(s, "html.parser").get_text(" ")
    s = re.sub(r"\s+", " ", s, flags=re.MULTILINE).strip()
    return s


def _looks_textual(value: str) -> bool:
    if not value:
        return False
    if len(value) < 10:
        return False
    # Reject if mostly digits/punctuation
    letters = sum(c.isalpha() for c in value)
    return letters >= max(5, len(value) * 0.3)


def _parse_value_tuples(values_blob: str) -> Iterator[Tuple[str, ...]]:
    """
    Parse VALUES blob like: (1,'a','b'),(2,'c'),(...);
    Returns tuples of column string values (unquoted, unescaped for text fields only). Non-string cells are kept as raw.
    """
    i = 0
    n = len(values_blob)
    current: list[str] = []
    in_tuple = False
    in_string = False
    escape = False
    buf: list[str] = []
    tuples: list[Tuple[str, ...]] = []

    def flush_cell():
        s = "".join(buf)
        buf.clear()
        current.append(s)

    while i < n:
        ch = values_blob[i]
        if not in_tuple:
            if ch == '(':
                in_tuple = True
                current = []
            i += 1
            continue
        # inside a tuple
        if in_string:
            if escape:
                buf.append(ch)
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == "'":
                in_string = False
            else:
                buf.append(ch)
            i += 1
            continue
        # not in string
        if ch == "'":
            in_string = True
            i += 1
            continue
        if ch == ',':
            flush_cell()
            i += 1
            continue
        if ch == ')':
            flush_cell()
            tuples.append(tuple(current))
            current = []
            in_tuple = False
            # advance to next tuple if comma follows
            i += 1
            # skip trailing comma
            while i < n and values_blob[i].isspace():
                i += 1
            if i < n and values_blob[i] == ',':
                i += 1
            continue
        # general char for non-string cell
        buf.append(ch)
        i += 1
    return iter(tuples)


def _maybe_row_id(cols_csv: str, row: Tuple[str, ...]) -> Optional[str]:
    try:
        col_names = [c.strip().strip('`') for c in cols_csv.split(',')]
    except Exception:
        return None
    if not col_names or not row:
        return None
    # heuristics: prefer id-like columns
    preferred = [i for i, c in enumerate(col_names) if c.lower() in {"id", "pk", "uuid"}]
    idx = preferred[0] if preferred else 0
    val = row[idx].strip()
    return val if val else None


def extract_text_from_sql(sql_path: str, min_text_len: int = 10) -> Iterator[TextRecord]:
    source = os.path.abspath(sql_path)
    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        in_block_comment = False
        insert_buffer: list[str] = []
        for raw_line in f:
            line = raw_line
            # handle block comments
            if in_block_comment:
                end = line.find("*/")
                if end != -1:
                    comment = line[:end]
                    cleaned = _clean_text(comment)
                    if len(cleaned) >= min_text_len and _looks_textual(cleaned):
                        yield TextRecord(source, None, None, cleaned)
                    line = line[end + 2 :]
                    in_block_comment = False
                else:
                    cleaned = _clean_text(line)
                    if len(cleaned) >= min_text_len and _looks_textual(cleaned):
                        yield TextRecord(source, None, None, cleaned)
                    continue

            # single-line comments
            if line.lstrip().startswith("--"):
                cleaned = _clean_text(line.lstrip()[2:])
                if len(cleaned) >= min_text_len and _looks_textual(cleaned):
                    yield TextRecord(source, None, None, cleaned)
                continue
            if "/*" in line:
                start = line.index("/*")
                before = line[:start]
                after = line[start + 2 :]
                # emit text before if any
                if before.strip():
                    cleaned = _clean_text(before)
                    if len(cleaned) >= min_text_len and _looks_textual(cleaned):
                        yield TextRecord(source, None, None, cleaned)
                # check if same-line end
                endpos = after.find("*/")
                if endpos != -1:
                    comment = after[:endpos]
                    cleaned = _clean_text(comment)
                    if len(cleaned) >= min_text_len and _looks_textual(cleaned):
                        yield TextRecord(source, None, None, cleaned)
                    line = after[endpos + 2 :]
                else:
                    in_block_comment = True
                    continue

            # handle INSERT buffering until semicolon
            if insert_buffer or line.lstrip().upper().startswith("INSERT"):
                insert_buffer.append(line)
                if ";" in line:
                    statement = "".join(insert_buffer)
                    insert_buffer.clear()
                    m = _INSERT_RE.match(statement.strip())
                    if not m:
                        continue
                    table = m.group("table")
                    cols_csv = m.group("cols")
                    vals = m.group("vals")
                    # strip trailing semicolon
                    vals = vals.rsplit(";", 1)[0]
                    try:
                        for row in _parse_value_tuples(vals):
                            # gather all textual cells
                            texts: list[str] = []
                            for cell in row:
                                cell_str = cell.strip()
                                # strip outer quotes if parser left them
                                if cell_str.startswith("'") and cell_str.endswith("'") and len(cell_str) >= 2:
                                    cell_str = cell_str[1:-1]
                                cell_str = _clean_text(cell_str)
                                if len(cell_str) >= min_text_len and _looks_textual(cell_str):
                                    texts.append(cell_str)
                            if not texts:
                                continue
                            rid = _maybe_row_id(cols_csv, row)
                            yield TextRecord(source, table, rid, " \n".join(texts))
                    except Exception:
                        # be forgiving; skip malformed
                        pass
                continue

            # other lines are ignored


