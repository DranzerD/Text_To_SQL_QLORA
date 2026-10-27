"""Shared Text-to-SQL helpers: prompt format, SQLite execution, and result comparison.

Used by data prep, training, evaluation and the demo, so the model is always
prompted and graded the same way.
"""
import re
import sqlite3
import time
from collections import Counter

import config as C


def build_prompt(schema: str, question: str) -> str:
    return (
        f"Database schema (SQLite):\n{schema}\n\n"
        f"Question: {question}\n\n"
        "Write a single SQLite query that answers the question. Reply with only the SQL."
    )


def build_db(context: str) -> sqlite3.Connection:
    """Fresh in-memory database from a dataset example's CREATE/INSERT statements."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(context)
    return conn


def _sql_literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    text = str(v)
    if len(text) > C.SAMPLE_VALUE_CHARS:
        text = text[: C.SAMPLE_VALUE_CHARS] + "..."
    return "'" + text.replace("'", "''") + "'"


def get_schema(conn: sqlite3.Connection, sample_rows: int = C.SAMPLE_ROWS) -> str:
    """CREATE statements, each followed by its first few rows as comments.

    The rows show how values are really stored ('WA' vs 'Washington', 'delivered' vs
    'Delivered'), which the CREATE statement alone can't tell the model. Pass 0 for CREATE only.
    """
    objects = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    ).fetchall()
    parts = []
    for kind, name, create in objects:
        parts.append(create.strip().rstrip(";") + ";")
        if kind != "table" or sample_rows <= 0:
            continue
        sample = conn.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}').fetchall()
        if sample:
            parts.append(f"-- Example rows from {name}:")
            parts.extend("-- (" + ", ".join(_sql_literal(v) for v in row) + ")" for row in sample)
    return "\n".join(parts)


def run_query(conn: sqlite3.Connection, sql: str, timeout_s: float = C.QUERY_TIMEOUT_S, with_headers: bool = False):
    """Run a query, aborting it after timeout_s. Raises sqlite3.Error on failure.

    Returns the rows, or (column names, rows) if with_headers is True.
    """
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.set_progress_handler(None, 0)
    if with_headers:
        return [d[0] for d in cur.description or []], rows
    return rows


_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_START_RE = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL query out of a model reply (handles ```sql fences and chatty preambles)."""
    text = _THINK_RE.sub("", text).strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = _START_RE.search(text)
    if start:
        text = text[start.start():]
    return text.split(";")[0].strip()


def _norm_value(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 4)
    return v


def _norm_row(row: tuple) -> tuple:
    # Sorting values inside a row makes the comparison ignore column order.
    return tuple(sorted((_norm_value(v) for v in row), key=lambda x: (type(x).__name__, str(x))))


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    """Execution-accuracy comparison: same rows (as a multiset, or as a list if the gold query sorts)."""
    g, p = [_norm_row(r) for r in gold], [_norm_row(r) for r in pred]
    return g == p if ordered else Counter(g) == Counter(p)


_PARENS_RE = re.compile(r"\([^()]*\)")


def is_ordered(sql: str) -> bool:
    """True if the *outer* query has ORDER BY (not just a subquery or window function)."""
    outer = sql
    while _PARENS_RE.search(outer):
        outer = _PARENS_RE.sub("", outer)
    return re.search(r"\border\s+by\b", outer, re.IGNORECASE) is not None


def grade(context: str, gold_sql: str, pred_sql: str) -> tuple[str, str]:
    """Return (status, detail): status is 'correct', 'wrong_result' or 'error'."""
    gold_rows = run_query(build_db(context), gold_sql)
    try:
        pred_rows = run_query(build_db(context), pred_sql)
    except (sqlite3.Error, ValueError) as e:
        return "error", str(e)[:200]
    if results_match(gold_rows, pred_rows, ordered=is_ordered(gold_sql)):
        return "correct", ""
    return "wrong_result", ""
