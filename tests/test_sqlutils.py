"""Tests for the grading logic. The headline accuracy numbers are only as good as this code.

    python -m pytest -q
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlutils import build_db, extract_sql, get_schema, grade, is_ordered, results_match, run_query

CONTEXT = """
CREATE TABLE sales (id INT, region TEXT, amount REAL);
INSERT INTO sales VALUES (1, 'North', 10.0), (2, 'South', 20.5), (3, 'North', 5.0);
"""


@pytest.mark.parametrize("reply, expected", [
    ("SELECT 1;", "SELECT 1"),
    ("```sql\nSELECT a FROM t;\n```", "SELECT a FROM t"),
    ("Here is the query:\nSELECT a FROM t;\nThis selects a.", "SELECT a FROM t"),
    ("<think>hmm</think>\n\nSELECT a FROM t", "SELECT a FROM t"),
    ("WITH x AS (SELECT 1) SELECT * FROM x;", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("SELECT a FROM t; SELECT b FROM t;", "SELECT a FROM t"),
])
def test_extract_sql(reply, expected):
    assert extract_sql(reply) == expected


def test_schema_without_sample_rows_has_no_data():
    schema = get_schema(build_db(CONTEXT), sample_rows=0)
    assert schema == "CREATE TABLE sales (id INT, region TEXT, amount REAL);"


def test_schema_sample_rows_show_stored_values():
    schema = get_schema(build_db(CONTEXT), sample_rows=2)
    assert schema.startswith("CREATE TABLE sales")
    assert "-- (1, 'North', 10.0)" in schema and "-- (2, 'South', 20.5)" in schema
    assert "5.0" not in schema                       # only the first 2 rows
    assert "INSERT" not in schema                    # shown as comments, not as statements


def test_schema_sample_rows_escape_and_truncate_text():
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE t (a TEXT, b TEXT); INSERT INTO t VALUES ('it''s', NULL);"
                       f"INSERT INTO t VALUES ('{'x' * 100}', 'y');")
    schema = get_schema(conn, sample_rows=2)
    assert "-- ('it''s', NULL)" in schema
    assert "x" * 100 not in schema and "..." in schema


def test_results_match_ignores_row_order_when_unordered():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)


def test_results_match_ignores_column_order_and_int_float():
    assert results_match([("North", 15)], [(15.0, "North")], ordered=False)


def test_results_match_counts_duplicates():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_is_ordered_only_looks_at_outer_query():
    assert is_ordered("SELECT a FROM t ORDER BY a")
    assert not is_ordered("SELECT a, ROW_NUMBER() OVER (ORDER BY a) FROM t")
    assert is_ordered("SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY SUM(amount) DESC")


def test_grade_statuses():
    gold = "SELECT region, SUM(amount) FROM sales GROUP BY region"
    assert grade(CONTEXT, gold, "SELECT SUM(amount), region FROM sales GROUP BY region")[0] == "correct"
    assert grade(CONTEXT, gold, "SELECT region, MAX(amount) FROM sales GROUP BY region")[0] == "wrong_result"
    assert grade(CONTEXT, gold, "SELECT nope FROM sales")[0] == "error"


def test_grade_is_isolated_from_destructive_predictions():
    # Each query runs on a fresh database, so a DROP in a prediction can't break the gold query.
    assert grade(CONTEXT, "SELECT COUNT(*) FROM sales", "DROP TABLE sales")[0] == "wrong_result"


def test_run_query_times_out():
    conn = build_db(CONTEXT)
    slow = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
    with pytest.raises(sqlite3.OperationalError):
        run_query(conn, slow, timeout_s=0.2)
