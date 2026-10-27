"""Step 1: build validated train / val / eval sets from gretelai/synthetic_text_to_sql.

Synthetic datasets contain broken examples, so every example is executed in
SQLite and kept only if:
  * its CREATE/INSERT context runs in SQLite,
  * its reference SQL is a read-only query that runs and returns rows,
  * the prompt + answer fit in MAX_SEQ_LENGTH tokens.
The prompt shows each table's CREATE statement plus SAMPLE_ROWS example rows (config.py).
Drop counts by reason are saved to results/data_stats.json.

    python scripts/01_prepare_data.py
"""
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from sqlutils import build_db, build_prompt, get_schema, run_query


def validate(row: dict, tokenizer) -> tuple[dict | None, str]:
    """Return (example, "ok") or (None, drop reason)."""
    if row["sql_task_type"] not in C.TASK_TYPES:
        return None, "not a read-only task"
    sql = row["sql"].strip().rstrip(";").strip()
    if not sql.upper().startswith(("SELECT", "WITH")):
        return None, "not a SELECT query"
    try:
        conn = build_db(row["sql_context"])
    except sqlite3.Error:
        return None, "context fails in SQLite"
    try:
        rows = run_query(conn, sql)
    except sqlite3.Error:
        return None, "reference SQL fails in SQLite"
    if not rows:
        return None, "reference SQL returns no rows"

    # Length is checked on the schema-only prompt, so the chosen examples are identical for every
    # SAMPLE_ROWS setting (and results with and without sample rows are directly comparable).
    # The prompt the model really sees, with sample rows, is checked against MAX_SEQ_LENGTH below.
    short_prompt = build_prompt(get_schema(conn, sample_rows=0), row["sql_prompt"])
    if len(tokenizer(short_prompt + sql, add_special_tokens=False)["input_ids"]) + 20 > C.SELECTION_MAX_TOKENS:
        return None, "too long"
    schema = get_schema(conn)
    full_prompt = build_prompt(schema, row["sql_prompt"])
    if len(tokenizer(full_prompt + sql, add_special_tokens=False)["input_ids"]) + 20 > C.MAX_SEQ_LENGTH:
        return None, "too long with sample rows"

    return {
        "id": row["id"],
        "domain": row["domain"],
        "complexity": row["sql_complexity"],
        "question": row["sql_prompt"],
        "schema": schema,
        "context": row["sql_context"],
        "sql": sql,
    }, "ok"


def collect(split, n_needed: int, tokenizer, desc: str) -> tuple[list[dict], Counter]:
    indices = list(range(len(split)))
    random.Random(C.SEED).shuffle(indices)
    kept, reasons = [], Counter()
    bar = tqdm(total=n_needed, desc=desc, unit="ex")
    for i in indices:
        example, reason = validate(split[i], tokenizer)
        reasons[reason] += 1
        if example:
            kept.append(example)
            bar.update(1)
            if len(kept) == n_needed:
                break
    bar.close()
    return kept, reasons


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    C.DATA_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    ds = load_dataset(C.DATASET)

    train, train_reasons = collect(ds["train"], C.TRAIN_POOL_SIZE + C.VAL_SIZE, tokenizer, "train")
    evals, eval_reasons = collect(ds["test"], C.EVAL_SIZE, tokenizer, "eval")
    # Shuffled order matters: the n1000 run trains on the first 1000 rows, n2000 on the first 2000, ...
    write_jsonl(C.TRAIN_FILE, train[: C.TRAIN_POOL_SIZE])
    write_jsonl(C.VAL_FILE, train[C.TRAIN_POOL_SIZE:])
    write_jsonl(C.EVAL_FILE, evals)

    def summary(reasons: Counter) -> dict:
        examined = sum(reasons.values())
        return {"examined": examined, "kept": reasons["ok"],
                "dropped": {k: v for k, v in reasons.most_common() if k != "ok"}}

    stats = {
        "train_split": summary(train_reasons),
        "test_split": summary(eval_reasons),
        "train_complexity": Counter(r["complexity"] for r in train[: C.TRAIN_POOL_SIZE]),
        "eval_complexity": Counter(r["complexity"] for r in evals),
    }
    C.DATA_STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {C.TRAIN_FILE.name} ({C.TRAIN_POOL_SIZE}), {C.VAL_FILE.name} ({C.VAL_SIZE}), {C.EVAL_FILE.name} ({len(evals)})")


if __name__ == "__main__":
    main()
