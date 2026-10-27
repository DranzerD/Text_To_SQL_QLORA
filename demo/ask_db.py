"""Ask a SQLite database questions in plain English, answered by the fine-tuned model in Ollama.

    python demo/ask_db.py "Which 3 customers spent the most?"
    python demo/ask_db.py                                  # interactive
    python demo/ask_db.py --db path/to/your.db "How many rows are in each table?"
    python demo/ask_db.py --model gemma3:4b "..."          # compare with another Ollama model

Without --db it uses demo/shop.db, a small sample shop database created on first run.
The database is opened read-only, so generated SQL can never modify it.
"""
import argparse
import random
import sqlite3
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from sqlutils import build_prompt, extract_sql, get_schema, run_query

SAMPLE_DB = Path(__file__).resolve().parent / "shop.db"


def make_sample_db(path: Path) -> None:
    rng = random.Random(7)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date DATE);
        CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
                             order_date DATE, status TEXT);
        CREATE TABLE order_items (order_id INTEGER REFERENCES orders(order_id),
                                  product_id INTEGER REFERENCES products(product_id), quantity INTEGER);
    """)
    names = ["Asha", "Ben", "Chen", "Diya", "Elena", "Farhan", "Grace", "Hiro", "Isla", "Jay", "Kavya", "Leo"]
    cities = ["Hyderabad", "Bengaluru", "Mumbai", "Pune", "Chennai"]
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)",
                     [(i + 1, n, rng.choice(cities), f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
                      for i, n in enumerate(names)])
    products = [("Laptop", "Electronics", 899.0), ("Headphones", "Electronics", 79.0), ("Keyboard", "Electronics", 45.0),
                ("Desk", "Furniture", 230.0), ("Chair", "Furniture", 150.0), ("Notebook", "Stationery", 4.5),
                ("Pen set", "Stationery", 12.0), ("Monitor", "Electronics", 210.0)]
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [(i + 1, *p) for i, p in enumerate(products)])
    for order_id in range(1, 61):
        conn.execute("INSERT INTO orders VALUES (?, ?, ?, ?)",
                     (order_id, rng.randint(1, len(names)), f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                      rng.choice(["delivered"] * 4 + ["shipped", "cancelled"])))
        for product_id in rng.sample(range(1, len(products) + 1), rng.randint(1, 3)):
            conn.execute("INSERT INTO order_items VALUES (?, ?, ?)", (order_id, product_id, rng.randint(1, 4)))
    conn.commit()
    conn.close()


def ask_model(model: str, prompt: str) -> str:
    resp = requests.post(f"{C.OLLAMA_URL}/api/chat", json={
        "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
        "options": {"temperature": 0, "num_predict": C.GEN_MAX_NEW_TOKENS},
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def print_table(headers: list[str], rows: list[tuple], limit: int = 20) -> None:
    cells = [[str(h) for h in headers]] + [["" if v is None else str(v) for v in r] for r in rows[:limit]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for i, row in enumerate(cells):
        print("  " + " | ".join(v.ljust(w) for v, w in zip(row, widths)))
        if i == 0:
            print("  " + "-+-".join("-" * w for w in widths))
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more rows")


def answer(conn: sqlite3.Connection, schema: str, model: str, question: str) -> None:
    start = time.time()
    sql = extract_sql(ask_model(model, build_prompt(schema, question)))
    print(f"\nSQL ({time.time() - start:.1f}s):\n  {sql}\n")
    try:
        headers, rows = run_query(conn, sql, with_headers=True)
    except sqlite3.Error as e:
        print(f"  Query failed: {e}")
        return
    if headers:
        print_table(headers, rows)
    else:
        print("  (statement returned no columns)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("--db", type=Path, default=SAMPLE_DB)
    parser.add_argument("--model", default=C.OLLAMA_MODEL_NAME)
    args = parser.parse_args()

    if args.db == SAMPLE_DB and not SAMPLE_DB.exists():
        make_sample_db(SAMPLE_DB)
    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}")
    conn = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    schema = get_schema(conn)

    if args.question:
        answer(conn, schema, args.model, args.question)
        return
    print(f"Database: {args.db.name}\n{schema}\n\nAsk a question (empty line quits).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        answer(conn, schema, args.model, q)


if __name__ == "__main__":
    main()
