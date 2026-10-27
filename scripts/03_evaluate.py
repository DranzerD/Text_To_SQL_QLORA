"""Step 3: measure execution accuracy on the 500 held-out eval questions.

A prediction counts as correct when it runs in SQLite and returns the same rows
as the reference query (column order ignored; row order checked only if the
reference query has ORDER BY).

    python scripts/03_evaluate.py --model base                 # Qwen3-1.7B, no fine-tuning
    python scripts/03_evaluate.py --model adapter:n8000        # fine-tuned adapter
    python scripts/03_evaluate.py --model ollama:gemma3:4b     # any local Ollama model
    python scripts/03_evaluate.py --model base --n 50          # quick check on 50 questions

Writes per-example predictions to outputs/eval/<name>.jsonl and a summary to
results/metrics/<name>.json (or outputs/eval/ for --n quick checks).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from sqlutils import build_prompt, extract_sql, grade


class HFModel:
    """Qwen3-1.7B in 4-bit, optionally with a LoRA adapter. Greedy, batched generation."""

    def __init__(self, adapter: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
        self.tokenizer.padding_side = "left"  # decoder-only models must be left-padded for batched generation
        self.model = AutoModelForCausalLM.from_pretrained(
            C.BASE_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, C.ADAPTERS_DIR / adapter)
        self.model.eval()

    def generate(self, prompts: list[str]) -> list[str]:
        texts = [
            self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            for p in prompts
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=C.GEN_MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class OllamaModel:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompts: list[str]) -> list[str]:
        replies = []
        for p in prompts:
            resp = requests.post(f"{C.OLLAMA_URL}/api/chat", json={
                "model": self.name,
                "messages": [{"role": "user", "content": p}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": C.GEN_MAX_NEW_TOKENS},
            }, timeout=600)
            resp.raise_for_status()
            replies.append(resp.json()["message"]["content"])
        return replies


def load_model(spec: str):
    """Returns (model, short name used for output files)."""
    if spec == "base":
        return HFModel(adapter=None), "base"
    if spec.startswith("adapter:"):
        run = spec.split(":", 1)[1]
        if not (C.ADAPTERS_DIR / run / "adapter_config.json").exists():
            sys.exit(f"No adapter {C.ADAPTERS_DIR / run}. Train it with scripts/02_train.py first.")
        return HFModel(adapter=run), f"finetuned-{run}"
    if spec.startswith("ollama:"):
        name = spec.split(":", 1)[1]
        return OllamaModel(name), "ollama-" + name.replace(":", "-")
    sys.exit(f"Unknown --model {spec!r}. Use base, adapter:<run>, or ollama:<model>.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base | adapter:<run> | ollama:<model>")
    parser.add_argument("--n", type=int, default=None, help="only evaluate the first N questions")
    args = parser.parse_args()

    with C.EVAL_FILE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.n]
    model, name = load_model(args.model)
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    start = time.time()
    for i in tqdm(range(0, len(rows), C.EVAL_BATCH_SIZE), desc=name, unit="batch"):
        batch = rows[i: i + C.EVAL_BATCH_SIZE]
        replies = model.generate([build_prompt(r["schema"], r["question"]) for r in batch])
        for r, reply in zip(batch, replies):
            pred = extract_sql(reply)
            status, detail = grade(r["context"], r["sql"], pred)
            records.append({"id": r["id"], "complexity": r["complexity"], "question": r["question"],
                            "gold": r["sql"], "pred": pred, "raw": reply, "status": status, "error": detail})
    seconds = time.time() - start

    by_complexity = defaultdict(lambda: [0, 0])
    for rec in records:
        by_complexity[rec["complexity"]][0] += rec["status"] == "correct"
        by_complexity[rec["complexity"]][1] += 1
    n = len(records)
    metrics = {
        "model": name,
        "n": n,
        "execution_accuracy": round(sum(r["status"] == "correct" for r in records) / n, 4),
        "valid_sql_rate": round(sum(r["status"] != "error" for r in records) / n, 4),
        "by_complexity": {k: {"correct": c, "n": t, "accuracy": round(c / t, 4)}
                          for k, (c, t) in sorted(by_complexity.items(), key=lambda kv: -kv[1][1])},
        "seconds_per_question": round(seconds / n, 3),
    }

    with (C.EVAL_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Partial runs are quick checks: keep them out of results/ so the report only sees full evaluations.
    metrics_file = C.METRICS_DIR / f"{name}.json" if args.n is None else C.EVAL_DIR / f"{name}-n{args.n}.metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n{name}: execution accuracy {metrics['execution_accuracy']:.1%}, "
          f"valid SQL {metrics['valid_sql_rate']:.1%}, {metrics['seconds_per_question']}s/question")
    for k, v in metrics["by_complexity"].items():
        print(f"  {k:18s} {v['accuracy']:.1%}  ({v['correct']}/{v['n']})")


if __name__ == "__main__":
    main()
