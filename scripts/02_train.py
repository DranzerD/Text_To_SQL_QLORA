"""Step 2: QLoRA fine-tune Qwen3-1.7B for Text-to-SQL.

The base model is loaded in 4-bit (QLoRA) and only small LoRA adapter
matrices are trained, which is what lets this fit in 4 GB of VRAM.
Loss is computed on the SQL answer only, not on the prompt.

    python scripts/02_train.py --train-size 8000           # -> outputs/adapters/n8000
    python scripts/02_train.py --train-size 1000           # ablation run
    python scripts/02_train.py --train-size 200 --max-steps 10 --run-name smoke
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from sqlutils import build_prompt


def chat_prompt(tokenizer, schema: str, question: str) -> str:
    """Chat-formatted prompt ending right where the assistant starts talking.

    enable_thinking=False makes Qwen3 emit an empty <think></think> block, so the
    model learns to answer directly instead of reasoning first.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(schema, question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenize(example: dict, tokenizer) -> dict:
    prompt_ids = tokenizer(chat_prompt(tokenizer, example["schema"], example["question"]),
                           add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["sql"] + ";<|im_end|>", add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[: C.MAX_SEQ_LENGTH]
    # -100 = "ignore in the loss", so the model is only graded on the SQL.
    labels = ([-100] * len(prompt_ids) + answer_ids)[: C.MAX_SEQ_LENGTH]
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(x["input_ids"]) for x in batch)
        ids, labels, mask = [], [], []
        for x in batch:
            pad = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            mask.append([1] * len(x["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=C.TRAIN_POOL_SIZE, help="use the first N training examples")
    parser.add_argument("--run-name", default=None, help="adapter folder name (default: n<train-size>)")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (smoke test)")
    args = parser.parse_args()
    run_name = args.run_name or f"n{args.train_size}"
    out_dir = C.ADAPTERS_DIR / run_name

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available to PyTorch. See README > Troubleshooting.")
    if not C.TRAIN_FILE.exists():
        sys.exit(f"{C.TRAIN_FILE} not found. Run scripts/01_prepare_data.py first.")

    train_rows = load_jsonl(C.TRAIN_FILE)[: args.train_size]
    val_rows = load_jsonl(C.VAL_FILE)
    print(f"Run '{run_name}': {len(train_rows)} train / {len(val_rows)} validation examples")

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    keep = ["schema", "question", "sql"]
    train_ds = Dataset.from_list([{k: r[k] for k in keep} for r in train_rows]).map(
        tokenize, fn_kwargs={"tokenizer": tokenizer}, remove_columns=keep)
    val_ds = Dataset.from_list([{k: r[k] for k in keep} for r in val_rows]).map(
        tokenize, fn_kwargs={"tokenizer": tokenizer}, remove_columns=keep)

    n_before = len(train_ds)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= C.TRAIN_MAX_TOKENS)
    print(f"Skipped {n_before - len(train_ds)} examples longer than {C.TRAIN_MAX_TOKENS} tokens (keeps GPU memory under 4 GB)")

    print("\nExample of what the model is trained on:\n" + "-" * 60)
    print(tokenizer.decode(train_ds[0]["input_ids"]))
    print("-" * 60 + "\n")

    print(f"Loading {C.BASE_MODEL} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(C.OUTPUT_DIR / "checkpoints"),
        num_train_epochs=C.NUM_EPOCHS,
        max_steps=args.max_steps,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.BATCH_SIZE,
        gradient_accumulation_steps=C.GRAD_ACCUM_STEPS,
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=C.SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    val_loss = trainer.evaluate()["eval_loss"]

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    stats = {
        "run": run_name,
        "train_examples": len(train_rows),
        "steps": trainer.state.global_step,
        "train_minutes": round(minutes, 1),
        "val_loss": round(val_loss, 4),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h],
    }
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nTraining took {minutes:.1f} min, final validation loss {val_loss:.4f}, "
          f"peak GPU memory {stats['peak_vram_gb']} GB")
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
