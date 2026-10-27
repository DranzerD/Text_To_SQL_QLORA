"""Step 5: turn the final adapter into an Ollama model.

1. Merge the LoRA adapter into the full-precision base model (on CPU).
2. Convert the merged model to an 8-bit GGUF with llama.cpp's converter script.
3. Write a Modelfile and run `ollama create`.

    python scripts/05_export_to_ollama.py
    python scripts/03_evaluate.py --model ollama:sql-qwen   # check the exported model kept its accuracy
"""
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# Ollama uses the chat template embedded in the GGUF (it ignores a Modelfile TEMPLATE),
# and Qwen3's stock template turns thinking on. This one matches training exactly:
# every answer starts after an empty <think></think> block, so the model never thinks out loud.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
)
ADAPTER_DIR = C.ADAPTERS_DIR / C.FINAL_RUN


def run(cmd: list[str], **kwargs) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def is_newer(output: Path, source: Path) -> bool:
    return output.exists() and output.stat().st_mtime > source.stat().st_mtime


def merge() -> None:
    if is_newer(C.MERGED_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"):
        print(f"Merged model in {C.MERGED_DIR} is up to date, skipping merge")
        return
    print("Merging adapter into base model on CPU (takes a few minutes) ...")
    base = AutoModelForCausalLM.from_pretrained(C.BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(C.MERGED_DIR, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(C.MERGED_DIR)


def convert_to_gguf() -> None:
    if is_newer(C.GGUF_FILE, C.MERGED_DIR / "config.json"):
        print(f"{C.GGUF_FILE.name} is up to date, skipping conversion")
        return
    if not C.LLAMA_CPP_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(C.LLAMA_CPP_DIR)])
    run([sys.executable, str(C.LLAMA_CPP_DIR / "convert_hf_to_gguf.py"), str(C.MERGED_DIR),
         "--outtype", C.GGUF_OUTTYPE, "--outfile", str(C.GGUF_FILE)])


def create_ollama_model() -> None:
    C.MODELFILE.write_text(
        f"FROM ./{C.GGUF_FILE.name}\n"
        'PARAMETER stop "<|im_start|>"\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0\n",  # SQL should be deterministic
        encoding="utf-8",
    )
    run(["ollama", "create", C.OLLAMA_MODEL_NAME, "-f", str(C.MODELFILE)], cwd=C.OUTPUT_DIR)


def main() -> None:
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        sys.exit(f"No adapter in {ADAPTER_DIR}. Run scripts/02_train.py --train-size {C.FINAL_RUN[1:]} first.")
    merge()
    convert_to_gguf()
    create_ollama_model()
    print(f"\nDone! Try it:  python demo/ask_db.py \"Which customers spent the most?\"")


if __name__ == "__main__":
    main()
