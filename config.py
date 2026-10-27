"""Central settings for the Text-to-SQL fine-tuning project.

Every script imports from here, so change things in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"      # big generated files (adapters, GGUF), git-ignored
RESULTS_DIR = ROOT / "results"     # small result files (metrics, charts), committed

SEED = 42

# ---------------------------------------------------------------- data
DATASET = "gretelai/synthetic_text_to_sql"
# Only read-only questions: their answers can be checked by comparing result rows.
TASK_TYPES = ["analytics and reporting", "data retrieval"]
TRAIN_POOL_SIZE = 8000     # validated training examples to keep (largest ablation size)
VAL_SIZE = 200             # extra validated examples for validation loss
EVAL_SIZE = 500            # validated examples from the dataset's *test* split
QUERY_TIMEOUT_S = 5        # kill generated queries that run too long (e.g. accidental cross joins)

# What the model is shown about the database: CREATE statements plus this many example rows per table.
# 0 reproduces the first version of this project (schema only).
SAMPLE_ROWS = 3
SAMPLE_VALUE_CHARS = 40    # long text values are cut so wide tables don't eat the prompt
SELECTION_MAX_TOKENS = 512 # examples are chosen by schema-only length, so every SAMPLE_ROWS setting uses the same questions

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
DATA_STATS_FILE = RESULTS_DIR / "data_stats.json"

# ---------------------------------------------------------------- training
BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTERS_DIR = OUTPUT_DIR / "adapters"
ABLATION_SIZES = [1000, 2000, 4000, 8000]
FINAL_RUN = "n8000"        # adapter that gets exported to Ollama

MAX_SEQ_LENGTH = 512       # lower to 384 if you run out of GPU memory
# Training examples longer than this are skipped. One long example makes PyTorch reserve more GPU
# memory, which then stays reserved; on a 4 GB card that spills into system RAM and slows training ~4-8x.
# 288 keeps the peak at what the schema-only version needed (its longest example was 279 tokens).
TRAIN_MAX_TOKENS = 288
LORA_R = 16                # lower to 8 if you run out of GPU memory
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8       # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

# ---------------------------------------------------------------- evaluation
EVAL_DIR = OUTPUT_DIR / "eval"            # per-example predictions
METRICS_DIR = RESULTS_DIR / "metrics"     # one summary JSON per evaluated model
EVAL_BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS = 256
OLLAMA_URL = "http://localhost:11434"

# ---------------------------------------------------------------- export
MERGED_DIR = OUTPUT_DIR / "merged"
LLAMA_CPP_DIR = ROOT / "llama.cpp"
# 8-bit GGUF straight from llama.cpp's converter (~1.8 GB, near-lossless, fits in 4 GB VRAM).
# Ollama can't quantize GGUF imports, and its safetensors quantizer needs MLX (not on Windows).
GGUF_OUTTYPE = "q8_0"
OLLAMA_MODEL_NAME = "sql-qwen"
GGUF_FILE = OUTPUT_DIR / f"{OLLAMA_MODEL_NAME}-{GGUF_OUTTYPE}.gguf"
MODELFILE = OUTPUT_DIR / "Modelfile"
