# Results

Execution accuracy on 500 held-out questions from the `gretelai/synthetic_text_to_sql` test split (validated in SQLite). 95% Wilson confidence intervals.

| Model | Execution accuracy | 95% CI | Valid SQL | Sec / question |
|---|---|---|---|---|
| Qwen3-1.7B (base) | **65.2%** | 60.9% – 69.2% | 94.6% | 0.569 |
| Qwen3-1.7B + QLoRA (8000 ex.) | **75.4%** | 71.4% – 79.0% | 97.8% | 0.916 |
| Gemma 3 4B (Ollama) | **69.4%** | 65.2% – 73.3% | 97.6% | 3.84 |
| sql-qwen (Ollama, q8_0 GGUF) | **76.2%** | 72.3% – 79.7% | 97.8% | 2.579 |
| finetuned-n1000 | **70.0%** | 65.8% – 73.9% | 97.4% | 0.758 |
| finetuned-n2000 | **72.2%** | 68.1% – 75.9% | 97.4% | 0.797 |
| finetuned-n4000 | **74.2%** | 70.2% – 77.8% | 97.8% | 0.722 |

![Model comparison](charts/model_comparison.png)

## Training-data ablation

| Training examples | Execution accuracy | Val loss | Train time | Peak VRAM |
|---|---|---|---|---|
| 1,000 | 70.0% | 0.1366 | 19.5 min | 3.43 GB |
| 2,000 | 72.2% | 0.1273 | 29.0 min | 3.43 GB |
| 4,000 | 74.2% | 0.1186 | 39.9 min | 3.43 GB |
| 8,000 | 75.4% | 0.1126 | 79.9 min | 3.15 GB |

![Data ablation](charts/data_ablation.png)

## By query type

![By complexity](charts/by_complexity.png)

## Effect of showing sample rows in the prompt

Same 500 questions, same training recipe. The first version of the project showed only the CREATE statements; this version adds 3 example rows per table (results of the first version are kept in `results/v1_schema_only/`).

| Model | Schema only | + 3 sample rows | Change |
|---|---|---|---|
| Qwen3-1.7B (base) | 52.2% | **65.2%** | +13.0 points |
| Qwen3-1.7B + QLoRA (8000 ex.) | 63.8% | **75.4%** | +11.6 points |
| Gemma 3 4B (Ollama) | 57.8% | **69.4%** | +11.6 points |
| sql-qwen (Ollama, q8_0 GGUF) | 64.6% | **76.2%** | +11.6 points |

## Data validation

Every candidate example was executed in SQLite; only examples whose schema, data and reference query all ran (and returned rows) were kept.

| Split | Examined | Kept | Dropped (reason: count) |
|---|---|---|---|
| train | 13,996 | 8,200 | reference SQL fails in SQLite: 2699, not a read-only task: 1455, reference SQL returns no rows: 1274, context fails in SQLite: 359, not a SELECT query: 9 |
| test | 888 | 500 | reference SQL fails in SQLite: 165, not a read-only task: 103, reference SQL returns no rows: 94, context fails in SQLite: 26 |
