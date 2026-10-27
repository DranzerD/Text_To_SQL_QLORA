"""Step 4: turn the metrics into results/RESULTS.md and three charts.

    python scripts/04_report.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# Reference data-viz palette (light mode).
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ACCENT, SECOND, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8b2"

DISPLAY = {
    "base": "Qwen3-1.7B (base)",
    f"finetuned-{C.FINAL_RUN}": f"Qwen3-1.7B + QLoRA ({C.FINAL_RUN[1:]} ex.)",
    "ollama-gemma3-4b": "Gemma 3 4B (Ollama)",
    f"ollama-{C.OLLAMA_MODEL_NAME}": f"{C.OLLAMA_MODEL_NAME} (Ollama, {C.GGUF_OUTTYPE} GGUF)",
}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy measured on n questions."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_facecolor(SURFACE)


def load_metrics() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(C.METRICS_DIR.glob("*.json"))}


def chart_models(metrics: dict, path: Path) -> None:
    names = [k for k in DISPLAY if k in metrics]
    accs = [metrics[k]["execution_accuracy"] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.4), facecolor=SURFACE)
    colors = [ACCENT if k.startswith("finetuned") or k == f"ollama-{C.OLLAMA_MODEL_NAME}" else NEUTRAL for k in names]
    y = range(len(names))[::-1]
    ax.barh(list(y), accs, color=colors, height=0.6)
    for yi, a in zip(y, accs):
        ax.text(a + 1, yi, f"{a:.1f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks(list(y), [DISPLAY[k] for k in names], color=INK, fontsize=10)
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Execution accuracy on 500 held-out questions (%)", color=INK_2)
    style(ax, "Fine-tuned 1.7B model vs. base and a larger model")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_ablation(metrics: dict, train_stats: dict, path: Path) -> None:
    sizes = [n for n in C.ABLATION_SIZES if f"finetuned-n{n}" in metrics]
    accs = [metrics[f"finetuned-n{n}"]["execution_accuracy"] * 100 for n in sizes]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    if "base" in metrics:
        base = metrics["base"]["execution_accuracy"] * 100
        ax.axhline(base, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5)
        ax.text(sizes[-1], base - 2.5, f"base model {base:.1f}%", color=INK_2, ha="right", va="top", fontsize=9)
    ax.plot(sizes, accs, color=ACCENT, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    for n, a in zip(sizes, accs):
        minutes = train_stats.get(f"n{n}", {}).get("train_minutes")
        label = f"{a:.1f}%" + (f"\n{minutes:.0f} min" if minutes else "")
        ax.annotate(label, (n, a), textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes], color=INK)
    ax.set_ylim(min(accs + [base if "base" in metrics else 100]) - 10, min(100, max(accs) + 12))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training examples (1 epoch each, label = training time on RTX 3050 4 GB)", color=INK_2)
    ax.set_ylabel("Execution accuracy (%)", color=INK_2)
    style(ax, "How much data does the fine-tune need?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_complexity(metrics: dict, path: Path) -> None:
    base, tuned = metrics.get("base"), metrics.get(f"finetuned-{C.FINAL_RUN}")
    if not (base and tuned):
        return
    cats = [k for k, v in tuned["by_complexity"].items() if v["n"] >= 10]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    for offset, m, color, label in ((-w / 2, base, NEUTRAL, DISPLAY["base"]),
                                    (w / 2, tuned, ACCENT, DISPLAY[f"finetuned-{C.FINAL_RUN}"])):
        vals = [m["by_complexity"].get(c, {"accuracy": 0})["accuracy"] * 100 for c in cats]
        bars = ax.bar([i + offset for i in x], vals, width=w - 0.03, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(list(x), [f"{c}\n(n={tuned['by_complexity'][c]['n']})" for c in cats], color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Execution accuracy (%)", color=INK_2)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    style(ax, "Accuracy by query type")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    if not metrics:
        sys.exit(f"No metrics in {C.METRICS_DIR}. Run scripts/03_evaluate.py first.")
    train_stats = {}
    for p in C.ADAPTERS_DIR.glob("*/train_stats.json"):
        s = json.loads(p.read_text(encoding="utf-8"))
        train_stats[s["run"]] = s
    data_stats = json.loads(C.DATA_STATS_FILE.read_text(encoding="utf-8")) if C.DATA_STATS_FILE.exists() else None

    charts = C.RESULTS_DIR / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    chart_models(metrics, charts / "model_comparison.png")
    chart_ablation(metrics, train_stats, charts / "data_ablation.png")
    chart_complexity(metrics, charts / "by_complexity.png")

    lines = ["# Results", "",
             f"Execution accuracy on {next(iter(metrics.values()))['n']} held-out questions from the "
             f"`{C.DATASET}` test split (validated in SQLite). 95% Wilson confidence intervals.", "",
             "| Model | Execution accuracy | 95% CI | Valid SQL | Sec / question |", "|---|---|---|---|---|"]
    order = list(DISPLAY) + sorted(k for k in metrics if k not in DISPLAY)
    for k in order:
        if k not in metrics:
            continue
        m = metrics[k]
        lo, hi = wilson(m["execution_accuracy"], m["n"])
        lines.append(f"| {DISPLAY.get(k, k)} | **{m['execution_accuracy']:.1%}** | {lo:.1%} – {hi:.1%} | "
                     f"{m['valid_sql_rate']:.1%} | {m['seconds_per_question']} |")
    lines += ["", "![Model comparison](charts/model_comparison.png)", "",
              "## Training-data ablation", "",
              "| Training examples | Execution accuracy | Val loss | Train time | Peak VRAM |", "|---|---|---|---|---|"]
    for n in C.ABLATION_SIZES:
        k, s = f"finetuned-n{n}", train_stats.get(f"n{n}", {})
        if k in metrics:
            lines.append(f"| {n:,} | {metrics[k]['execution_accuracy']:.1%} | {s.get('val_loss', '–')} | "
                         f"{s.get('train_minutes', '–')} min | {s.get('peak_vram_gb', '–')} GB |")
    lines += ["", "![Data ablation](charts/data_ablation.png)", "",
              "## By query type", "", "![By complexity](charts/by_complexity.png)", ""]
    v1_dir = C.RESULTS_DIR / "v1_schema_only" / "metrics"
    if C.SAMPLE_ROWS and v1_dir.is_dir():
        v1 = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in v1_dir.glob("*.json")}
        lines += ["## Effect of showing sample rows in the prompt", "",
                  f"Same 500 questions, same training recipe. The first version of the project showed only the CREATE "
                  f"statements; this version adds {C.SAMPLE_ROWS} example rows per table (results of the first version "
                  "are kept in `results/v1_schema_only/`).", "",
                  f"| Model | Schema only | + {C.SAMPLE_ROWS} sample rows | Change |", "|---|---|---|---|"]
        for k in DISPLAY:
            if k in metrics and k in v1:
                a, b = v1[k]["execution_accuracy"], metrics[k]["execution_accuracy"]
                lines.append(f"| {DISPLAY[k]} | {a:.1%} | **{b:.1%}** | {100 * (b - a):+.1f} points |")
        lines.append("")
    if data_stats:
        lines += ["## Data validation", "",
                  "Every candidate example was executed in SQLite; only examples whose schema, data and reference "
                  "query all ran (and returned rows) were kept.", "",
                  "| Split | Examined | Kept | Dropped (reason: count) |", "|---|---|---|---|"]
        for split in ("train_split", "test_split"):
            s = data_stats[split]
            reasons = ", ".join(f"{k}: {v}" for k, v in s["dropped"].items())
            lines.append(f"| {split.replace('_split', '')} | {s['examined']:,} | {s['kept']:,} | {reasons} |")
        lines.append("")

    out = C.RESULTS_DIR / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out} and charts in {charts}")


if __name__ == "__main__":
    main()
