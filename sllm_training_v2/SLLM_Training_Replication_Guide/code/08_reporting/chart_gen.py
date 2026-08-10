import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

STATS_PATH = r"D:\Dev\sllm_training_v2\report_assets\stats.json"
OUT_DIR = r"D:\Dev\sllm_training_v2\report_assets"

with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = json.load(f)

# ---- palette (dataviz skill reference palette, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax, hide_y_spine=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_y_spine:
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


def hbar_chart(labels, values, title, fname, color=BLUE, value_fmt="{:,}", figsize=(9, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=12, fontweight="bold")
    maxv = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v + maxv * 0.012, i, value_fmt.format(v), va="center", fontsize=8.7, color=INK_SECONDARY)
    ax.set_xlim(0, maxv * 1.14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=160)
    plt.close(fig)


def hist_pair(data_a, title_a, data_b, title_b, suptitle, fname, bins=30, color=BLUE):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, t in zip(axes, [data_a, data_b], [title_a, title_b]):
        ax.hist(data, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        style_ax(ax, hide_y_spine=False)
        ax.spines["left"].set_color(BASELINE)
        ax.set_title(t, fontsize=11, color=INK_SECONDARY, loc="left")
    fig.suptitle(suptitle, fontsize=13, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=160)
    plt.close(fig)


# ---- 1. Top 20 table types (clean dataset) ----
tt_counter = stats["dataset"]["table_type_counter"]
top20 = sorted(tt_counter.items(), key=lambda kv: -kv[1])[:20]
labels = [k[:42] for k, _ in reversed(top20)]
values = [v for _, v in reversed(top20)]
hbar_chart(labels, values, "Top 20 table_type labels — clean training dataset (6,109 records)",
           "top20_table_types.png")

# ---- 2. row_type distribution ----
rt_counter = stats["dataset"]["row_type_counter"]
rt_sorted = sorted(rt_counter.items(), key=lambda kv: -kv[1])
labels = [k for k, _ in reversed(rt_sorted)]
values = [v for _, v in reversed(rt_sorted)]
hbar_chart(labels, values, "Row-type distribution across all labeled rows", "row_type_distribution.png",
           figsize=(8, 3.2))

# ---- 3. column_type distribution ----
ct_counter = stats["dataset"]["column_type_counter"]
ct_sorted = sorted(ct_counter.items(), key=lambda kv: -kv[1])[:15]
labels = [k[:30] for k, _ in reversed(ct_sorted)]
values = [v for _, v in reversed(ct_sorted)]
hbar_chart(labels, values, "Column-type distribution (top 15)", "column_type_distribution.png",
           color=ORANGE, figsize=(8, 5))

# ---- 4. row / col counts + prompt / completion length distributions (recompute from clean jsonl) ----
CLEAN_JSONL = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.clean.jsonl"
row_counts, col_counts, prompt_lens, completion_lens = [], [], [], []
with open(CLEAN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt_lens.append(len(obj.get("prompt", "")))
        completion_lens.append(len(obj.get("completion", "")))
        try:
            comp = json.loads(obj["completion"])
            row_counts.append(len(comp.get("rows", {})))
            col_counts.append(len(comp.get("columns", {})))
        except Exception:
            pass

hist_pair(row_counts, "Rows per table", col_counts, "Columns per table",
          "Table shape distribution — clean training dataset", "table_shape_hist.png", bins=40)

hist_pair(prompt_lens, "Prompt length (chars)", completion_lens, "Completion length (chars)",
          "Prompt / completion length distribution — clean training dataset",
          "prompt_completion_len_hist.png", bins=40, color=ORANGE)

# ---- 5. Top 20 documents by number of financial tables ----
doc_counts = stats["doc_table_counts"]
top_docs = sorted(doc_counts.items(), key=lambda kv: -kv[1])[:20]


def short_name(name, n=48):
    base = os.path.splitext(name)[0]
    return base if len(base) <= n else base[: n - 1] + "…"


labels = [short_name(k) for k, _ in reversed(top_docs)]
values = [v for _, v in reversed(top_docs)]
hbar_chart(labels, values, "Top 20 source documents by number of labeled financial tables",
           "top20_documents_by_tables.png", figsize=(9.5, 7.5))

# ---- 6. Distribution of tables-per-document across the whole corpus ----
all_doc_values = list(doc_counts.values())
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(all_doc_values, bins=range(0, max(all_doc_values) + 4, 2), color=BLUE, edgecolor=SURFACE,
        linewidth=0.4, zorder=3)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax, hide_y_spine=False)
ax.spines["left"].set_color(BASELINE)
ax.set_title(f"Financial tables per document — {len(all_doc_values)} documents", fontsize=12,
             color=INK, loc="left", fontweight="bold")
ax.set_xlabel("Tables per document", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("Document count", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "tables_per_document_hist.png"), dpi=160)
plt.close(fig)

print("Charts written to", OUT_DIR)
print("row_counts n=", len(row_counts), "prompt_lens n=", len(prompt_lens))
