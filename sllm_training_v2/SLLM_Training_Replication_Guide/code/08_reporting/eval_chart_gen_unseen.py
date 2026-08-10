import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STATS_PATH = r"D:\Dev\sllm_training_v2\report_assets\eval_stats_unseen.json"
OUT_DIR = r"D:\Dev\sllm_training_v2\report_assets"

with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = json.load(f)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"

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


# ---- 1. grouped overall metrics bar chart ----
GROUPS = {
    "Table type": (BLUE, [
        ("table_type_lexical_match_rate", "table_type — lexical match"),
        ("table_type_embedding_match_rate", "table_type — embedding match"),
    ]),
    "Row-level": (ORANGE, [
        ("row_type_accuracy", "row_type accuracy"),
        ("row_is_signed_negative_accuracy", "is_signed_negative accuracy"),
        ("row_direction_accuracy", "direction accuracy"),
        ("row_note_ref_value_accuracy", "note_ref_value accuracy"),
        ("row_contributing_rows_accuracy", "contributing_rows accuracy"),
    ]),
    "Column-level": (AQUA, [
        ("column_type_accuracy", "column_type accuracy"),
        ("column_direction_accuracy", "direction accuracy"),
        ("column_contributing_columns_accuracy", "contributing_columns accuracy"),
    ]),
    "Cell-level": (YELLOW, [
        ("cell_concept_id_lexical_accuracy", "concept_id — lexical"),
        ("cell_concept_id_embedding_accuracy", "concept_id — embedding"),
        ("cell_unit_accuracy", "unit accuracy"),
        ("cell_scale_accuracy", "scale accuracy"),
        ("cell_scale_multiplier_accuracy", "scale_multiplier accuracy"),
        ("cell_reporting_period_accuracy", "reporting_period accuracy"),
        ("cell_reporting_period_normalized_accuracy", "reporting_period (normalized)"),
        ("cell_concept_meaning_lexical_match_rate", "concept_meaning — lexical match"),
        ("cell_concept_meaning_avg_lexical_similarity", "concept_meaning — lexical similarity"),
        ("cell_concept_meaning_embedding_match_rate", "concept_meaning — embedding match"),
        ("cell_concept_meaning_avg_embedding_similarity", "concept_meaning — embedding similarity"),
    ]),
}

wo = stats["weighted_overall"]
rows = []
for group, (color, fields) in GROUPS.items():
    for key, label in fields:
        v = wo.get(key)
        if v is not None:
            rows.append((group, color, label, v))

rows.sort(key=lambda r: r[3], reverse=True)

fig, ax = plt.subplots(figsize=(10, 9.5))
y_labels = [r[2] for r in rows]
y = range(len(rows))
ax.barh(list(y), [r[3] for r in rows], color=[r[1] for r in rows], height=0.62, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(y_labels, fontsize=9)
ax.invert_yaxis()
ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_xlim(0, 1.12)
for i, r in enumerate(rows):
    ax.text(r[3] + 0.015, i, f"{r[3]*100:.1f}%", va="center", fontsize=8.3, color=INK_SECONDARY)
ax.set_title(f"Reference (Azure) vs. quantized local model — genuinely unseen documents only\n"
             f"({stats['total_tables_compared']} tables, {stats['n_jobs_with_compare']} jobs, job-weighted average, best → worst)",
             fontsize=12.5, color=INK, loc="left", fontweight="bold", pad=12)
handles = [mpatches.Patch(color=c, label=g) for g, (c, _) in GROUPS.items()]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_overall_metrics.png"), dpi=160)
plt.close(fig)

# ---- 2. distribution histograms: row_type_accuracy / column_type_accuracy ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, vals, t, color in zip(
    axes,
    [stats["row_type_acc_vals"], stats["column_type_acc_vals"]],
    [f"Per-table row_type accuracy (n={len(stats['row_type_acc_vals'])})",
     f"Per-table column_type accuracy (n={len(stats['column_type_acc_vals'])})"],
    [ORANGE, AQUA],
):
    ax.hist(vals, bins=20, range=(0, 1), color=color, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax, hide_y_spine=False)
    ax.spines["left"].set_color(BASELINE)
    ax.set_title(t, fontsize=10.5, color=INK_SECONDARY, loc="left")
    ax.set_xlim(0, 1)
fig.suptitle("Structural accuracy distribution — unseen documents only", fontsize=13, color=INK, x=0.02,
             ha="left", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_structural_accuracy_hist.png"), dpi=160)
plt.close(fig)

# ---- 3. distribution histograms: concept_id lexical vs embedding accuracy ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, vals, t in zip(
    axes,
    [stats["concept_id_lex_vals"], stats["concept_id_emb_vals"]],
    [f"concept_id — lexical accuracy (n={len(stats['concept_id_lex_vals'])})",
     f"concept_id — embedding accuracy (n={len(stats['concept_id_emb_vals'])})"],
):
    ax.hist(vals, bins=20, range=(0, 1), color=YELLOW, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    style_ax(ax, hide_y_spine=False)
    ax.spines["left"].set_color(BASELINE)
    ax.set_title(t, fontsize=10.5, color=INK_SECONDARY, loc="left")
    ax.set_xlim(0, 1)
fig.suptitle("Cell concept-tagging accuracy distribution — unseen documents only", fontsize=13, color=INK, x=0.02,
             ha="left", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_concept_id_hist.png"), dpi=160)
plt.close(fig)

# ---- 4a. best documents by mean row_type_accuracy (min 3 tables) ----
best_docs = [d for d in stats["doc_scores_best20"] if d["n_tables"] >= 3][:15]
labels = [f"{d['docname'][:38]} (n={d['n_tables']})" for d in best_docs]
values = [d["mean_row_type_accuracy"] for d in best_docs]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = range(len(labels))
ax.barh(list(y), values, color=AQUA, height=0.6, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_xlim(0, 1.1)
for i, v in enumerate(values):
    ax.text(v + 0.015, i, f"{v*100:.1f}%", va="center", fontsize=8.5, color=INK_SECONDARY)
ax.set_title("Highest-scoring unseen documents by mean row_type accuracy (≥3 tables compared)",
             fontsize=12, color=INK, loc="left", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_best_documents.png"), dpi=160)
plt.close(fig)

# ---- 4b. best individual tables: perfect score, ranked by table size ----
best_tables = stats["best_tables"][:15]
labels = [f"{t['docname'][:30]} / {t['table_id']} ({t['row_count_compared']} rows)" for t in best_tables]
values = [t["cell_count_compared"] for t in best_tables]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = range(len(labels))
ax.barh(list(y), values, color=AQUA, height=0.6, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8.7)
ax.invert_yaxis()
ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
for i, v in enumerate(values):
    ax.text(v + max(values) * 0.015, i, f"{v} cells, 100%", va="center", fontsize=8.3, color=INK_SECONDARY)
ax.set_xlim(0, max(values) * 1.28)
ax.set_title("Largest unseen-document tables scored with a perfect clean sweep\n(row_type, column_type, concept_id all 100% correct)",
             fontsize=12, color=INK, loc="left", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_best_tables.png"), dpi=160)
plt.close(fig)

# ---- 4c. does table complexity (row count) correlate with accuracy? ----
buckets = stats["complexity_buckets"]
labels = [f"{b['label']}\n({b['row_count_range'][0]}-{b['row_count_range'][1]} rows, n={b['n_tables']})" for b in buckets]
values = [b["mean_row_type_accuracy"] for b in buckets]
fig, ax = plt.subplots(figsize=(8, 4.4))
x = range(len(labels))
ax.bar(list(x), values, color=BLUE, width=0.55, zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8.8)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax, hide_y_spine=False)
ax.spines["left"].set_color(BASELINE)
ax.set_ylim(0, 1.1)
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=9, color=INK_SECONDARY)
ax.set_title("Does table size hurt accuracy on unseen documents? row_type accuracy by size quartile",
             fontsize=11.7, color=INK, loc="left", fontweight="bold", pad=10)
ax.set_ylabel("Mean row_type accuracy", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_complexity_vs_accuracy.png"), dpi=160)
plt.close(fig)

# ---- 5 (worst). worst documents by mean row_type_accuracy (min 3 tables) ----
worst_docs = [d for d in stats["doc_scores_worst20"] if d["n_tables"] >= 3][:15]
worst_docs = list(reversed(worst_docs))
labels = [f"{d['docname'][:38]} (n={d['n_tables']})" for d in worst_docs]
values = [d["mean_row_type_accuracy"] for d in worst_docs]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = range(len(labels))
ax.barh(list(y), values, color=RED, height=0.6, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9)
ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_xlim(0, 1.1)
for i, v in enumerate(values):
    ax.text(v + 0.015, i, f"{v*100:.1f}%", va="center", fontsize=8.5, color=INK_SECONDARY)
ax.set_title("Lowest-scoring unseen documents by mean row_type accuracy (≥3 tables compared)",
             fontsize=12, color=INK, loc="left", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_worst_documents.png"), dpi=160)
plt.close(fig)

# ---- 5. worst individual tables by row_type_accuracy ----
worst_tables = list(reversed(stats["worst_tables"][:15]))
labels = [f"{t['docname'][:30]} / {t['table_id']}" for t in worst_tables]
values = [t["row_type_accuracy"] for t in worst_tables]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
y = range(len(labels))
ax.barh(list(y), values, color=ORANGE, height=0.6, zorder=3)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8.7)
ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_xlim(0, 1.1)
for i, v in enumerate(values):
    ax.text(v + 0.015, i, f"{v*100:.1f}%", va="center", fontsize=8.3, color=INK_SECONDARY)
ax.set_title("Lowest-scoring individual tables, unseen documents only", fontsize=12, color=INK,
             loc="left", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_worst_tables.png"), dpi=160)
plt.close(fig)

# ---- 6. inference latency ----
lat = stats["per_table_latency_sec_describe"]
fig, ax = plt.subplots(figsize=(8, 3.6))
labels = ["min", "median", "mean", "max"]
vals = [lat["min"], lat["median"], lat["mean"], lat["max"]]
ax.bar(labels, vals, color=BLUE, width=0.55, zorder=3)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax, hide_y_spine=False)
ax.spines["left"].set_color(BASELINE)
for i, v in enumerate(vals):
    ax.text(i, v + 0.6, f"{v:.1f}s", ha="center", fontsize=9.5, color=INK_SECONDARY)
ax.set_title("Local quantized-model inference latency per table — unseen documents (seconds)", fontsize=11.5, color=INK,
             loc="left", fontweight="bold", pad=10)
ax.set_ylabel("Seconds / table", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_latency.png"), dpi=160)
plt.close(fig)

# ---- 7. this pure-unseen run vs. the earlier mixed-corpus benchmark ----
bc = stats["benchmark_comparison"]
metrics = [
    ("table_type_lexical_match_rate", "table_type\nlexical match"),
    ("row_type_accuracy", "row_type\naccuracy"),
    ("column_type_accuracy", "column_type\naccuracy"),
    ("cell_concept_id_lexical_accuracy", "concept_id\nlexical accuracy"),
]
unseen_vals = [bc["unseen"]["weighted_overall"][k] for k, _ in metrics]
mixed_vals = [bc["mixed_report_2026_07_24"]["weighted_overall"][k] for k, _ in metrics]
labels = [lbl for _, lbl in metrics]

x = range(len(labels))
width = 0.32
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar([i - width / 2 for i in x], mixed_vals, width=width, color=BLUE, zorder=3,
       label=f"Earlier mixed-corpus report (in-sample + held-out, n={bc['mixed_report_2026_07_24']['n_jobs']} jobs / {bc['mixed_report_2026_07_24']['n_tables']} tables)")
ax.bar([i + width / 2 for i in x], unseen_vals, width=width, color=ORANGE, zorder=3,
       label=f"This report — genuinely unseen only (n={bc['unseen']['n_jobs']} jobs / {bc['unseen']['n_tables']} tables)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9.5)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax, hide_y_spine=False)
ax.spines["left"].set_color(BASELINE)
ax.set_ylim(0, 1.32)
for i, (v1, v2) in enumerate(zip(mixed_vals, unseen_vals)):
    ax.text(i - width / 2, v1 + 0.02, f"{v1*100:.1f}%", ha="center", fontsize=8.3, color=INK_SECONDARY)
    ax.text(i + width / 2, v2 + 0.02, f"{v2*100:.1f}%", ha="center", fontsize=8.3, color=INK_SECONDARY)
ax.legend(loc="upper center", frameon=False, fontsize=8.1, ncol=1)
ax.set_title("Does restricting to purely unseen documents change the picture?", fontsize=12.5, color=INK,
             loc="left", fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_unseen_vs_mixed_benchmark.png"), dpi=160)
plt.close(fig)

print("Charts written to", OUT_DIR)
