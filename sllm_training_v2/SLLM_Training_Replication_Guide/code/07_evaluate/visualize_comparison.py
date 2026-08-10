"""
Render a stakeholder-facing PNG chart from compare_summary.json (produced by
compare_llm_responses.py): the custom LLM vs. GPT-4.1 agreement rate for a single
source document.

The chart has four sections, ordered for a non-technical reader first:
  1. Headline KPI tiles (tables compared, overall agreement, strong-agreement count,
     needs-review count)
  2. A table-level agreement distribution bar (how many of the source document's
     tables fall into Strong / Partial / Needs-review agreement bands)
  3. Field-level detail for fields compared semantically (table_type, concept_id,
     concept_meaning) - lexical / embedding match rates side by side (exact-match is
     deliberately not shown here; see compare_llm_responses.py's module docstring)
  4. Field-level detail for fields compared exactly (row_type, column_type, ...)

The document name is pulled from job.json (written by
export_table_classification_inputs_filterfin.py) if it sits alongside the job
directory - job_dir/job.json, where job_dir is compare_summary.json's grandparent
(job_dir/compare/compare_summary.json). Falls back to no document name if not found.

Usage:
    python visualize_comparison.py --summary "<job_dir>/compare/compare_summary.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from _chart_style import (
    COLOR_EXACT, COLOR_LEXICAL, COLOR_EMBEDDING,
    STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE, SURFACE,
    EXCELLENT_THRESHOLD, PARTIAL_THRESHOLD,
)


def _pct(x):
    return None if x is None else round(x * 100, 1)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_job_metadata(summary_path: Path, job_json_override: str | None) -> dict:
    """Best-effort lookup of job.json's docx path, so the chart can name the source
    document. Returns {} (not None) on any miss so callers never need a None-check."""
    if job_json_override:
        job_json_path = Path(job_json_override)
    else:
        job_json_path = summary_path.parent.parent / "job.json"

    if not job_json_path.exists():
        return {}
    try:
        data = json.loads(job_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] Found {job_json_path} but couldn't parse it ({e}); omitting document name.", file=sys.stderr)
        return {}

    docx_path = data.get("docx")
    return {
        "docx_name": Path(docx_path).name if docx_path else None,
        "table_count": data.get("table_count"),
        "financial_table_count": data.get("financial_table_count"),
    }


def build_semantic_rows(overall: dict):
    """[(label, lexical_pct, embedding_pct), ...] - either slot may be None (e.g. no
    sentence-transformers installed) and is simply omitted from that row's bar group.
    Exact match is intentionally not included here - see compare_llm_responses.py's
    module docstring for why it isn't a useful aggregate for these fields."""
    return [
        (
            "Table Type",
            _pct(overall.get("table_type_lexical_match_rate")),
            _pct(overall.get("table_type_embedding_match_rate")),
        ),
        (
            "Concept ID",
            _pct(overall.get("cell_concept_id_lexical_accuracy")),
            _pct(overall.get("cell_concept_id_embedding_accuracy")),
        ),
        (
            "Concept Meaning",
            _pct(overall.get("cell_concept_meaning_lexical_match_rate")),
            _pct(overall.get("cell_concept_meaning_embedding_match_rate")),
        ),
    ]


def build_objective_rows(overall: dict):
    """[(label, accuracy_pct), ...] for fields compared by exact value only."""
    rows = [
        ("Row Type", _pct(overall.get("row_type_accuracy"))),
        ("Row Sign (+/-)", _pct(overall.get("row_is_signed_negative_accuracy"))),
        ("Row Direction", _pct(overall.get("row_direction_accuracy"))),
        ("Row Note Reference", _pct(overall.get("row_note_ref_value_accuracy"))),
        ("Row Contributing Rows", _pct(overall.get("row_contributing_rows_accuracy"))),
        ("Column Type", _pct(overall.get("column_type_accuracy"))),
        ("Column Direction", _pct(overall.get("column_direction_accuracy"))),
        ("Column Contributing Columns", _pct(overall.get("column_contributing_columns_accuracy"))),
        ("Unit", _pct(overall.get("cell_unit_accuracy"))),
        ("Scale", _pct(overall.get("cell_scale_accuracy"))),
        ("Scale Multiplier", _pct(overall.get("cell_scale_multiplier_accuracy"))),
        ("Reporting Period (exact)", _pct(overall.get("cell_reporting_period_accuracy"))),
        ("Reporting Period (normalized)", _pct(overall.get("cell_reporting_period_normalized_accuracy"))),
    ]
    return [r for r in rows if r[1] is not None]


def compute_table_composite(row: dict) -> float | None:
    """Blend a table's per-field signals (already in compare_summary.json's
    per_table list) into one 0-1 agreement score for that table, preferring the
    embedding-based metric where available and falling back to lexical - the same
    "don't replace, prefer the better signal" approach used throughout this script."""
    parts = []

    table_type_match = row.get("table_type_embedding_match")
    if table_type_match is None:
        table_type_match = row.get("table_type_lexical_match")
    if table_type_match is not None:
        parts.append(1.0 if table_type_match else 0.0)

    concept_id = row.get("concept_id_embedding_accuracy")
    if concept_id is None:
        concept_id = row.get("concept_id_lexical_accuracy")
    if concept_id is not None:
        parts.append(concept_id)

    meaning = row.get("concept_meaning_avg_embedding_similarity")
    if meaning is None:
        meaning = row.get("concept_meaning_avg_lexical_similarity")
    if meaning is not None:
        parts.append(meaning)

    if row.get("row_type_accuracy") is not None:
        parts.append(row["row_type_accuracy"])
    if row.get("column_type_accuracy") is not None:
        parts.append(row["column_type_accuracy"])

    return sum(parts) / len(parts) if parts else None


def bucket_tables(per_table: list[dict]) -> dict[str, list[str]]:
    """{"excellent": [table_id, ...], "partial": [...], "needs_review": [...]}"""
    bands = {"excellent": [], "partial": [], "needs_review": []}
    for row in per_table:
        score = compute_table_composite(row)
        if score is None:
            continue
        if score >= EXCELLENT_THRESHOLD:
            bands["excellent"].append(row["table_id"])
        elif score >= PARTIAL_THRESHOLD:
            bands["partial"].append(row["table_id"])
        else:
            bands["needs_review"].append(row["table_id"])
    return bands


def _style_axes(ax, n_rows):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(0, 112)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=10, colors=INK_PRIMARY)
    ax.tick_params(axis="x", length=0, labelsize=9, colors=INK_MUTED)
    ax.set_ylim(-0.5, n_rows - 0.5)


def _bar_label(ax, bar, value):
    ax.text(
        bar.get_width() + 1.8,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center", ha="left", fontsize=8.5, color=INK_SECONDARY,
    )


def _render_kpi_row(ax, tiles):
    """tiles: [(value_str, label, color), ...] - plain stat tiles (no plot chrome),
    separated by thin hairline dividers."""
    ax.set_xlim(0, len(tiles))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (value_str, label, color) in enumerate(tiles):
        cx = i + 0.5
        ax.text(cx, 0.62, value_str, ha="center", va="center", fontsize=23, fontweight="bold", color=color)
        ax.text(cx, 0.16, label, ha="center", va="center", fontsize=9.5, color=INK_SECONDARY)
        if i > 0:
            ax.axvline(i, ymin=0.08, ymax=0.92, color=GRIDLINE, linewidth=1)


def _render_distribution_panel(ax, bands, title):
    total = sum(len(v) for v in bands.values())
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.6, 0.6)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK_SECONDARY, loc="left", pad=4, x=0.0)

    if total == 0:
        ax.text(0, 0, "No scored tables", fontsize=10, color=INK_MUTED, va="center")
        return

    segments = [
        ("Strong agreement", bands["excellent"], STATUS_GOOD, "o"),
        ("Partial agreement", bands["partial"], STATUS_WARNING, "^"),
        ("Needs review", bands["needs_review"], STATUS_CRITICAL, "s"),
    ]

    x = 0.0
    for _, ids, color, _ in segments:
        count = len(ids)
        if count == 0:
            continue
        width = count / total * 100
        ax.barh(0, width, left=x, height=0.55, color=color, zorder=3)
        if width >= 6:
            ax.text(
                x + width / 2, 0, str(count), ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold", zorder=4,
            )
        x += width

    # loc="lower center" with NO bbox_to_anchor stays confined to this axes' own
    # bounding box (the ylim below the bar was extended specifically to make room for
    # it) - unlike bbox_to_anchor, which places the legend in axes-fraction space and
    # can escape into the next subplot's territory.
    handles = [Line2D([0], [0], marker=m, color=c, linestyle="none", markersize=8) for _, _, c, m in segments]
    legend_labels = [f"{label} — {len(ids)} of {total} table(s)" for label, ids, _, _ in segments]
    ax.legend(
        handles, legend_labels,
        loc="lower center",
        ncol=3, frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )


def _render_grouped_panel(ax, rows, title):
    n = len(rows)
    bar_h = 0.24
    bar_pitch = bar_h + 0.05
    y_positions = list(range(n))[::-1]  # top row reads first

    for y, (label, lexical, embedding) in zip(y_positions, rows):
        series = [
            (lexical, COLOR_LEXICAL),
            (embedding, COLOR_EMBEDDING),
        ]
        present = [s for s in series if s[0] is not None]
        top_offset = (len(present) - 1) / 2 * bar_pitch
        for j, (value, color) in enumerate(present):
            y_pos = y + top_offset - j * bar_pitch
            bar = ax.barh(y_pos, value, height=bar_h, color=color, zorder=3)[0]
            _bar_label(ax, bar, value)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title(title, fontsize=11, color=INK_SECONDARY, loc="left", pad=28)
    _style_axes(ax, n)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR_LEXICAL, COLOR_EMBEDDING)]
    ax.legend(
        handles,
        ["Lexical similarity", "Embedding similarity"],
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=1.1,
        handleheight=1.1,
    )


def _render_single_panel(ax, rows, title):
    n = len(rows)
    y_positions = list(range(n))[::-1]
    for y, (label, value) in zip(y_positions, rows):
        bar = ax.barh(y, value, height=0.5, color=COLOR_EXACT, zorder=3)[0]
        _bar_label(ax, bar, value)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title(title, fontsize=11, color=INK_SECONDARY, loc="left", pad=10)
    _style_axes(ax, n)


def _status_color(pct: float | None) -> str:
    if pct is None:
        return INK_PRIMARY
    if pct >= EXCELLENT_THRESHOLD * 100:
        return STATUS_GOOD
    if pct >= PARTIAL_THRESHOLD * 100:
        return STATUS_WARNING
    return STATUS_CRITICAL


def render(summary: dict, out_path: Path, job_meta: dict, subtitle: str | None = None):
    overall = summary["overall"]
    per_table = summary.get("per_table", [])
    semantic_rows = build_semantic_rows(overall)
    objective_rows = build_objective_rows(overall)
    bands = bucket_tables(per_table)
    total_scored = sum(len(v) for v in bands.values())

    composite_scores = [compute_table_composite(r) for r in per_table]
    composite_scores = [s for s in composite_scores if s is not None]
    overall_agreement_pct = _pct(sum(composite_scores) / len(composite_scores)) if composite_scores else None
    needs_attention = len(bands["partial"]) + len(bands["needs_review"])

    kpi_tiles = [
        (str(overall.get("tables_compared", "—")), "Tables Compared", INK_PRIMARY),
        (
            f"{overall_agreement_pct:.0f}%" if overall_agreement_pct is not None else "—",
            "Overall Agreement",
            _status_color(overall_agreement_pct),
        ),
        (str(len(bands["excellent"])), "Strong Agreement", STATUS_GOOD),
        (str(needs_attention), "Needs Review", STATUS_CRITICAL if needs_attention else STATUS_GOOD),
    ]

    n_semantic, n_objective = len(semantic_rows), len(objective_rows)
    fig = plt.figure(figsize=(9.5, 5.6 + 0.6 * (n_semantic + n_objective)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.9, 1.1, n_semantic, n_objective],
        hspace=0.55,
        top=0.90, bottom=0.05, left=0.16, right=0.97,
    )
    ax_kpi = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])
    ax_semantic = fig.add_subplot(gs[2])
    ax_objective = fig.add_subplot(gs[3])

    fig.suptitle(
        "Custom LLM vs. GPT-4.1 — Classification Agreement",
        fontsize=15, fontweight="bold", color=INK_PRIMARY, x=0.03, ha="left", y=0.985,
    )
    doc_name = job_meta.get("docx_name")
    default_sub = f"{overall.get('tables_compared', '?')} table(s) compared"
    if doc_name:
        default_sub += f"  |  Document: {doc_name}"
    sub = subtitle or default_sub
    fig.text(0.03, 0.945, sub, fontsize=10, color=INK_SECONDARY, ha="left")

    _render_kpi_row(ax_kpi, kpi_tiles)
    _render_distribution_panel(ax_dist, bands, f"Table-level agreement ({total_scored} table(s) scored)")
    _render_grouped_panel(ax_semantic, semantic_rows, "Compared semantically (lexical / embedding)")
    _render_single_panel(ax_objective, objective_rows, "Compared exactly (objective values)")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PNG summary chart from compare_summary.json")
    parser.add_argument("--summary", required=True, help="Path to compare_summary.json")
    parser.add_argument("--out", help="Output PNG path (default: same name as --summary, .png extension)")
    parser.add_argument("--subtitle", help="Optional subtitle override (default: table count + document name)")
    parser.add_argument(
        "--job-json",
        help="Path to job.json for the document name (default: auto-detected as <job_dir>/job.json, "
        "where job_dir is --summary's grandparent directory)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = load_summary(summary_path)
    job_meta = load_job_metadata(summary_path, args.job_json)
    out_path = Path(args.out) if args.out else summary_path.with_suffix(".png")

    render(summary, out_path, job_meta, subtitle=args.subtitle)
    print(f"Chart written to {out_path}")


if __name__ == "__main__":
    main()
