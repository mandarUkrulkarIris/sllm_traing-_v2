"""
Render ONE table's compare_<table_id>.json (written by compare_llm_responses.py) as a
financial-statement-style review, not a raw row/column-index grid:
  - Rows are actual line items, labeled by concept_id (e.g. "RoomRevenue"), not "Row 14"
  - Columns are actual reporting periods (e.g. "2025", "2024"), not "Col 2" - derived
    from each column's own cell data, not the raw structural index
  - Each line item x period shows a compact agreement dot, plus an overall per-row
    verdict - not a cramped block of unit/scale/concept text
  - Structural rows with no cell data (row_header, blank separators) collapse to thin
    divider bands instead of full-height empty rows

Visual style: no per-cell grid borders (a stroke around every cell is what makes a
table read as a spreadsheet dump) - only a single outer frame around the whole table
and thin horizontal hairlines between rows. Status is a colored dot beside neutral-ink
text, never colored text itself. Only rows that need attention get a background tint;
full-agreement rows sit on the plain page background so the eye goes straight to the
exceptions.

Every concrete disagreement - row/column classification, direction, note reference,
contributing rows/columns, concept_id, unit / scale / scale_multiplier /
reporting_period, or a concept_meaning that diverges even after accounting for
paraphrasing - is spelled out in plain language in a "Discrepancies" list underneath,
so a financial reviewer gets both an at-a-glance statement view and the exact wording
of anything worth a second look.

Usage:
    python visualize_table.py --report "<job_dir>/compare/compare_tbl-41.json"
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

from _chart_style import (
    STATUS_GOOD, STATUS_WARNING, STATUS_CRITICAL,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE, SURFACE,
)

TINT_GOOD = "#e3f5e3"
TINT_WARNING = "#fdf1d9"
TINT_CRITICAL = "#fbe3e2"

TIER_GOOD, TIER_WARNING, TIER_CRITICAL = 0, 1, 2
TIER_STYLE = {
    TIER_GOOD: ("OK", "OK", STATUS_GOOD, TINT_GOOD),
    TIER_WARNING: ("~", "Partial", STATUS_WARNING, TINT_WARNING),
    TIER_CRITICAL: ("X", "Review", STATUS_CRITICAL, TINT_CRITICAL),
}

_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[A-Z]+|\d+")
WRAP_CHARS = 20
LINE_UNIT = 0.19
MIN_ROW_H = 0.4
DIVIDER_H = 0.24
SENTENCE_WRAP_WIDTH = 100

DOT_R_SMALL = 0.032   # inline dot beside a text label
DOT_R_LARGE = 0.05    # standalone status dot (period cells)
PAD_X = 0.14          # left inset for left-aligned text
ROW_HAIRLINE_LW = 0.7
HEADER_RULE_LW = 1.3
FRAME_LW = 1.0


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sort_key(idx: str):
    return (0, int(idx)) if idx.isdigit() else (1, idx)


def sorted_ids(d: dict) -> list[str]:
    return sorted(d.keys(), key=_sort_key)


def _wrap_label(text, max_chars: int = WRAP_CHARS) -> str:
    """Break a long PascalCase/snake_case label onto multiple lines at word
    boundaries - these identifiers have no spaces for matplotlib's own wrapping."""
    text = str(text) if text else "—"
    if len(text) <= max_chars:
        return text
    words = _CAMEL_SPLIT_RE.findall(text.replace("_", " ")) or [text]
    lines, current = [], ""
    for w in words:
        candidate = current + w
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = w
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def cell_tier(cell: dict) -> int:
    """0=full agreement, 1=partial (paraphrasing/granularity only), 2=needs review."""
    concept_id = cell.get("concept_id", {})
    cid_exact = concept_id.get("exact_match", False)
    cid_embed_ok = concept_id.get("embedding_match", concept_id.get("lexical_match", cid_exact))
    meta_ok = all(cell.get(f, {}).get("match", True) for f in ("unit", "scale", "scale_multiplier"))
    period = cell.get("reporting_period", {})
    period_ok = period.get("period_match", period.get("match", True))
    meaning = cell.get("concept_meaning")
    meaning_ok = meaning.get("embedding_match", True) if meaning else True

    if not meta_ok or not period_ok or not cid_embed_ok:
        return TIER_CRITICAL
    if not cid_exact or not meaning_ok:
        return TIER_WARNING
    return TIER_GOOD


def build_statement(report: dict):
    """Returns (entries, period_col_ids, col_period_label).
    entries: ordered (by original row_id) list of either
      {"kind": "divider", "row_id", "row_type"} for rows with no cell data, or
      {"kind": "item", "row_id", "label", "row_type", "row_type_mismatch", "cells", "tier"}
    """
    row_ids = sorted_ids(report["rows"]["per_row"])
    col_ids = sorted_ids(report["columns"]["per_column"])
    per_cell = report["cells"]["per_cell"]
    per_row = report["rows"]["per_row"]

    # Period columns are identified by actually carrying cell data, and labeled by the
    # majority reporting_period value found there - a financial reviewer thinks in
    # periods, not raw column indices.
    col_period_label = {}
    for col_id in col_ids:
        periods = [
            per_cell[f"{r}.{col_id}"].get("reporting_period", {}).get("candidate")
            for r in row_ids if f"{r}.{col_id}" in per_cell
        ]
        periods = [p for p in periods if p]
        if periods:
            col_period_label[col_id] = max(set(periods), key=periods.count)
    period_col_ids = [c for c in col_ids if c in col_period_label]

    entries = []
    for row_id in row_ids:
        row_cells = {c: per_cell.get(f"{row_id}.{c}") for c in period_col_ids}
        rt = per_row[row_id].get("row_type", {})
        if not any(v is not None for v in row_cells.values()):
            entries.append({"kind": "divider", "row_id": row_id, "row_type": rt})
            continue

        first_cell = next(v for v in row_cells.values() if v is not None)
        label = first_cell.get("concept_id", {}).get("candidate") or "—"
        row_type_mismatch = not rt.get("match", True)

        tier = TIER_CRITICAL if row_type_mismatch else TIER_GOOD
        for cell in row_cells.values():
            tier = max(tier, TIER_WARNING if cell is None else cell_tier(cell))

        entries.append({
            "kind": "item", "row_id": row_id, "label": label, "row_type": rt,
            "row_type_mismatch": row_type_mismatch, "cells": row_cells, "tier": tier,
        })

    return entries, period_col_ids, col_period_label


def build_discrepancies(report: dict, entries: list, col_period_label: dict) -> list[str]:
    items = []

    tt = report.get("table_type", {}).get("value", {})
    if not tt.get("exact_match", True) and not tt.get("embedding_match", True):
        items.append(
            f"Table type: reference classifies this table as ‘{tt.get('reference')}’; candidate "
            f"classifies it as ‘{tt.get('candidate')}’ — a substantive disagreement, not just wording."
        )

    for col_id, col_entry in report["columns"]["per_column"].items():
        period = col_period_label.get(col_id, f"column {col_id}")
        ct = col_entry.get("column_type", {})
        if not ct.get("match", True):
            items.append(
                f"Column {col_id} ({period}): reference column type ‘{ct.get('reference')}’; "
                f"candidate ‘{ct.get('candidate')}’."
            )
        direction = col_entry.get("direction", {})
        if not direction.get("match", True):
            items.append(
                f"Column {col_id} ({period}): direction differs — reference ‘{direction.get('reference')}’, "
                f"candidate ‘{direction.get('candidate')}’."
            )
        contributing = col_entry.get("contributing_columns", {})
        if not contributing.get("match", True):
            items.append(
                f"Column {col_id} ({period}): contributing columns differ — reference "
                f"{contributing.get('reference')}, candidate {contributing.get('candidate')}."
            )

    for entry in entries:
        if entry["kind"] != "item":
            continue
        row_id, label, rt = entry["row_id"], entry["label"], entry["row_type"]
        row_full = report["rows"]["per_row"].get(row_id, {})
        if entry["row_type_mismatch"]:
            items.append(
                f"Row {row_id} ({label}): reference classifies this row as ‘{rt.get('reference')}’; "
                f"candidate classifies it as ‘{rt.get('candidate')}’."
            )
        direction = row_full.get("direction", {})
        if not direction.get("match", True):
            items.append(
                f"Row {row_id} ({label}): direction differs — reference ‘{direction.get('reference')}’, "
                f"candidate ‘{direction.get('candidate')}’."
            )
        note_ref = row_full.get("note_ref_value", {})
        if not note_ref.get("match", True):
            items.append(
                f"Row {row_id} ({label}): note reference differs — reference {note_ref.get('reference')}, "
                f"candidate {note_ref.get('candidate')}."
            )
        contributing_rows = row_full.get("contributing_rows", {})
        if not contributing_rows.get("match", True):
            items.append(
                f"Row {row_id} ({label}): contributing rows differ — reference "
                f"{contributing_rows.get('reference')}, candidate {contributing_rows.get('candidate')}."
            )

        for col_id, cell in entry["cells"].items():
            period = col_period_label.get(col_id, f"column {col_id}")
            if cell is None:
                items.append(f"Row {row_id} ({label}), {period}: present in one model's output but missing from the other's.")
                continue

            concept_id = cell.get("concept_id", {})
            if not concept_id.get("exact_match", False):
                embed_ok = concept_id.get("embedding_match", concept_id.get("lexical_match", False))
                verdict = "partial agreement" if embed_ok else "disagreement"
                sim = concept_id.get("embedding_similarity", concept_id.get("lexical_similarity"))
                sim_txt = f", similarity {sim:.2f}" if sim is not None else ""
                items.append(
                    f"Row {row_id}, {period}: concept differs — reference ‘{concept_id.get('reference')}’, "
                    f"candidate ‘{concept_id.get('candidate')}’ ({verdict}{sim_txt})."
                )

            for field in ("unit", "scale", "scale_multiplier"):
                f = cell.get(field, {})
                if not f.get("match", True):
                    items.append(
                        f"Row {row_id}, {period}: {field.replace('_', ' ')} differs — "
                        f"reference ‘{f.get('reference')}’, candidate ‘{f.get('candidate')}’."
                    )

            # reporting_period uses period_match (normalized/structural comparison),
            # not plain string equality - "2024" vs "June 2024" is NOT a discrepancy
            # (a bare year is compatible with any period within it), but "June 2024"
            # vs "July 2024" or "2023" vs "2024" still correctly flag.
            rp = cell.get("reporting_period", {})
            if not rp.get("period_match", rp.get("match", True)):
                items.append(
                    f"Row {row_id}, {period}: reporting period differs — "
                    f"reference ‘{rp.get('reference')}’, candidate ‘{rp.get('candidate')}’."
                )

            meaning = cell.get("concept_meaning")
            if meaning and concept_id.get("exact_match") and not meaning.get("embedding_match", True):
                sim = meaning.get("embedding_similarity")
                sim_txt = f" (similarity {sim:.2f})" if sim is not None else ""
                items.append(
                    f"Row {row_id}, {period}: concept tag matches (‘{concept_id.get('candidate')}’) but the "
                    f"two models' explanations diverge meaningfully{sim_txt} — worth a second look."
                )

    return items


def _dot(ax, x, y, color, r=DOT_R_SMALL):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=5))


def _text_block(ax, x, w, y, h, text, color, bold=False, fontsize=8.6, ha="left"):
    """Draw (possibly multi-line) text vertically CENTERED within row band (y, y+h),
    rather than anchored to the top - keeps short single-line cells from looking like
    they're floating at the top of a taller row."""
    lines = (text or "—").split("\n")
    n = len(lines)
    line_h = min(h / n, 0.22)
    block_h = line_h * n
    y_top = y + h / 2 + block_h / 2 - line_h / 2
    tx = {"left": x + PAD_X, "center": x + w / 2, "right": x + w - PAD_X}[ha]
    for i, line in enumerate(lines):
        ax.text(
            tx, y_top - i * line_h, line, ha=ha, va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.1,
        )


def _row_band(ax, x0, y, w, h, bg):
    """One plain background wash for the whole row - no per-cell borders, no per-cell
    boxes. Separation between columns comes from spacing/alignment, not ruled lines."""
    if bg and bg != SURFACE:
        ax.add_patch(Rectangle((x0, y), w, h, facecolor=bg, edgecolor="none", zorder=1))


def render(report: dict, out_path: Path):
    entries, period_col_ids, col_period_label = build_statement(report)
    discrepancies = build_discrepancies(report, entries, col_period_label)

    label_w = 2.9
    class_w = 1.55
    period_w = 0.95
    status_w = 1.3
    n_periods = len(period_col_ids)

    header_h = 0.46
    row_heights = []
    for entry in entries:
        if entry["kind"] == "divider":
            row_heights.append(DIVIDER_H)
        else:
            wrapped = _wrap_label(entry["label"])
            n_lines = wrapped.count("\n") + 1
            row_heights.append(max(MIN_ROW_H, LINE_UNIT * (n_lines + 1)))
    table_h = header_h + sum(row_heights)
    table_w = label_w + class_w + period_w * n_periods + status_w

    # Discrepancies section sizing
    disc_title_h = 0.4
    disc_line_h = 0.19
    wrapped_disc = [textwrap.wrap(f"{i + 1}. {d}", width=SENTENCE_WRAP_WIDTH) or [""] for i, d in enumerate(discrepancies)]
    disc_lines_total = sum(len(w) for w in wrapped_disc)
    disc_h = disc_title_h + (disc_lines_total * disc_line_h + len(wrapped_disc) * 0.08 + 0.3 if discrepancies else 0.35)

    fig_w = max(table_w + 0.7, 9.5)
    fig_h = table_h + disc_h + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, fig_w - 0.7)
    ax.set_ylim(0, table_h + disc_h)
    ax.axis("off")

    top_y = table_h + disc_h
    col_x = {
        "label": 0.0,
        "class": label_w,
        "status": label_w + class_w + period_w * n_periods,
    }
    period_x = {col_id: label_w + class_w + j * period_w for j, col_id in enumerate(period_col_ids)}

    # --- Outer frame (one card border, not a grid of cell borders) ---
    ax.add_patch(Rectangle((0, top_y - table_h), table_w, table_h, facecolor="none", edgecolor=BASELINE, linewidth=FRAME_LW, zorder=6))

    # --- Header row ---
    hy = top_y - header_h
    _text_block(ax, col_x["label"], label_w, hy, header_h, "LINE ITEM", INK_SECONDARY, bold=True, fontsize=8)
    _text_block(ax, col_x["class"], class_w, hy, header_h, "CLASSIFICATION", INK_SECONDARY, bold=True, fontsize=8, ha="center")
    for col_id in period_col_ids:
        _text_block(ax, period_x[col_id], period_w, hy, header_h, str(col_period_label[col_id]).upper(), INK_SECONDARY, bold=True, fontsize=8, ha="center")
    _text_block(ax, col_x["status"], status_w, hy, header_h, "STATUS", INK_SECONDARY, bold=True, fontsize=8, ha="center")
    ax.plot([0, table_w], [hy, hy], color=BASELINE, linewidth=HEADER_RULE_LW, zorder=6)

    # --- Rows ---
    y_cursor = hy
    for entry, rh in zip(entries, row_heights):
        y = y_cursor - rh
        if entry["kind"] == "divider":
            rt_text = entry["row_type"].get("candidate") or "—"
            mismatch = not entry["row_type"].get("match", True)
            if mismatch:
                _dot(ax, PAD_X - 0.02, y + rh / 2, STATUS_CRITICAL)
                label = f"{rt_text}  (ref: {entry['row_type'].get('reference')})"
                _text_block(ax, PAD_X + 0.09, table_w, y, rh, label, INK_SECONDARY, fontsize=7.4)
            else:
                _text_block(ax, PAD_X, table_w, y, rh, rt_text, INK_MUTED, fontsize=7.4)
        else:
            tier = entry["tier"]
            _, status_word, tier_color, tier_tint = TIER_STYLE[tier]
            # Only tint rows that actually need attention - a full-agreement row
            # stays on the plain page background, so the eye is drawn to the
            # exceptions instead of a wash of color across the whole statement.
            if tier != TIER_GOOD:
                _row_band(ax, 0, y, table_w, rh, tier_tint)

            label_x = col_x["label"]
            if tier != TIER_GOOD:
                _dot(ax, label_x + PAD_X - 0.02, y + rh / 2, tier_color)
                label_x_text = label_x + 0.1
            else:
                label_x_text = label_x
            _text_block(ax, label_x_text, label_w - (label_x_text - label_x), y, rh, _wrap_label(entry["label"]), INK_PRIMARY, bold=True)

            rt = entry["row_type"]
            if entry["row_type_mismatch"]:
                class_text = f"{rt.get('candidate')}\n(ref: {rt.get('reference')})"
            else:
                class_text = rt.get("candidate") or "—"
            _text_block(ax, col_x["class"], class_w, y, rh, class_text, INK_SECONDARY, fontsize=7.8, ha="center")

            for col_id in period_col_ids:
                x = period_x[col_id]
                cell = entry["cells"].get(col_id)
                if cell is None:
                    _text_block(ax, x, period_w, y, rh, "—", INK_MUTED, ha="center")
                else:
                    _, _, glyph_color, _ = TIER_STYLE[cell_tier(cell)]
                    _dot(ax, x + period_w / 2, y + rh / 2, glyph_color, r=DOT_R_LARGE)

            _dot(ax, col_x["status"] + status_w / 2 - 0.34, y + rh / 2, tier_color)
            _text_block(ax, col_x["status"] + 0.06, status_w - 0.12, y, rh, status_word, INK_SECONDARY, bold=True, fontsize=8.4, ha="center")
        ax.plot([0, table_w], [y, y], color=GRIDLINE, linewidth=ROW_HAIRLINE_LW, zorder=2)
        y_cursor = y

    # --- Discrepancies section ---
    dy = y_cursor - disc_title_h
    ax.text(
        0, dy + disc_title_h * 0.35, f"Discrepancies ({len(discrepancies)})",
        fontsize=12, fontweight="bold", color=INK_PRIMARY, ha="left", va="center",
    )
    if not discrepancies:
        _dot(ax, 0.03, dy - 0.14, STATUS_GOOD, r=DOT_R_SMALL)
        ax.text(
            0.12, dy - 0.14, "None — every line item, period, and classification agreed.",
            fontsize=9.5, color=INK_SECONDARY, ha="left", va="center", fontweight="bold",
        )
    else:
        y_cursor2 = dy - 0.08
        for wrapped in wrapped_disc:
            for line_idx, line in enumerate(wrapped):
                ax.text(
                    0.16 if line_idx else 0, y_cursor2, line,
                    fontsize=9, color=INK_SECONDARY, ha="left", va="top",
                )
                y_cursor2 -= disc_line_h
            y_cursor2 -= 0.09

    # --- Title / subtitle / legend ---
    table_id = report.get("table_id", "?")
    repairs = report.get("candidate_structural_repairs") or []
    fig.suptitle(
        f"Table {table_id} — Statement Review", fontsize=14.5, fontweight="bold",
        color=INK_PRIMARY, x=0.02, ha="left", y=0.997,
    )
    sub = f"GPT-4.1 (reference) vs. Custom LLM (candidate)  |  {len(period_col_ids)} period(s) compared"
    if repairs:
        sub += f"  |  [!] candidate JSON needed {len(repairs)} structural repair(s)"
    fig.text(0.02, 1 - 0.42 / fig_h, sub, fontsize=9, color=INK_SECONDARY, ha="left")

    legend_handles = [
        Line2D([0], [0], marker="o", color=TIER_STYLE[t][2], linestyle="none", markersize=8)
        for t in (0, 1, 2)
    ]
    fig.legend(
        legend_handles, ["Full agreement", "Partial agreement", "Needs review"],
        loc="upper right", bbox_to_anchor=(0.99, 0.997), ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )

    fig.subplots_adjust(top=1 - 0.5 / fig_h, bottom=0.02, left=0.04, right=0.97)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single compare_<table_id>.json as a financial-statement-style review")
    parser.add_argument("--report", required=True, help="Path to compare_<table_id>.json")
    parser.add_argument("--out", help="Output PNG path (default: same name as --report, .png extension)")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = load_report(report_path)
    out_path = Path(args.out) if args.out else report_path.with_suffix(".png")

    render(report, out_path)
    print(f"Chart written to {out_path}")


if __name__ == "__main__":
    main()
