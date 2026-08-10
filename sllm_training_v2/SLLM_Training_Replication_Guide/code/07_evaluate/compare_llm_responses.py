"""
Compare GPT-4.1 responses (classify_<table_id>_response.json) against the custom LLM's
responses (classify_<table_id>_response_customllm.json, written by
generate_customllm_responses.py) sitting in the same output/<job_id> folder.

Produces one compare_<table_id>.json per table (field-by-field diff) plus a
compare_summary.json with aggregate accuracy across table_type / row_type / column_type /
cell concept_id classification, so the two models can be benchmarked against each other.

For table_type, concept_id, and concept_meaning - fields where two different models
(e.g. GPT-4.1 vs. a LoRA-tuned local model) can validly disagree on exact wording while
describing the same thing - every comparison reports two independent signals side by
side, neither replacing the other: a dependency-free lexical token/character-overlap
match, and (when sentence-transformers is installed) a sentence-embedding cosine-
similarity match. The gap between them is itself informative: a field with low
lexical-match but high semantic-match tells you the models mostly agree and are just
phrasing things differently, not disagreeing on substance.

Exact string match is still tracked per-cell internally (available in each
compare_<table_id>.json report and used by visualize_table.py's agreement tiers), but
is deliberately not surfaced as an aggregate metric (table_type_exact_match_rate /
cell_concept_id_exact_accuracy) - it isn't a meaningful summary statistic for fields
where two models can validly differ on wording alone.

Usage:
    python compare_llm_responses.py --job-dir "D:\\Dev\\git_repo\\TFS\\IntelligentDocumentParser\\output\\<job_id>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

RESERVED_TOP_KEYS = {"table_type", "rows", "columns", "cells", "table_index"}
NUMERIC_KEY_RE = re.compile(r"^\d+$")
RESPONSE_FILE_RE = re.compile(r"^classify_(.+)_response\.json$")
_WORD_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# table_type / concept_id are short labels ("OtherOperatingExpenses" vs
# "OtherOperatingExpensesSubtotal"). Calibrated empirically on real concept_id pairs:
# same-concept pairs scored 0.75-1.0 on both metrics, different-concept pairs scored
# 0.19-0.30 (lexical) / 0.19-0.30 (embedding) - both thresholds sit in the gap.
LABEL_LEXICAL_THRESHOLD = 0.5
LABEL_EMBEDDING_THRESHOLD = 0.6

# concept_meaning is a full sentence. Calibrated separately since embeddings and the
# lexical blend have different scales: same-concept paraphrases scored ~0.78-0.88
# (embedding) and mixed (lexical, since prose paraphrases often share few exact
# tokens); different-concept pairs scored ~0.53-0.64 (embedding, since all these
# descriptions share boilerplate like "during the reporting period").
MEANING_LEXICAL_THRESHOLD = 0.6
MEANING_EMBEDDING_THRESHOLD = 0.70

# reporting_period ("2024", "June 2024", "Q1 2024", "FY24", "2024-06", ...) isn't a
# free-form-wording field like concept_id/concept_meaning - it's a discrete calendar
# fact, so the right comparison is normalization + containment, not fuzzy similarity:
# parse both sides into a (year, granularity, unit) tuple and check whether one
# period's month-range fully contains the other's (a bare year is treated as
# compatible with any narrower period inside that year, e.g. "2024" ~ "June 2024" ~
# "Q2 2024" - the model that gave the year alone isn't "wrong", just less specific).
_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_QUARTER_COMPACT_RE = re.compile(r"\b([1-4])\s*q\s*['’]?\s*(\d{2,4})\b", re.IGNORECASE)
_QUARTER_COMPACT_RE2 = re.compile(r"\bq\s*([1-4])\s*['’]?\s*(\d{2,4})\b", re.IGNORECASE)
_FY_RE = re.compile(r"\bFY\s*['’]?\s*(\d{2,4})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER_WORD_RE = re.compile(r"\bq(?:uarter)?\s*([1-4])\b", re.IGNORECASE)
_HALF_RE = re.compile(r"\bh\s*([1-2])\b", re.IGNORECASE)
_HALF_WORD_RE = re.compile(r"\b(first|1st)\s*half\b|\b(second|2nd)\s*half\b", re.IGNORECASE)
_MONTH_NUM_RE1 = re.compile(r"\b(0?[1-9]|1[0-2])[/-](\d{4})\b")
_MONTH_NUM_RE2 = re.compile(r"\b(\d{4})[/-](0?[1-9]|1[0-2])\b")
_MONTHS_IN_QUARTER = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11, 12)}
_MONTHS_IN_HALF = {1: tuple(range(1, 7)), 2: tuple(range(7, 13))}


def _expand_year(digits: str) -> int:
    year = int(digits)
    if year >= 1000:
        return year
    return 2000 + year if year < 70 else 1900 + year


def normalize_period(text) -> tuple | None:
    """Parse a reporting-period label into a canonical (year, granularity, unit)
    tuple - "June 2024", "Jun 2024", "2024-06" all normalize to (2024, "month", 6)
    regardless of wording/format. Returns None if no year can be identified at all."""
    if not text:
        return None
    t = str(text).strip().lower()

    # Compact quarter+year forms first (e.g. "1Q24", "Q124") - the year digits are
    # concatenated right onto the quarter marker, so the generic word-boundary
    # quarter/year regexes below can't separate them.
    m = _QUARTER_COMPACT_RE.search(t)
    if m:
        return (_expand_year(m.group(2)), "quarter", int(m.group(1)))
    m = _QUARTER_COMPACT_RE2.search(t)
    if m:
        return (_expand_year(m.group(2)), "quarter", int(m.group(1)))

    year = None
    m = _FY_RE.search(t)
    if m:
        year = _expand_year(m.group(1))
    else:
        m = _YEAR_RE.search(t)
        if m:
            year = int(m.group(0))
    if year is None:
        return None

    m = _QUARTER_WORD_RE.search(t)
    if m:
        return (year, "quarter", int(m.group(1)))
    m = _HALF_RE.search(t)
    if m:
        return (year, "half", int(m.group(1)))
    if _HALF_WORD_RE.search(t):
        return (year, "half", 1 if re.search(r"first|1st", t) else 2)
    for name, num in _MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", t):
            return (year, "month", num)
    m = _MONTH_NUM_RE1.search(t) or _MONTH_NUM_RE2.search(t)
    if m:
        g = m.groups()
        month = int(g[0]) if len(g[0]) <= 2 else int(g[1])
        return (year, "month", month)
    return (year, "year", None)


def _period_month_range(norm: tuple) -> set[int]:
    year, gran, unit = norm
    if gran == "year":
        return set(range(1, 13))
    if gran == "half":
        return set(_MONTHS_IN_HALF[unit])
    if gran == "quarter":
        return set(_MONTHS_IN_QUARTER[unit])
    if gran == "month":
        return {unit}
    return set()


def periods_match(ref_norm: tuple | None, cand_norm: tuple | None) -> bool:
    """Same year AND one period's month-range fully contains (or equals) the
    other's - so a bare year matches any narrower period within it, a half matches
    any quarter/month within that half, etc. Different years, or two specific but
    non-overlapping periods (different months, different quarters), do not match."""
    if ref_norm is None or cand_norm is None:
        return False
    if ref_norm[0] != cand_norm[0]:
        return False
    ref_months, cand_months = _period_month_range(ref_norm), _period_month_range(cand_norm)
    return ref_months <= cand_months or cand_months <= ref_months


def compare_period(ref, cand) -> dict:
    """Like compare_scalar, but also reports period_match (the normalized/structural
    comparison) alongside the plain exact match - neither replaces the other."""
    exact = ref == cand
    ref_norm, cand_norm = normalize_period(ref), normalize_period(cand)
    if ref_norm is not None and cand_norm is not None:
        period_match = periods_match(ref_norm, cand_norm)
    else:
        # Unparseable on at least one side - fall back to exact string match rather
        # than silently guessing.
        period_match = exact
    return {
        "reference": ref,
        "candidate": cand,
        "match": exact,
        "period_match": period_match,
        "reference_normalized": ref_norm,
        "candidate_normalized": cand_norm,
    }


def _unwrap(payload: Any) -> Any:
    """classify_*_response.json can hold a dict, or occasionally a single-item list."""
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload


def _sort_key(idx: str):
    return (0, int(idx)) if idx.isdigit() else (1, idx)


def normalize_response(payload: Any) -> tuple[dict, list[str]]:
    """Return (normalized_dict, repair_notes).

    Defensively re-nests any numeric-string top-level keys that should live under
    "cells" but ended up as siblings instead - the malformed-JSON pattern seen from
    CPU-inference runs (model closes the "cells" object early; json_repair can't know
    where the orphaned keys belong, so it leaves them at the top level).
    """
    payload = _unwrap(payload)
    if not isinstance(payload, dict):
        return {"table_type": {}, "rows": {}, "columns": {}, "cells": {}}, ["not_a_dict"]

    normalized = {
        "table_type": payload.get("table_type") or {},
        "rows": payload.get("rows") or {},
        "columns": payload.get("columns") or {},
        "cells": dict(payload.get("cells") or {}),
    }

    repairs = []
    for key, value in payload.items():
        if key in RESERVED_TOP_KEYS:
            continue
        if NUMERIC_KEY_RE.match(key) and isinstance(value, dict):
            normalized["cells"].setdefault(key, value)
            repairs.append(key)

    return normalized, repairs


def tokenize(s: str) -> set[str]:
    """Split a PascalCase/snake_case/space-separated label into lowercase word tokens,
    e.g. "OtherOperatingExpensesSubtotal" -> {"other", "operating", "expenses", "subtotal"}."""
    if not s:
        return set()
    words = _WORD_SPLIT_RE.findall(s.replace("_", " "))
    return {w.lower() for w in words if w}


def lexical_similarity(ref: str, cand: str) -> float:
    """Dependency-free blend of character-level and token-overlap similarity. This is
    the original matching logic - kept exactly as-is and always computed, regardless
    of whether the embedding model below is also available."""
    ref, cand = ref or "", cand or ""
    if not ref and not cand:
        return 1.0
    char_sim = SequenceMatcher(None, ref, cand).ratio()
    ref_tokens, cand_tokens = tokenize(ref), tokenize(cand)
    union = ref_tokens | cand_tokens
    token_sim = (len(ref_tokens & cand_tokens) / len(union)) if union else char_sim
    return max(char_sim, token_sim)


_embedder = None
_embedder_load_failed = False


def _get_embedder():
    """Lazily load the sentence-transformer model once and cache it. Returns None
    (after printing a one-time warning) if sentence-transformers/torch aren't
    installed, so the script still runs - with the embedding_similarity/embedding_match
    fields simply absent - in environments without that optional dependency."""
    global _embedder, _embedder_load_failed
    if _embedder is not None or _embedder_load_failed:
        return _embedder
    try:
        # We only ever need the PyTorch backend. If TensorFlow also happens to be
        # installed in this environment (pulled in by something unrelated),
        # `transformers` auto-detects and validates every backend it finds - which
        # can hard-fail on a Keras 3 / tf-keras version mismatch even though TF is
        # never actually used here. Telling it up front to skip TF/Flax avoids that
        # failure mode entirely instead of requiring an unrelated `tf-keras` install.
        import os
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("USE_TORCH", "1")

        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        _embedder_load_failed = True
        print(
            f"[!] sentence-transformers unavailable ({e}); embedding_similarity/"
            f"embedding_match will be omitted (lexical + exact match still reported). "
            f"Run `pip install sentence-transformers` to enable it.",
            file=sys.stderr,
        )
    return _embedder


def embedding_cosine_similarities(pairs: list[tuple[str, str]]) -> list[float] | None:
    """Batch-encode a list of (reference, candidate) text pairs and return their
    cosine similarities, or None if the embedding model isn't available. One model
    call per batch (a whole table's worth of cells) rather than one per cell."""
    if not pairs:
        return []
    embedder = _get_embedder()
    if embedder is None:
        return None

    texts = [t for pair in pairs for t in pair]  # [ref0, cand0, ref1, cand1, ...]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    # embeddings are L2-normalized, so dot product == cosine similarity
    return [float(embeddings[2 * i] @ embeddings[2 * i + 1]) for i in range(len(pairs))]


def compare_scalar(ref, cand) -> dict:
    return {"reference": ref, "candidate": cand, "match": ref == cand}


def compare_list_as_set(ref, cand) -> dict:
    ref_set = set(ref or [])
    cand_set = set(cand or [])
    return {"reference": ref, "candidate": cand, "match": ref_set == cand_set}


def build_field_diff(
    ref, cand, lexical_threshold: float, embedding_threshold: float, embedding_sim: float | None
) -> dict:
    """Report exact match, the lexical blend, AND the embedding similarity for a field -
    each with its own calibrated threshold and match verdict. None replaces another:
    all three are always present (embedding fields only when the model is available),
    so the comparison stays fully auditable rather than collapsing to one number."""
    ref_s = "" if ref is None else str(ref)
    cand_s = "" if cand is None else str(cand)
    lex_sim = lexical_similarity(ref_s, cand_s)
    result = {
        "reference": ref,
        "candidate": cand,
        "exact_match": ref == cand,
        "lexical_similarity": round(lex_sim, 3),
        "lexical_match": lex_sim >= lexical_threshold,
    }
    if embedding_sim is not None:
        result["embedding_similarity"] = round(embedding_sim, 3)
        result["embedding_match"] = embedding_sim >= embedding_threshold
    return result


def compare_table_type(ref: dict, cand: dict) -> dict:
    ref_val = (ref or {}).get("value")
    cand_val = (cand or {}).get("value")
    sims = embedding_cosine_similarities([(str(ref_val or ""), str(cand_val or ""))])
    emb_sim = sims[0] if sims else None
    return {"value": build_field_diff(ref_val, cand_val, LABEL_LEXICAL_THRESHOLD, LABEL_EMBEDDING_THRESHOLD, emb_sim)}


def compare_rows(ref_rows: dict, cand_rows: dict) -> dict:
    all_idx = sorted(set(ref_rows) | set(cand_rows), key=_sort_key)
    per_row = {}
    matches = {"row_type": 0, "is_signed_negative": 0, "direction": 0, "note_ref_value": 0, "contributing_rows": 0}
    total = 0
    missing_ref, missing_cand = [], []

    for idx in all_idx:
        r, c = ref_rows.get(idx), cand_rows.get(idx)
        if r is None:
            missing_ref.append(idx)
            continue
        if c is None:
            missing_cand.append(idx)
            continue
        if not isinstance(r, dict) or not isinstance(c, dict):
            continue
        total += 1
        row_diff = {
            "row_type": compare_scalar(r.get("row_type"), c.get("row_type")),
            "is_signed_negative": compare_scalar(r.get("is_signed_negative"), c.get("is_signed_negative")),
            "direction": compare_scalar(r.get("direction"), c.get("direction")),
            "note_ref_value": compare_list_as_set(r.get("note_ref_value"), c.get("note_ref_value")),
            "contributing_rows": compare_list_as_set(r.get("contributing_rows"), c.get("contributing_rows")),
        }
        for field in matches:
            if row_diff[field]["match"]:
                matches[field] += 1
        per_row[idx] = row_diff

    accuracy = {field: (count / total if total else None) for field, count in matches.items()}
    return {
        "per_row": per_row,
        "accuracy": accuracy,
        "row_count_compared": total,
        "missing_in_reference": missing_ref,
        "missing_in_candidate": missing_cand,
    }


def compare_columns(ref_cols: dict, cand_cols: dict) -> dict:
    all_idx = sorted(set(ref_cols) | set(cand_cols), key=_sort_key)
    per_col = {}
    matches = {"column_type": 0, "direction": 0, "contributing_columns": 0}
    total = 0
    missing_ref, missing_cand = [], []

    for idx in all_idx:
        r, c = ref_cols.get(idx), cand_cols.get(idx)
        if r is None:
            missing_ref.append(idx)
            continue
        if c is None:
            missing_cand.append(idx)
            continue
        if not isinstance(r, dict) or not isinstance(c, dict):
            continue
        total += 1
        col_diff = {
            "column_type": compare_scalar(r.get("column_type"), c.get("column_type")),
            "direction": compare_scalar(r.get("direction"), c.get("direction")),
            "contributing_columns": compare_list_as_set(r.get("contributing_columns"), c.get("contributing_columns")),
        }
        for field in matches:
            if col_diff[field]["match"]:
                matches[field] += 1
        per_col[idx] = col_diff

    accuracy = {field: (count / total if total else None) for field, count in matches.items()}
    return {
        "per_column": per_col,
        "accuracy": accuracy,
        "column_count_compared": total,
        "missing_in_reference": missing_ref,
        "missing_in_candidate": missing_cand,
    }


def compare_cells(ref_cells: dict, cand_cells: dict) -> dict:
    ref_flat = {
        (r, c): v for r, cols in ref_cells.items() if isinstance(cols, dict) for c, v in cols.items()
    }
    cand_flat = {
        (r, c): v for r, cols in cand_cells.items() if isinstance(cols, dict) for c, v in cols.items()
    }
    all_keys = sorted(set(ref_flat) | set(cand_flat), key=lambda k: (_sort_key(k[0]), _sort_key(k[1])))

    per_cell = {}
    exact_fields = {"unit": 0, "scale": 0, "scale_multiplier": 0}
    reporting_period_exact_count = 0
    reporting_period_normalized_count = 0
    total = 0
    missing_ref, missing_cand = [], []
    concept_id_pairs = []  # (label, ref_id, cand_id)
    meaning_pairs = []  # (label, ref_meaning, cand_meaning)

    for row_idx, col_idx in all_keys:
        key = (row_idx, col_idx)
        r, c = ref_flat.get(key), cand_flat.get(key)
        label = f"{row_idx}.{col_idx}"
        if r is None:
            missing_ref.append(label)
            continue
        if c is None:
            missing_cand.append(label)
            continue
        if not isinstance(r, dict) or not isinstance(c, dict):
            continue
        total += 1
        cell_diff = {
            "unit": compare_scalar(r.get("unit"), c.get("unit")),
            "scale": compare_scalar(r.get("scale"), c.get("scale")),
            "scale_multiplier": compare_scalar(r.get("scale_multiplier"), c.get("scale_multiplier")),
            "reporting_period": compare_period(r.get("reporting_period"), c.get("reporting_period")),
        }
        for field in exact_fields:
            if cell_diff[field]["match"]:
                exact_fields[field] += 1
        if cell_diff["reporting_period"]["match"]:
            reporting_period_exact_count += 1
        if cell_diff["reporting_period"]["period_match"]:
            reporting_period_normalized_count += 1

        per_cell[label] = cell_diff
        concept_id_pairs.append((label, r.get("concept_id") or "", c.get("concept_id") or ""))
        meaning_pairs.append((label, r.get("concept_meaning") or "", c.get("concept_meaning") or ""))

    # One embedding-model call per table for concept_id, and one for concept_meaning -
    # batching across the table's cells is what makes the embedding model practical.
    concept_id_emb_sims = embedding_cosine_similarities([(rid, cid) for _, rid, cid in concept_id_pairs])
    meaning_emb_sims = embedding_cosine_similarities([(rm, cm) for _, rm, cm in meaning_pairs])

    concept_id_lexical_count = concept_id_embedding_count = 0
    concept_id_lexical_sum = concept_id_embedding_sum = 0.0
    for i, (label, ref_id, cand_id) in enumerate(concept_id_pairs):
        emb_sim = concept_id_emb_sims[i] if concept_id_emb_sims is not None else None
        diff = build_field_diff(ref_id, cand_id, LABEL_LEXICAL_THRESHOLD, LABEL_EMBEDDING_THRESHOLD, emb_sim)
        per_cell[label]["concept_id"] = diff
        if diff["lexical_match"]:
            concept_id_lexical_count += 1
        concept_id_lexical_sum += diff["lexical_similarity"]
        if "embedding_similarity" in diff:
            if diff["embedding_match"]:
                concept_id_embedding_count += 1
            concept_id_embedding_sum += diff["embedding_similarity"]

    meaning_lexical_count = meaning_embedding_count = 0
    meaning_lexical_sum = meaning_embedding_sum = 0.0
    for i, (label, ref_m, cand_m) in enumerate(meaning_pairs):
        emb_sim = meaning_emb_sims[i] if meaning_emb_sims is not None else None
        diff = build_field_diff(ref_m, cand_m, MEANING_LEXICAL_THRESHOLD, MEANING_EMBEDDING_THRESHOLD, emb_sim)
        # concept_meaning is prose, not a field worth an "exact_match" callout - drop it
        diff.pop("exact_match", None)
        per_cell[label]["concept_meaning"] = diff
        if diff["lexical_match"]:
            meaning_lexical_count += 1
        meaning_lexical_sum += diff["lexical_similarity"]
        if "embedding_similarity" in diff:
            if diff["embedding_match"]:
                meaning_embedding_count += 1
            meaning_embedding_sum += diff["embedding_similarity"]

    have_embeddings = concept_id_emb_sims is not None

    accuracy = {field: (count / total if total else None) for field, count in exact_fields.items()}
    # reporting_period keeps its old key (plain exact-match rate, unchanged meaning)
    # and gains a new normalized rate alongside it - same "add, don't replace" pattern
    # used everywhere else. The normalized rate is the more meaningful one for this
    # field (a bare year vs. a specific month within it isn't really a disagreement),
    # but the exact rate stays available for anyone auditing raw string equality.
    accuracy["reporting_period"] = _rate(reporting_period_exact_count, total)
    accuracy["reporting_period_normalized_accuracy"] = _rate(reporting_period_normalized_count, total)
    accuracy["concept_id_lexical_accuracy"] = _rate(concept_id_lexical_count, total)
    accuracy["concept_id_avg_lexical_similarity"] = round(concept_id_lexical_sum / total, 3) if total else None
    accuracy["concept_meaning_lexical_match_rate"] = _rate(meaning_lexical_count, total)
    accuracy["concept_meaning_avg_lexical_similarity"] = round(meaning_lexical_sum / total, 3) if total else None
    if have_embeddings:
        accuracy["concept_id_embedding_accuracy"] = _rate(concept_id_embedding_count, total)
        accuracy["concept_id_avg_embedding_similarity"] = (
            round(concept_id_embedding_sum / total, 3) if total else None
        )
        accuracy["concept_meaning_embedding_match_rate"] = _rate(meaning_embedding_count, total)
        accuracy["concept_meaning_avg_embedding_similarity"] = (
            round(meaning_embedding_sum / total, 3) if total else None
        )

    return {
        "per_cell": per_cell,
        "accuracy": accuracy,
        "cell_count_compared": total,
        "missing_in_reference": missing_ref,
        "missing_in_candidate": missing_cand,
    }


def compare_table(table_id: str, ref_raw: Any, cand_raw: Any) -> dict:
    ref, ref_repairs = normalize_response(ref_raw)
    cand, cand_repairs = normalize_response(cand_raw)

    return {
        "table_id": table_id,
        "reference_structural_repairs": ref_repairs,
        "candidate_structural_repairs": cand_repairs,
        "table_type": compare_table_type(ref["table_type"], cand["table_type"]),
        "rows": compare_rows(ref["rows"], cand["rows"]),
        "columns": compare_columns(ref["columns"], cand["columns"]),
        "cells": compare_cells(ref["cells"], cand["cells"]),
    }


def discover_pairs(job_dir: Path, candidate_suffix: str) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    pairs, missing_candidates = [], []
    for ref_path in sorted(job_dir.glob("classify_*_response.json")):
        m = RESPONSE_FILE_RE.match(ref_path.name)
        if not m:
            continue
        table_id = m.group(1)
        cand_path = job_dir / f"classify_{table_id}_response{candidate_suffix}.json"
        if cand_path.exists():
            pairs.append((table_id, ref_path, cand_path))
        else:
            missing_candidates.append(table_id)
    return pairs, missing_candidates


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 3) if total else None


def build_summary(table_reports: list[dict]) -> dict:
    def field_avg(getter):
        return _avg([getter(r) for r in table_reports])

    overall = {
        "tables_compared": len(table_reports),
        # table_type / concept_id / concept_meaning: lexical and embedding side by
        # side, on purpose - the gap between them IS the interesting signal. Exact
        # match is intentionally not reported here (see compare_llm_responses.py
        # module docstring) - it's still tracked per-cell internally (used by
        # visualize_table.py's agreement tiers) but isn't a useful aggregate on its
        # own for fields where two models can validly differ on exact wording.
        "table_type_lexical_match_rate": field_avg(
            lambda r: 1.0 if r["table_type"]["value"]["lexical_match"] else 0.0
        ),
        "table_type_embedding_match_rate": field_avg(
            lambda r: (1.0 if r["table_type"]["value"].get("embedding_match") else 0.0)
            if "embedding_match" in r["table_type"]["value"]
            else None
        ),
        "row_type_accuracy": field_avg(lambda r: r["rows"]["accuracy"]["row_type"]),
        "row_is_signed_negative_accuracy": field_avg(lambda r: r["rows"]["accuracy"]["is_signed_negative"]),
        "row_direction_accuracy": field_avg(lambda r: r["rows"]["accuracy"]["direction"]),
        "row_note_ref_value_accuracy": field_avg(lambda r: r["rows"]["accuracy"]["note_ref_value"]),
        "row_contributing_rows_accuracy": field_avg(lambda r: r["rows"]["accuracy"]["contributing_rows"]),
        "column_type_accuracy": field_avg(lambda r: r["columns"]["accuracy"]["column_type"]),
        "column_direction_accuracy": field_avg(lambda r: r["columns"]["accuracy"]["direction"]),
        "column_contributing_columns_accuracy": field_avg(
            lambda r: r["columns"]["accuracy"]["contributing_columns"]
        ),
        "cell_concept_id_lexical_accuracy": field_avg(
            lambda r: r["cells"]["accuracy"]["concept_id_lexical_accuracy"]
        ),
        "cell_concept_id_embedding_accuracy": field_avg(
            lambda r: r["cells"]["accuracy"].get("concept_id_embedding_accuracy")
        ),
        "cell_unit_accuracy": field_avg(lambda r: r["cells"]["accuracy"]["unit"]),
        "cell_scale_accuracy": field_avg(lambda r: r["cells"]["accuracy"]["scale"]),
        "cell_scale_multiplier_accuracy": field_avg(lambda r: r["cells"]["accuracy"]["scale_multiplier"]),
        "cell_reporting_period_accuracy": field_avg(lambda r: r["cells"]["accuracy"]["reporting_period"]),
        "cell_reporting_period_normalized_accuracy": field_avg(
            lambda r: r["cells"]["accuracy"]["reporting_period_normalized_accuracy"]
        ),
        "cell_concept_meaning_lexical_match_rate": field_avg(
            lambda r: r["cells"]["accuracy"]["concept_meaning_lexical_match_rate"]
        ),
        "cell_concept_meaning_avg_lexical_similarity": field_avg(
            lambda r: r["cells"]["accuracy"]["concept_meaning_avg_lexical_similarity"]
        ),
        "cell_concept_meaning_embedding_match_rate": field_avg(
            lambda r: r["cells"]["accuracy"].get("concept_meaning_embedding_match_rate")
        ),
        "cell_concept_meaning_avg_embedding_similarity": field_avg(
            lambda r: r["cells"]["accuracy"].get("concept_meaning_avg_embedding_similarity")
        ),
    }
    per_table = [
        {
            "table_id": r["table_id"],
            "table_type_lexical_match": r["table_type"]["value"]["lexical_match"],
            "table_type_embedding_match": r["table_type"]["value"].get("embedding_match"),
            "row_type_accuracy": r["rows"]["accuracy"]["row_type"],
            "column_type_accuracy": r["columns"]["accuracy"]["column_type"],
            "concept_id_lexical_accuracy": r["cells"]["accuracy"]["concept_id_lexical_accuracy"],
            "concept_id_embedding_accuracy": r["cells"]["accuracy"].get("concept_id_embedding_accuracy"),
            "concept_meaning_avg_lexical_similarity": r["cells"]["accuracy"]["concept_meaning_avg_lexical_similarity"],
            "concept_meaning_avg_embedding_similarity": r["cells"]["accuracy"].get(
                "concept_meaning_avg_embedding_similarity"
            ),
            "reference_structural_repairs": r["reference_structural_repairs"],
            "candidate_structural_repairs": r["candidate_structural_repairs"],
        }
        for r in table_reports
    ]
    return {"overall": overall, "per_table": per_table}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GPT-4.1 classify_<id>_response.json files against the custom "
        "LLM's classify_<id>_response_customllm.json files (from generate_customllm_responses.py) "
        "in the same job folder, and report per-field classification accuracy."
    )
    parser.add_argument("--job-dir", required=True, help="Path to output/<job_id> folder holding both response sets")
    parser.add_argument("--candidate-suffix", default="_customllm", help="Suffix used for the custom LLM's response files")
    parser.add_argument("--out-dir", help="Where to write comparison reports (default: <job-dir>/compare)")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    out_dir = Path(args.out_dir) if args.out_dir else job_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing_candidates = discover_pairs(job_dir, args.candidate_suffix)
    if missing_candidates:
        print(
            f"[!] No candidate response for {len(missing_candidates)} table(s), skipping: {', '.join(missing_candidates)}",
            file=sys.stderr,
        )
    if not pairs:
        print(f"No comparable (reference, candidate) pairs found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    table_reports = []
    for table_id, ref_path, cand_path in pairs:
        try:
            ref_raw = load_json(ref_path)
            cand_raw = load_json(cand_path)
        except Exception as e:
            print(f"[!] Skipping {table_id}: could not parse JSON ({e})", file=sys.stderr)
            continue

        cand_unwrapped = _unwrap(cand_raw)
        if isinstance(cand_unwrapped, dict) and "error" in cand_unwrapped:
            print(f"[!] {table_id}: candidate response is an error, skipping scoring: {cand_unwrapped['error']}")
            continue

        report = compare_table(table_id, ref_raw, cand_raw)
        table_reports.append(report)

        (out_dir / f"compare_{table_id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        tt = report["table_type"]["value"]
        tt_status = "OK" if tt.get("embedding_match", tt["lexical_match"]) else "DIFF"
        print(
            f"  {table_id}: table_type={tt_status} | "
            f"row_type={report['rows']['accuracy']['row_type']} | "
            f"column_type={report['columns']['accuracy']['column_type']} | "
            f"concept_id(lexical/embedding)="
            f"{report['cells']['accuracy']['concept_id_lexical_accuracy']}/"
            f"{report['cells']['accuracy'].get('concept_id_embedding_accuracy')} | "
            f"meaning_sim(lexical/embedding)="
            f"{report['cells']['accuracy']['concept_meaning_avg_lexical_similarity']}/"
            f"{report['cells']['accuracy'].get('concept_meaning_avg_embedding_similarity')}"
        )

    if not table_reports:
        print("No tables were scored (all candidates were errors or unparseable).", file=sys.stderr)
        sys.exit(1)

    summary = build_summary(table_reports)
    summary_path = out_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nPer-table reports: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
