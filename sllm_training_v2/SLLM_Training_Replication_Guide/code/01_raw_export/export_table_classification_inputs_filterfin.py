from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path so `app` is importable when running this
# script directly (e.g. `python scripts/...`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import settings
from app.core.logging import job_id_ctx
from app.parsers.docx.stage import extract_document
from app.schemas.output import build_output
from app.classification.classifier import TableClassifier, _get_headings_context
from app.classification.prompts import build_table_classification_prompt
from app.ai.factory import create_ai_client


from win32com import client

# Mapping for input cell short keys -> full names
_INPUT_CELL_KEY_MAP = {
    "ci": "column_index",
    "cs": "colspan",
    "rs": "rowspan",
}

# Mapping for AI response short keys -> full names
_OUTPUT_KEY_MAP = {
    # top-level
    "tt": "table_type",
    "r": "rows",
    "c": "columns",
    "cs": "cells",
    # table type
    "v": "value",
    "cf": "confidence",
    # row object
    "rt": "row_type",
    "cr": "contributing_rows",
    "nr": "note_ref_value",
    "d": "direction",
    "sn": "is_signed_negative",
    # column object
    "ct": "column_type",
    "cc": "contributing_columns",
    # cell object
    "p": "reporting_period",
    "cid": "concept_id",
    "cm": "concept_meaning",
    "u": "unit",
    "s": "scale",
    "sm": "scale_multiplier",
    "ci": "column_index",
}


def _expand_keys(obj: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace dict keys according to mapping."""
    if isinstance(obj, dict):
        new: dict[Any, Any] = {}
        for k, v in obj.items():
            new_key = mapping.get(k, k)
            new[new_key] = _expand_keys(v, mapping)
        return new
    elif isinstance(obj, list):
        return [_expand_keys(v, mapping) for v in obj]
    else:
        return obj


def _postprocess_input(input_data: Any) -> Any:
    """Return a copy of input_data with short cell keys expanded to full names.

    Only applies the input cell mapping; leaves other keys intact.
    """
    # We only need to translate ci/cs/rs inside cell dicts, but it's safe
    # to apply recursively with the smaller mapping.
    return _expand_keys(input_data, _INPUT_CELL_KEY_MAP)


def _postprocess_response_file(path: Path) -> None:
    """Read JSON file at path, expand short keys to full names, and overwrite file.

    If file does not contain JSON object/array, leaves it unchanged.
    """
    try:
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
    except Exception:
        return

    if not isinstance(parsed, (dict, list)):
        return

    expanded = _expand_keys(parsed, _OUTPUT_KEY_MAP)
    try:
        path.write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return

import re

FINANCIAL_TERMS = {

    # ---------- English ----------
    "revenue","sales","income","profit","loss","asset","assets",
    "liability","liabilities","equity","cash","cash flow",
    "expense","expenses","cost","costs","operating",
    "gross","net","ebit","ebitda","earnings","eps",
    "tax","interest","dividend","receivable","receivables",
    "payable","payables","inventory","goodwill",
    "depreciation","amortization","share capital",
    "retained earnings","book value","statement",
    "balance sheet","financial position",
    "income statement","cash flows","shareholder",
    "borrowings","loan","finance","financial",

    # ---------- Spanish ----------
    "ingresos","ventas","beneficio","ganancia","pérdida",
    "activo","activos","pasivo","pasivos","patrimonio",
    "flujo de efectivo","efectivo","gastos","coste","costos",
    "impuestos","dividendo","intereses","inventario",
    "balance","estado financiero",

    # ---------- French ----------
    "revenus","chiffre d'affaires","bénéfice","perte",
    "actif","actifs","passif","capitaux propres",
    "trésorerie","flux de trésorerie","charges",
    "coût","amortissement","impôt","dividende",
    "bilan","état financier",

    # ---------- German ----------
    "umsatz","erlöse","einnahmen","gewinn","verlust",
    "vermögen","aktiva","passiva","eigenkapital",
    "bilanz","abschreibung","steuer","zins",
    "liquidität","cashflow","kosten","aufwand",

    # ---------- Portuguese ----------
    "receita","receitas","lucro","prejuízo",
    "ativos","passivos","patrimônio",
    "fluxo de caixa","caixa","despesas",
    "custos","balanço","impostos",

    # ---------- Italian ----------
    "ricavi","utile","perdita","attività",
    "passività","patrimonio","bilancio",
    "flusso di cassa","costi","spese","imposte",

    # ---------- Dutch ----------
    "omzet","inkomsten","winst","verlies",
    "activa","passiva","eigen vermogen",
    "kasstroom","balans","kosten",

    # ---------- Russian ----------
    "выручка","доход","прибыль","убыток",
    "активы","пассивы","капитал",
    "денежный поток","баланс",
    "расходы","налог","дивиденд",

    # ---------- Polish ----------
    "przychody","zysk","strata",
    "aktywa","pasywa","kapitał",
    "przepływy pieniężne","bilans",

    # ---------- Turkish ----------
    "gelir","hasılat","kâr","zarar",
    "varlıklar","yükümlülükler",
    "özkaynak","nakit akışı",
    "bilanço","gider",

    # ---------- Swedish ----------
    "intäkter","omsättning","vinst","förlust",
    "tillgångar","skulder","eget kapital",
    "kassaflöde","balansräkning",

    # ---------- Norwegian ----------
    "inntekter","resultat","tap",
    "eiendeler","gjeld","egenkapital",
    "kontantstrøm","balanse",

    # ---------- Danish ----------
    "indtægter","overskud","tab",
    "aktiver","passiver",
    "egenkapital","pengestrøm",

    # ---------- Finnish ----------
    "liikevaihto","tuotot","voitto",
    "tappio","varat","velat",
    "oma pääoma","rahavirta",

    # ---------- Chinese ----------
    "收入","营业收入","利润","净利润",
    "亏损","资产","负债","权益",
    "现金流","资产负债表","财务报表",

    # ---------- Japanese ----------
    "売上","収益","利益","損失",
    "資産","負債","純資産",
    "キャッシュフロー","貸借対照表",

    # ---------- Korean ----------
    "매출","수익","이익","손실",
    "자산","부채","자본",
    "현금흐름","재무상태표",

    # ---------- Arabic ----------
    "الإيرادات","الأرباح","الخسائر",
    "الأصول","الخصوم","حقوق الملكية",
    "التدفقات النقدية","الميزانية",

    # ---------- Hindi ----------
    "राजस्व","आय","लाभ","हानि",
    "संपत्ति","देनदारियां",
    "नकदी प्रवाह","बैलेंस शीट",

    # ---------- Thai ----------
    "รายได้","กำไร","ขาดทุน",
    "สินทรัพย์","หนี้สิน",
    "ส่วนของผู้ถือหุ้น","กระแสเงินสด"
}

CURRENCY_TERMS = {

    # Symbols
    "$","€","£","¥","₹","₩","₽","₪","₫","₺",
    "฿","₴","₦","₱","₡","₲","₵","₭",

    # ISO Codes
    "usd","eur","gbp","jpy","cny","cad","aud",
    "chf","sek","nok","dkk","pln","rub","try",
    "inr","krw","thb","myr","idr","vnd","sgd",
    "hkd","zar","aed","qar","sar","kwd","bhd",
    "brl","mxn","ars","clp","cop","pen","ngn",

    # Scale Words (English)
    "thousand","thousands",
    "million","millions",
    "billion","billions",
    "trillion",
    "mn","bn","tn",
    "000","000s",

    # Spanish
    "mil","miles","millón","millones","mil millones",

    # French
    "mille","million","millions","milliard",

    # German
    "tausend","million","millionen","milliarde",

    # Portuguese
    "mil","milhão","milhões","bilhão",

    # Italian
    "mille","milione","milioni",

    # Dutch
    "duizend","miljoen","miljard",

    # Russian
    "тысяч","миллион","миллиард",

    # Chinese
    "千","万","百万","千万","亿",

    # Japanese
    "千","万","百万","億",

    # Korean
    "천","만","백만","억",

    # Arabic
    "ألف","مليون","مليار",

    # Hindi
    "हज़ार","लाख","करोड़","मिलियन",

    # Thai
    "พัน","ล้าน"
}


from pathlib import Path

def convert_doc_to_docx(doc_path: Path) -> Path:

    """
    Convert legacy .doc to .docx using Microsoft Word.
    """

    if doc_path.suffix.lower() != ".doc":
        return doc_path

    docx_path = doc_path.with_suffix(".docx")

    if docx_path.exists():
        return docx_path

    from win32com import client

    word = client.Dispatch("Word.Application")

    word.Visible = False

    try:

        document = word.Documents.Open(str(doc_path.resolve()))

        document.SaveAs(
            str(docx_path.resolve()),
            FileFormat=16,
        )

        document.Close(False)

    finally:

        word.Quit()

    return docx_path


def is_financial_table(
    rows_info,
    columns_info,
    surrounding_context=None,
):
    """
    Returns:
        is_financial, score, reasons

    Deterministic financial table detector.

    Stage 1 : Fast rejection
    Stage 2 : Financial scoring
    """

    import re

    score = 0
    reasons = []

    ####################################################
    # Extract table content
    ####################################################

    labels = []
    headers = []
    values = []

    row_count = len(rows_info)
    column_count = len(columns_info)

    for row in rows_info:

        lbl = str(row.get("label", "")).strip()

        if lbl:
            labels.append(lbl.lower())

        for v in row.get("values", []):
            values.append(str(v).strip())

    for col in columns_info:

        h = str(col.get("header", "")).strip()

        if h:
            headers.append(h.lower())

    context = ""

    if surrounding_context:
        context = " ".join(
            str(x).lower()
            for x in surrounding_context
        )

    searchable = " ".join(
        labels + headers + [context]
    )

    ####################################################
    # FAST REJECT SECTION
    ####################################################

    NON_FINANCIAL_SECTIONS = {

        "table of contents",

        "risk factors",

        "risk factors summary",

        "forward-looking",

        "overview",

        "business",

        "competition",

        "employees",

        "properties",

        "legal proceedings",

        "cybersecurity",

        "privacy",

        "glossary",

        "definitions",

        "appendix",

        "index",

        "summary",

        "abbreviations",

        "management",

        "executive officers"

    }

    for s in NON_FINANCIAL_SECTIONS:

        if s in context:

            return (
                False,
                -100,
                [f"Non-financial section ({s})"]
            )

    ####################################################
    # Page number table
    ####################################################

    if row_count == 1 and column_count == 1:

        if len(values) == 1:

            text = values[0]

            if re.fullmatch(r"\d{1,3}|[ivxlcdm]+", text.lower()):

                return (
                    False,
                    -100,
                    ["Page number"]
                )

    ####################################################
    # Bullet list detector
    ####################################################

    bullet_rows = 0

    for value in values:

        if value in {

            "●",

            "•",

            "■",

            "▪",

            "►",

            "-"

        }:

            bullet_rows += 1

    if row_count:

        if bullet_rows / row_count > 0.50:

            return (
                False,
                -100,
                ["Bullet list"]
            )

    ####################################################
    # Numeric analysis
    ####################################################

    numeric_cells = 0

    money_like = 0

    accounting_negatives = 0

    percentages = 0

    total_words = 0

    total_chars = 0

    for value in values:

        text = value

        total_words += len(text.split())

        total_chars += len(text)

        if re.search(r"\(\s*\d", text):

            accounting_negatives += 1

        if "%" in text:

            percentages += 1

        if re.fullmatch(

            r"-?\(?[\d,]+(?:\.\d+)?\)?%?",

            text,

        ):

            numeric_cells += 1

            digits = re.sub(r"\D", "", text)

            if len(digits) >= 4:

                money_like += 1

    total_cells = max(len(values), 1)

    numeric_ratio = numeric_cells / total_cells

    avg_words = total_words / total_cells

    avg_chars = total_chars / total_cells

    ####################################################
    # Narrative detector
    ####################################################

    if numeric_cells == 0 and avg_chars > 40:

        return (
            False,
            -100,
            ["Narrative table"]
        )

    if numeric_ratio < 0.05 and avg_words > 8:

        return (
            False,
            -100,
            ["Mostly prose"]
        )

    ####################################################
    # Financial keywords
    ####################################################

    keyword_hits = 0

    for kw in FINANCIAL_TERMS:

        if kw in searchable:

            keyword_hits += 1

    if keyword_hits:

        score += min(keyword_hits * 2, 10)

        reasons.append(
            f"{keyword_hits} financial keywords"
        )

    ####################################################
    # Currency
    ####################################################

    currency_hits = 0

    for kw in CURRENCY_TERMS:

        if kw in searchable:

            currency_hits += 1

    if currency_hits:

        score += 5

        reasons.append("Currency")

    ####################################################
    # Numeric score
    ####################################################

    if numeric_ratio >= 0.50:

        score += 6

        reasons.append("Mostly numeric")

    elif numeric_ratio >= 0.30:

        score += 3

        reasons.append("Many numeric values")

    if money_like >= 5:

        score += 2

        reasons.append("Money values")

    if accounting_negatives:

        score += 3

        reasons.append("Accounting negatives")

    if percentages >= 2:

        score += 1

        reasons.append("Percentages")

    ####################################################
    # Reporting periods
    ####################################################

    year_headers = 0

    for h in headers:

        if re.search(r"(19|20)\d{2}", h):

            year_headers += 1

    if year_headers >= 2:

        score += 2

        reasons.append("Reporting periods")

    ####################################################
    # Tiny tables
    ####################################################

    if row_count < 2:

        score -= 2

    ####################################################
    # Require evidence of financial data
    ####################################################

    if (
        numeric_cells < 5
        and currency_hits == 0
        and year_headers == 0
    ):

        return (
            False,
            score,
            ["Insufficient financial evidence"]
        )

    ####################################################
    # Final decision
    ####################################################

    return (
        score >= 7,
        score,
        reasons,
    )

async def export_table_inputs(
    docx_path: str,
    job_id: str | None = None,
    include_formatting: bool = False,
    classify: bool = False,
) -> Path:
    job_id = job_id or str(uuid.uuid4())
    job_id_ctx.set(job_id)

    out_dir = Path(settings.OUTPUT_DIR) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "schema_version": getattr(settings, "PARSER_SCHEMA_VERSION", None),
        "include_formatting": include_formatting,
    }

    # Stage 1: Extract
    parsed_doc = await extract_document(docx_path, config, job_id=job_id)

    # Build IDPOutput once (we do not attach an AI client here)
    idp_output = build_output(
        parsed_doc=parsed_doc,
        job_id=job_id,
        schema_version=config["schema_version"],
        ai_client=None,
        include_formatting=include_formatting,
    )

    # Use a classifier instance with no AI client for prompt building
    builder_classifier = TableClassifier(ai_client=None)

    tables = getattr(parsed_doc, "tables", []) or []

    #########################################################
    # Filter only financial tables
    #########################################################

    financial_tables = []

    print("\n==============================")
    print("Financial Table Detection")
    print("==============================")

    for table in tables:

        try:

            rows_count = len(getattr(table, "rows", []) or [])
            row_indices = list(range(rows_count))

            try:
                rows_info = builder_classifier._build_rows_summary(
                    table,
                    row_indices
                )
            except Exception:

                rows_info = []

                for i in row_indices:
                    rows_info.append(
                        {
                            "index": i,
                            "label": "",
                            "values": [],
                            "current_type": "data",
                            "needs_classification": True,
                        }
                    )

            try:
                columns_info = builder_classifier._build_columns_summary(
                    table
                )
            except Exception:

                cols = getattr(table, "ncols", None) or 0

                columns_info = [
                    {
                        "index": i,
                        "header": "",
                        "sample_values": [],
                        "current_type": "unknown",
                    }
                    for i in range(cols)
                ]

            unresolved_indices = [
                r["index"]
                for r in rows_info
                if r.get("needs_classification")
            ]

            if not unresolved_indices:
                unresolved_indices = [
                    r["index"] for r in rows_info
                ]

            unresolved_column_indices = [
                c["index"]
                for c in columns_info
                if c.get("current_type") in (None, "unknown")
            ]

            if not unresolved_column_indices:
                unresolved_column_indices = [
                    c["index"] for c in columns_info
                ]

            surrounding_context = None

            try:
                surrounding_context = _get_headings_context(
                    table,
                    idp_output.index.paragraphs,
                )
            except Exception:
                pass

            _, input_data = build_table_classification_prompt(
                rows=rows_info,
                columns=columns_info,
                unresolved_indices=unresolved_indices,
                surrounding_context=surrounding_context,
                unresolved_column_indices=unresolved_column_indices,
                use_json_grid=True,
            )

            ok, score, reasons = is_financial_table(
    rows_info,
    columns_info,
    surrounding_context,
)

            table_id = getattr(
                table,
                "table_index",
                "?"
            )

            if ok:

                financial_tables.append(table)

                print(
                    f"[KEEP] Table {table_id}"
                    f" | Score={score}"
                    f" | {'; '.join(reasons)}"
                )

            else:

                print(
                    f"[SKIP] Table {table_id}"
                    f" | Score={score}"
                    f" | {'; '.join(reasons)}"
                )

        except Exception as e:

            print(
                f"Financial detection failed "
                f"for table {table.table_index}: {e}"
            )

    print(
        f"\nDetected "
        f"{len(financial_tables)} financial tables "
        f"out of {len(tables)} total tables.\n"
    )

    # Only Azure OpenAI is supported for classification in this script.
    # This also guarantees the `use_json_grid` path in the classifier.
    ai_client = create_ai_client("azure") if classify else None

    for table in financial_tables:
        try:
            # Build rows summary
            rows_count = len(getattr(table, "rows", []) or [])
            row_indices = list(range(rows_count))
            try:
                rows_info = builder_classifier._build_rows_summary(table, row_indices)
            except Exception:
                # Fallback: synthesize minimal rows_info
                rows_info = []
                for i in row_indices:
                    rows_info.append({"index": i, "label": "", "values": [], "current_type": "data", "needs_classification": True})

            # Build columns summary
            try:
                columns_info = builder_classifier._build_columns_summary(table)
            except Exception:
                # Fallback: synthesize columns from table.matrix or width
                cols = getattr(table, "ncols", None) or 0
                columns_info = [{"index": i, "header": "", "sample_values": [], "current_type": "unknown"} for i in range(cols)]

            # Determine unresolved indices (prefer rows marked as needing classification)
            unresolved_indices = [r["index"] for r in rows_info if r.get("needs_classification")]
            if not unresolved_indices:
                unresolved_indices = [r["index"] for r in rows_info]

            unresolved_column_indices = [c["index"] for c in columns_info if c.get("current_type") in (None, "unknown")]
            if not unresolved_column_indices:
                unresolved_column_indices = [c["index"] for c in columns_info]

            # Surrounding context (headings)
            surrounding_context = None
            try:
                surrounding_context = _get_headings_context(table, idp_output.index.paragraphs)
            except Exception:
                surrounding_context = None

            # Build prompt/input payload
            prompt_msgs, input_data = build_table_classification_prompt(
                rows=rows_info,
                columns=columns_info,
                unresolved_indices=unresolved_indices,
                surrounding_context=surrounding_context,
                unresolved_column_indices=unresolved_column_indices,
                use_json_grid=True,
            )

            table_id = getattr(table, "element_id", None) or getattr(table, "table_id", None) or str(getattr(table, "table_index", "unknown"))
            input_path = out_dir / f"classify_{table_id}_input.json"
            # Save a post-processed input where short cell keys are expanded to full names
            try:
                full_input = _postprocess_input(input_data)
            except Exception:
                full_input = input_data
            with input_path.open("w", encoding="utf-8") as fh:
                json.dump(full_input, fh, ensure_ascii=False, indent=2)

            # Optionally save the human-readable prompt messages too
            prompt_path = out_dir / f"classify_{table_id}_prompt.json"
            with prompt_path.open("w", encoding="utf-8") as fh:
                json.dump(prompt_msgs, fh, ensure_ascii=False, indent=2)

        except Exception as exc:  # keep processing remaining tables
            print(f"Failed to build input for table {getattr(table, 'table_index', '?')}: {exc}")

    # Write a minimal job manifest
    manifest = {
    "job_id": job_id,
    "docx": str(docx_path),

    "table_count": len(tables),
    "financial_table_count": len(financial_tables),
    "skipped_table_count": len(tables) - len(financial_tables),

    "classified": bool(ai_client),
    "ai_provider": "azure" if ai_client else None,
}
    with (out_dir / "job.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # Run classification using the AI client (this also persists raw responses)
    if ai_client is not None and financial_tables:
        try:
            await TableClassifier(ai_client=ai_client).classify_tables_batch_async(
    financial_tables,
    job_id=job_id,
    idp_paragraphs=idp_output.index.paragraphs,
    store_input_output=True,
)
            # Post-process any saved AI response files to expand short keys to full names
            try:
                for resp_path in out_dir.glob("classify_*_response.json"):
                    _postprocess_response_file(resp_path)
            except Exception:
                # non-fatal; proceed
                pass
        except Exception as exc:
            print(f"Warning: classification step failed: {exc}", file=sys.stderr)

    print("\n========================================")
    print("Processing Summary")
    print("========================================")
    print(f"Total tables        : {len(tables)}")
    print(f"Financial tables    : {len(financial_tables)}")
    print(f"Skipped tables      : {len(tables)-len(financial_tables)}")
    print("========================================\n")

    return out_dir


async def process_directory(
    input_dir: str,
    classify: bool,
    include_formatting: bool,
):

    input_dir = Path(input_dir)

    files = []

    files.extend(sorted(input_dir.glob("*.doc")))

    files.extend(sorted(input_dir.glob("*.docx")))

    print(f"\nFound {len(files)} documents\n")

    for idx, file in enumerate(files, 1):

        print("=" * 70)
        print(f"[{idx}/{len(files)}] {file.name}")
        print("=" * 70)

        try:

            docx_file = convert_doc_to_docx(file)

            await export_table_inputs(
                str(docx_file),
                classify=classify,
                include_formatting=include_formatting,
            )

        except Exception as e:

            print(f"FAILED : {file.name}")
            print(e)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export table LLM classification inputs from a DOCX")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--docx",
        help="Single DOCX/DOC file",
    )
    group.add_argument(
        "--input",
        help="Directory containing DOC/DOCX files",
    )
    p.add_argument("--job-id", required=False, help="Optional job id (defaults to uuid4)")
    p.add_argument("--include-formatting", action="store_true", help="Preserve formatting in output prompts")
    p.add_argument("--classify", action="store_true", help="Run AI classification and save responses (Azure only)")
    return p.parse_args()


def main() -> None:

    args = _parse_args()

    if args.docx:

        doc_file = convert_doc_to_docx(
            Path(args.docx)
        )

        out_dir = asyncio.run(
            export_table_inputs(
                str(doc_file),
                job_id=args.job_id,
                include_formatting=args.include_formatting,
                classify=args.classify,
            )
        )

        print(f"\nFinished.")
        print(f"Outputs written to:\n{out_dir}")

    else:

        asyncio.run(
            process_directory(
                args.input,
                classify=args.classify,
                include_formatting=args.include_formatting,
            )
        )

        print("\nBatch processing completed.")


if __name__ == "__main__":
    main()
