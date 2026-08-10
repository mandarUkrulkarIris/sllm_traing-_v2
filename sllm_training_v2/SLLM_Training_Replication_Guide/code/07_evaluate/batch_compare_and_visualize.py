"""
Batch-run compare_llm_responses.py, then visualize_comparison.py, then
visualize_table.py for every table, across every job directory found under a parent
input directory - so a whole benchmark run (many job_id folders, each already
containing GPT-4.1 + custom-LLM response pairs from generate_customllm_responses.py)
gets compared and fully charted in one shot.

A "job directory" is any immediate subdirectory of --input-dir that contains at least
one classify_*_response.json file (the same discovery rule compare_llm_responses.py
itself uses to find GPT-4.1 reference responses). --input-dir may also directly BE a
single job directory - both cases are handled.

Per job, in order:
  1. compare_llm_responses.py            -> compare_<id>.json + compare_summary.json
  2. visualize_comparison.py             -> compare_summary.png (aggregate chart)
  3. visualize_table.py, once per table  -> compare_<id>.png (per-table grid)

All three scripts are imported and driven in-process rather than subprocess'd, so the
sentence-transformers embedding model loads once (~15s) and is reused across every
job and every table in the batch, instead of paying that cost repeatedly.

Usage:
    python batch_compare_and_visualize.py --input-dir "D:\\Dev\\output"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import compare_llm_responses as comparer
import visualize_comparison as visualizer
import visualize_table as table_visualizer


def find_job_dirs(input_dir: Path) -> list[Path]:
    """Every immediate subdirectory containing classify_*_response.json files, or
    input_dir itself if it directly contains them (so a single job dir also works)."""
    if any(input_dir.glob("classify_*_response.json")):
        return [input_dir]

    job_dirs = []
    for child in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if any(child.glob("classify_*_response.json")):
            job_dirs.append(child)
    return job_dirs


def find_table_reports(compare_dir: Path) -> list[Path]:
    """Every per-table compare_<table_id>.json written by compare_llm_responses.py,
    excluding the aggregate compare_summary.json."""
    return sorted(p for p in compare_dir.glob("compare_*.json") if p.name != "compare_summary.json")


def run_with_argv(module, argv: list[str]) -> bool:
    """Call module.main() with a temporary sys.argv, returning True on success.
    main() calls sys.exit() on its own expected failure paths (e.g. no comparable
    pairs in a job dir) - catching SystemExit here lets the batch move on to the next
    job/table instead of aborting the whole run over one bad one."""
    old_argv = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
        return True
    except SystemExit as e:
        return e.code in (None, 0)
    except Exception as e:
        print(f"[!] Unexpected error running {module.__name__}: {e}", file=sys.stderr)
        return False
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compare_llm_responses.py, visualize_comparison.py, and "
        "visualize_table.py (per table) over every job directory found under --input-dir."
    )
    parser.add_argument("--input-dir", required=True, help="Parent directory containing one or more job directories")
    parser.add_argument(
        "--candidate-suffix", default="_customllm", help="Suffix used for the custom LLM's response files"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    job_dirs = find_job_dirs(input_dir)

    if not job_dirs:
        print(f"No job directories with classify_*_response.json files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    plural = "y" if len(job_dirs) == 1 else "ies"
    print(f"Found {len(job_dirs)} job director{plural} under {input_dir}.\n")

    results = []
    for i, job_dir in enumerate(job_dirs, 1):
        print("=" * 70)
        print(f"[{i}/{len(job_dirs)}] {job_dir.name}")
        print("=" * 70)

        compare_ok = run_with_argv(
            comparer, ["--job-dir", str(job_dir), "--candidate-suffix", args.candidate_suffix]
        )

        compare_dir = job_dir / "compare"
        summary_path = compare_dir / "compare_summary.json"

        chart_ok = False
        if compare_ok and summary_path.exists():
            chart_ok = run_with_argv(visualizer, ["--summary", str(summary_path)])
        elif not compare_ok:
            print(f"[!] Comparison failed for {job_dir.name}, skipping charts.", file=sys.stderr)

        table_total = table_failures = 0
        if compare_ok:
            table_reports = find_table_reports(compare_dir)
            table_total = len(table_reports)
            print(f"Rendering {table_total} per-table chart(s)...")
            for report_path in table_reports:
                ok = run_with_argv(table_visualizer, ["--report", str(report_path)])
                if not ok:
                    table_failures += 1
                    print(f"[!] Table chart failed for {report_path.name}", file=sys.stderr)

        results.append((job_dir, compare_ok, chart_ok, table_total, table_failures))
        print()

    print("=" * 70)
    print("Batch summary")
    print("=" * 70)
    for job_dir, compare_ok, chart_ok, table_total, table_failures in results:
        if not compare_ok:
            status = "compare failed"
        elif not chart_ok:
            status = "summary chart failed"
        elif table_failures:
            status = f"{table_failures}/{table_total} table chart(s) failed"
        else:
            status = f"OK ({table_total} table chart(s))"
        print(f"  [{status}] {job_dir.name}")

    failures = sum(1 for _, c, v, _, tf in results if not (c and v) or tf)
    if failures:
        print(f"\n{failures}/{len(results)} job(s) had a problem - see output above.")
        sys.exit(1)

    print(f"\nAll {len(results)} job(s) compared and charted successfully.")


if __name__ == "__main__":
    main()
