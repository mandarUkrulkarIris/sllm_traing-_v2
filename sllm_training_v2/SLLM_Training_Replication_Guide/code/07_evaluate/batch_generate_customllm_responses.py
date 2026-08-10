"""
Batch-run generate_customllm_responses.py, one job directory after another, across
every job directory found under a parent input directory - so a whole folder of
docx-exported jobs (each containing classify_<id>_input.json files from
export_table_classification_inputs_filterfin.py) can be pushed through the custom LLM
sequentially without invoking the script by hand for each one.

Jobs run strictly one at a time (not concurrently): each job is fully submitted,
polled to completion, and its responses written before the next job starts. This
matches how the underlying script already talks to api_GPU.py's shared job queue -
submitting several jobs' worth of tables at once wouldn't make them process any
faster (they'd just compete for the same GPU slots), so sequential keeps behavior
predictable and per-job progress output readable.

Usage:
    python batch_generate_customllm_responses.py --input-dir "D:\\Dev\\git_repo\\TFS\\IntelligentDocumentParser\\output"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import generate_customllm_responses as generator


def find_job_dirs(input_dir: Path) -> list[Path]:
    """Every immediate subdirectory containing classify_*_input.json files, or
    input_dir itself if it directly contains them (so a single job dir also works)."""
    if any(input_dir.glob("classify_*_input.json")):
        return [input_dir]

    job_dirs = []
    for child in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if any(child.glob("classify_*_input.json")):
            job_dirs.append(child)
    return job_dirs


def run_with_argv(module, argv: list[str]) -> bool:
    """Call module.main() with a temporary sys.argv, returning True on success.
    main() calls sys.exit() on its own expected failure paths (no input files found,
    job ended failed/cancelled, timed out) - catching SystemExit here lets the batch
    move on to the next job instead of aborting the whole run over one bad job."""
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
        description="Run generate_customllm_responses.py sequentially over every job "
        "directory found under --input-dir."
    )
    parser.add_argument("--input-dir", required=True, help="Parent directory containing one or more job directories")
    parser.add_argument("--base-url", default="https://devtablelayout.iriscarbon.com/", help="api_GPU.py base URL")
    parser.add_argument("--api-key", help="X-API-Key header, if the server has API_KEY set")
    parser.add_argument("--priority", action="store_true", help="Submit each job as a priority job")
    parser.add_argument("--suffix", default="_customllm", help="Suffix inserted before .json in output filenames")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Max seconds to wait for each job to finish")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    job_dirs = find_job_dirs(input_dir)

    if not job_dirs:
        print(f"No job directories with classify_*_input.json files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    plural = "y" if len(job_dirs) == 1 else "ies"
    print(f"Found {len(job_dirs)} job director{plural} under {input_dir}.\n")

    common_args = [
        "--base-url", args.base_url,
        "--suffix", args.suffix,
        "--poll-interval", str(args.poll_interval),
        "--timeout", str(args.timeout),
    ]
    if args.api_key:
        common_args += ["--api-key", args.api_key]
    if args.priority:
        common_args += ["--priority"]

    results = []
    for i, job_dir in enumerate(job_dirs, 1):
        print("=" * 70)
        print(f"[{i}/{len(job_dirs)}] {job_dir.name}")
        print("=" * 70)

        ok = run_with_argv(generator, ["--job-dir", str(job_dir)] + common_args)
        results.append((job_dir, ok))
        print()

    print("=" * 70)
    print("Batch summary")
    print("=" * 70)
    for job_dir, ok in results:
        print(f"  [{'OK' if ok else 'FAILED'}] {job_dir.name}")

    failures = sum(1 for _, ok in results if not ok)
    if failures:
        print(f"\n{failures}/{len(results)} job(s) failed - see output above.")
        sys.exit(1)

    print(f"\nAll {len(results)} job(s) generated successfully.")


if __name__ == "__main__":
    main()
