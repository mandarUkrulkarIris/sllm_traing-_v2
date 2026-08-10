"""
Run tables already exported for GPT-4.1 through the custom LLM served by api_GPU.py,
for benchmarking one against the other.

`export_table_classification_inputs_filterfin.py` (IntelligentDocumentParser) writes one
classify_<table_id>_input.json per financial table into output/<job_id>/, then (with
--classify) sends each through GPT-4.1 and saves classify_<table_id>_response.json next
to it. This script reuses those same *_input.json files as-is, submits them to
api_GPU.py's /predict queue, and writes the custom model's answers as
classify_<table_id>_response_customllm.json in the same folder - so both responses for
a given table sit side by side under the same table_id and can be diffed directly.

Usage:
    python generate_customllm_responses.py --job-dir "D:\\Dev\\git_repo\\TFS\\IntelligentDocumentParser\\output\\<job_id>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

INPUT_FILE_RE = re.compile(r"^classify_(.+)_input\.json$")


def discover_tables(job_dir: Path) -> list[tuple[str, dict]]:
    """Return [(table_id, table_json), ...] for every classify_<id>_input.json in job_dir."""
    entries = []
    for path in sorted(job_dir.glob("classify_*_input.json")):
        m = INPUT_FILE_RE.match(path.name)
        if not m:
            continue
        table_id = m.group(1)
        try:
            table_json = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] Skipping {path.name}: could not parse JSON ({e})", file=sys.stderr)
            continue
        entries.append((table_id, table_json))
    return entries


def submit_job(base_url: str, tables: list[dict], api_key: str | None, priority: bool) -> str:
    payload = json.dumps(tables, ensure_ascii=False).encode("utf-8")
    files = {"file": ("tables.json", payload, "application/json")}
    data = {"priority": "true" if priority else "false"}
    headers = {"X-API-Key": api_key} if api_key else {}

    resp = requests.post(f"{base_url}/predict", files=files, data=data, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["job_id"]


def wait_for_completion(base_url: str, job_id: str, api_key: str | None, poll_interval: float, timeout: float) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    deadline = time.monotonic() + timeout

    while True:
        resp = requests.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        status = resp.json()

        print(
            f"\r[{status['status']}] {status['completed_tables']}/{status['total_tables']} tables "
            f"({status['progress']}%)",
            end="",
            flush=True,
        )

        if status["status"] in ("completed", "failed", "cancelled"):
            print()
            return status

        if time.monotonic() > deadline:
            print()
            raise TimeoutError(f"Job {job_id} did not finish within {timeout}s (last status: {status['status']})")

        time.sleep(poll_interval)


def fetch_results(base_url: str, job_id: str, api_key: str | None) -> list:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.get(f"{base_url}/results/{job_id}", headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tables already exported for GPT-4.1 through api_GPU.py's custom LLM, "
        "saving matching classify_<table_id>_response_customllm.json files for comparison."
    )
    parser.add_argument("--job-dir", required=True, help="Path to output/<job_id> folder containing classify_*_input.json")
    parser.add_argument("--out-dir", help="Where to write response files (default: same as --job-dir)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="api_GPU.py base URL")
    parser.add_argument("--api-key", help="X-API-Key header, if the server has API_KEY set")
    parser.add_argument("--priority", action="store_true", help="Submit as a priority job")
    parser.add_argument("--suffix", default="_customllm", help="Suffix inserted before .json in output filenames")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between status polls")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Max seconds to wait for the job to finish")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    out_dir = Path(args.out_dir) if args.out_dir else job_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_tables(job_dir)
    if not entries:
        print(f"No classify_*_input.json files found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    table_ids = [tid for tid, _ in entries]
    tables = [t for _, t in entries]
    print(f"Found {len(tables)} table(s) in {job_dir}: {', '.join(table_ids)}")

    print(f"Submitting to {args.base_url}/predict ...")
    job_id = submit_job(args.base_url, tables, args.api_key, args.priority)
    print(f"Queued as job {job_id}")

    status = wait_for_completion(args.base_url, job_id, args.api_key, args.poll_interval, args.timeout)

    if status["status"] != "completed":
        print(f"Job ended with status '{status['status']}': {status.get('error')}", file=sys.stderr)
        sys.exit(1)

    results = fetch_results(args.base_url, job_id, args.api_key)
    if len(results) != len(table_ids):
        print(
            f"[!] Warning: got {len(results)} result(s) for {len(table_ids)} submitted table(s); "
            f"writing by position, verify alignment.",
            file=sys.stderr,
        )

    manifest = {
        "job_id": job_id,
        "model_name": status.get("model_name"),
        "source_job_dir": str(job_dir),
        "time_taken": status.get("time_taken"),
        "tables": [],
    }

    for idx, result in enumerate(results):
        table_id = table_ids[idx] if idx < len(table_ids) else f"index-{idx}"
        out_path = out_dir / f"classify_{table_id}_response{args.suffix}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["tables"].append({"table_id": table_id, "output_file": out_path.name, "error": result.get("error")})
        marker = "ERROR" if "error" in result else "ok"
        print(f"  [{marker}] {table_id} -> {out_path.name}")

    manifest_path = out_dir / f"customllm_manifest{args.suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(results)} response(s) written to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
