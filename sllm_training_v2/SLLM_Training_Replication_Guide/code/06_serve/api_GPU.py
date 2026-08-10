import os
import json
import re
import asyncio
import uuid
import logging
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Depends, Header
from fastapi.responses import PlainTextResponse
from json_repair import repair_json
from enum import Enum
from pydantic import BaseModel

import httpx

# ==========================================================
# LOGGING
# ==========================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("table_layout_api")

# ==========================================================
# CONFIG & GLOBAL STATE
# ==========================================================
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/completion")
LLAMA_HEALTH_URL = os.getenv("LLAMA_HEALTH_URL", "http://127.0.0.1:8080/health")
LLAMA_CHAT_URL = os.getenv("LLAMA_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")

DB_PATH = os.getenv("DB_PATH", "data/jobs.db")

MODEL_PATH = os.getenv("MODEL_PATH", "unknown_model")
MODEL_NAME = os.path.basename(MODEL_PATH)

CLEAR_QUEUE_ON_RESTART = os.getenv("CLEAR_QUEUE_ON_RESTART", "false").lower() == "true"

# Controls how many tables process in parallel on the GPU
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))

# Transient inference-call retries before a table is recorded as failed
INFERENCE_MAX_RETRIES = int(os.getenv("INFERENCE_MAX_RETRIES", "3"))

# Reject uploads larger than this before they're fully buffered in memory
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Completed/failed/cancelled jobs older than this are purged by the dispatcher.
# 0 disables the sweep.
JOB_RETENTION_DAYS = int(os.getenv("JOB_RETENTION_DAYS", "30"))
_last_retention_sweep = 0.0

# If unset, authentication is disabled (local/dev use only).
API_KEY = os.getenv("API_KEY")

ACTIVE_TASKS: Dict[str, set] = {}

# Simple in-process counters surfaced on /metrics
REPAIR_SUCCESS_COUNT = 0
REPAIR_FAILURE_COUNT = 0


def format_timestamp(epoch: Optional[float]) -> Optional[str]:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


async def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# ==========================================================
# DATABASE SETUP & HELPERS
# ==========================================================
@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(conn, table: str, column: str, coltype: str):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                input_data TEXT,
                error TEXT,
                time_taken REAL DEFAULT 0.0,
                completed_tables INTEGER DEFAULT 0,
                total_tables INTEGER DEFAULT 0,
                created_at REAL,
                start_time REAL,
                priority INTEGER DEFAULT 0,
                input_file_name TEXT,
                model_name TEXT
            )
        """)
        _ensure_column(conn, "jobs", "input_file_name", "TEXT")
        _ensure_column(conn, "jobs", "model_name", "TEXT")

        # Per-table results, written incrementally instead of rewriting one growing
        # JSON blob on every table completion (see run_table_task).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_table_results (
                job_id TEXT NOT NULL,
                table_index INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY (job_id, table_index)
            )
        """)

        if CLEAR_QUEUE_ON_RESTART:
            logger.info("CLEAR_QUEUE_ON_RESTART is True. Cancelling all leftover jobs...")
            conn.execute("""
                UPDATE jobs
                SET status = 'cancelled', error = 'Cancelled due to server restart flush.'
                WHERE status IN ('queued', 'processing', 'paused')
            """)


def _cleanup_old_jobs():
    """Purge completed/failed/cancelled jobs older than JOB_RETENTION_DAYS. Throttled to
    run at most once per hour since this is called from the dispatcher's poll loop."""
    global _last_retention_sweep
    if JOB_RETENTION_DAYS <= 0:
        return
    now = time.time()
    if now - _last_retention_sweep < 3600:
        return
    _last_retention_sweep = now

    cutoff = now - JOB_RETENTION_DAYS * 86400
    with get_db() as conn:
        old_ids = [row["id"] for row in conn.execute(
            "SELECT id FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') AND created_at < ?",
            (cutoff,)
        ).fetchall()]

        if old_ids:
            placeholders = ",".join("?" * len(old_ids))
            conn.execute(f"DELETE FROM job_table_results WHERE job_id IN ({placeholders})", old_ids)
            conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", old_ids)
            logger.info("Retention sweep removed %d job(s) older than %d day(s).", len(old_ids), JOB_RETENTION_DAYS)


# ==========================================================
# INFERENCE LOGIC FUNCTIONS
# ==========================================================
async def run_inference_async(prompt_text: str) -> str:
    payload = {"prompt": prompt_text, "n_predict": 8192, "temperature": 0.0, "stop": ["<|im_end|>", "<|endoftext|>"]}
    last_exc = None

    for attempt in range(INFERENCE_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(LLAMA_SERVER_URL, json=payload)
                response.raise_for_status()
                return response.json().get("content", "")
        except httpx.HTTPError as e:
            last_exc = e
            if attempt < INFERENCE_MAX_RETRIES - 1:
                backoff = 2 ** attempt
                logger.warning(
                    "Inference call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, INFERENCE_MAX_RETRIES, e, backoff
                )
                await asyncio.sleep(backoff)

    raise last_exc


async def process_single_table(table: Any, idx: int) -> dict:
    global REPAIR_SUCCESS_COUNT, REPAIR_FAILURE_COUNT

    table_string = json.dumps(table, ensure_ascii=False)
    assigned_index = table.get("table_index", idx) if isinstance(table, dict) else idx
    prompt = f"###Instruction: You are a financial table structure analyst. Analyze the provided table JSON (row/column/cell matrix) and output JSON object only.\n### Input:\n{table_string}\n### Response:\n"

    try:
        raw_response = await run_inference_async(prompt)
    except Exception as e:
        logger.error("Inference failed for table %s after retries: %s", assigned_index, e)
        REPAIR_FAILURE_COUNT += 1
        return {"error": f"Inference failed: {e}", "table_index": assigned_index}

    cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()

    try:
        parsed_json = repair_json(cleaned_response, return_objects=True)
        if isinstance(parsed_json, list) and len(parsed_json) > 0: parsed_json = parsed_json[0]
        if not isinstance(parsed_json, dict): parsed_json = {"error": "Invalid format", "raw": cleaned_response}
        parsed_json["table_index"] = assigned_index

        REPAIR_SUCCESS_COUNT += 1
        return parsed_json
    except Exception as e:
        REPAIR_FAILURE_COUNT += 1
        return {"error": str(e), "raw_output": raw_response, "table_index": assigned_index}


# ==========================================================
# GLOBAL PARALLEL JOB DISPATCHER
# ==========================================================
# Tables are dispatched across ALL active jobs against ONE shared pool of
# MAX_CONCURRENT_REQUESTS GPU slots (matches llama-server's -np setting), instead
# of a per-job pool. This lets a 1-table job share the GPU with tables pulled
# from other queued jobs rather than monopolizing/wasting concurrency alone.
#
# NOTE: ACTIVE_JOBS/ACTIVE_TASKS/IN_FLIGHT are in-process memory, and job claiming
# below only guards against this process racing itself. This API must run as a
# single process (one uvicorn worker, one container replica) — running multiple
# workers/replicas against the same DB_PATH will double-process jobs.
class JobRuntime:
    def __init__(self, job_id, tables, total, completed_count, start_time, previous_time_taken):
        self.job_id = job_id
        self.tables = tables
        self.total = total
        self.next_index = completed_count  # next table index not yet dispatched
        self.completed_count = completed_count
        self.start_time = start_time
        self.previous_time_taken = previous_time_taken
        self.lock = asyncio.Lock()


ACTIVE_JOBS: Dict[str, JobRuntime] = {}
IN_FLIGHT = 0  # count of table-processing tasks currently running against the GPU, across all jobs


async def finalize_job(job: JobRuntime):
    with get_db() as conn:
        status_check = conn.execute("SELECT status FROM jobs WHERE id = ?", (job.job_id,)).fetchone()
        if status_check and status_check[0] == 'processing':
            session_time = time.time() - job.start_time
            conn.execute(
                "UPDATE jobs SET status = 'completed', time_taken = ? WHERE id = ?",
                (job.previous_time_taken + session_time, job.job_id)
            )
    ACTIVE_JOBS.pop(job.job_id, None)
    ACTIVE_TASKS.pop(job.job_id, None)


async def run_table_task(job: JobRuntime):
    global IN_FLIGHT
    idx = job.next_index
    job.next_index += 1
    table = job.tables[idx]
    current_task = asyncio.current_task()

    try:
        # Check if job was paused/cancelled before starting heavy GPU workload
        with get_db() as conn:
            status_check = conn.execute("SELECT status FROM jobs WHERE id = ?", (job.job_id,)).fetchone()
            if not status_check or status_check[0] in ('cancelled', 'paused'):
                return

        # Transient inference/parse failures are recorded as a per-table error result
        # inside process_single_table and never raise here, so one bad table can't
        # take down the rest of the job.
        res = await process_single_table(table, idx)

        async with job.lock:
            with get_db() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO job_table_results (job_id, table_index, result_json) VALUES (?, ?, ?)",
                    (job.job_id, idx, json.dumps(res, ensure_ascii=False))
                )
                conn.execute(
                    "UPDATE jobs SET completed_tables = completed_tables + 1 WHERE id = ?",
                    (job.job_id,)
                )
            job.completed_count += 1

        if job.completed_count >= job.total:
            await finalize_job(job)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception("Processing error for Job %s (table %s)", job.job_id, idx)
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status = 'failed', error = ? WHERE id = ?", (str(e), job.job_id))
        ACTIVE_JOBS.pop(job.job_id, None)
        ACTIVE_TASKS.pop(job.job_id, None)
    finally:
        IN_FLIGHT -= 1
        ACTIVE_TASKS.get(job.job_id, set()).discard(current_task)


async def job_dispatcher_worker():
    global IN_FLIGHT
    logger.info("Global Parallel Dispatcher Started (Max Concurrency: %d).", MAX_CONCURRENT_REQUESTS)
    while True:
        try:
            _cleanup_old_jobs()

            with get_db() as conn:
                cur = conn.execute("""
                    SELECT id, status, input_data, completed_tables, total_tables, time_taken
                    FROM jobs
                    WHERE status IN ('queued', 'processing')
                    ORDER BY priority DESC, created_at ASC
                """)
                rows = cur.fetchall()

            active_ids = set()
            for row in rows:
                job_id = row["id"]
                active_ids.add(job_id)

                if job_id not in ACTIVE_JOBS:
                    json_data = json.loads(row["input_data"])
                    start_time = time.time()

                    if row["status"] == "queued":
                        # Only 'queued' rows need claiming — atomically flip to 'processing'
                        # so a second process/cycle racing on the same row backs off instead
                        # of double-processing it.
                        with get_db() as conn:
                            claim = conn.execute(
                                "UPDATE jobs SET status = 'processing', start_time = ? WHERE id = ? AND status = 'queued'",
                                (start_time, job_id)
                            )
                            if claim.rowcount == 0:
                                continue  # lost the race — pick it up next cycle
                    # else: already 'processing' (e.g. re-adopted after this process restarted
                    # without CLEAR_QUEUE_ON_RESTART) — no claim needed, just resume it.

                    ACTIVE_JOBS[job_id] = JobRuntime(
                        job_id, json_data, row["total_tables"], row["completed_tables"],
                        start_time, row["time_taken"]
                    )
                    ACTIVE_TASKS.setdefault(job_id, set())

            # Jobs no longer queued/processing (paused/cancelled) drop out of the runtime pool
            for stale_id in list(ACTIVE_JOBS.keys()):
                if stale_id not in active_ids:
                    ACTIVE_JOBS.pop(stale_id, None)
                    for t in ACTIVE_TASKS.pop(stale_id, set()):
                        t.cancel()

            # Fill available global GPU slots, preserving priority/creation order across jobs.
            # Round-robin over `rows` repeatedly so ONE job (e.g. the only job in queue) can
            # claim all free slots instead of just one table per pass.
            dispatched = True
            while IN_FLIGHT < MAX_CONCURRENT_REQUESTS and dispatched:
                dispatched = False
                for row in rows:
                    if IN_FLIGHT >= MAX_CONCURRENT_REQUESTS:
                        break
                    job = ACTIVE_JOBS.get(row["id"])
                    if job and job.next_index < job.total:
                        IN_FLIGHT += 1
                        task = asyncio.create_task(run_table_task(job))
                        ACTIVE_TASKS[job.job_id].add(task)
                        dispatched = True

            await asyncio.sleep(0.5)

        except Exception:
            logger.exception("Critical Dispatcher Loop Error")
            await asyncio.sleep(2)


# ==========================================================
# LIFESPAN / STARTUP
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    if not API_KEY:
        logger.warning("API_KEY is not set - authentication is DISABLED. Do not run like this in production.")

    logger.info("Checking connection to local llama.cpp server...")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(LLAMA_HEALTH_URL)
            if response.status_code == 200:
                logger.info("Successfully connected to local llama-server! Active Model: %s", MODEL_NAME)
    except httpx.HTTPError:
        logger.warning("Could not connect to llama-server. Ensure it's running on port 8080.")

    worker_task = asyncio.create_task(job_dispatcher_worker())
    yield
    worker_task.cancel()


app = FastAPI(title=f"Native Math TableLayout API v2 - Active Model: {MODEL_NAME}", lifespan=lifespan)


# ==========================================================
# RESPONSE MODELS
# ==========================================================
class HealthResponse(BaseModel):
    status: str
    llama_server_connected: bool
    active_model: str
    jobs_in_queue: int
    max_concurrent_requests: int


class PredictResponse(BaseModel):
    status: str
    job_id: str
    priority: bool
    input_file_name: str
    model_name: str
    created_at: Optional[str] = None
    message: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    completed_tables: int
    total_tables: int
    time_taken: float
    error: Optional[str] = None
    priority: bool
    input_file_name: Optional[str] = None
    model_name: Optional[str] = None
    created_at: Optional[str] = None
    progress: float


class JobSummary(BaseModel):
    id: str
    status: str
    completed_tables: int
    total_tables: int
    time_taken: float
    priority: bool
    input_file_name: Optional[str] = None
    model_name: Optional[str] = None
    created_at: Optional[str] = None
    progress: float


class JobsListResponse(BaseModel):
    count: int
    jobs: List[JobSummary]


class ActionResponse(BaseModel):
    status: str
    message: str


# ==========================================================
# ENDPOINTS
# ==========================================================
@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            req = await client.get(LLAMA_HEALTH_URL)
            model_up = req.status_code == 200
    except httpx.HTTPError:
        model_up = False

    with get_db() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'")
        queue_size = cur.fetchone()[0]

    return HealthResponse(
        status="healthy" if model_up else "degraded",
        llama_server_connected=model_up,
        active_model=MODEL_NAME,
        jobs_in_queue=queue_size,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    with get_db() as conn:
        status_counts = {
            row["status"]: row["c"]
            for row in conn.execute("SELECT status, COUNT(*) as c FROM jobs GROUP BY status").fetchall()
        }

    lines = [
        "# HELP table_layout_in_flight_requests Table-processing tasks currently running against the GPU.",
        "# TYPE table_layout_in_flight_requests gauge",
        f"table_layout_in_flight_requests {IN_FLIGHT}",
        "# HELP table_layout_max_concurrent_requests Configured max concurrent GPU slots.",
        "# TYPE table_layout_max_concurrent_requests gauge",
        f"table_layout_max_concurrent_requests {MAX_CONCURRENT_REQUESTS}",
        "# HELP table_layout_jobs_by_status Number of jobs currently in each status.",
        "# TYPE table_layout_jobs_by_status gauge",
    ]
    for status_name, count in status_counts.items():
        lines.append(f'table_layout_jobs_by_status{{status="{status_name}"}} {count}')

    lines += [
        "# HELP table_layout_json_repair_success_total Table responses successfully parsed as JSON.",
        "# TYPE table_layout_json_repair_success_total counter",
        f"table_layout_json_repair_success_total {REPAIR_SUCCESS_COUNT}",
        "# HELP table_layout_json_repair_failure_total Table responses that failed inference or JSON parsing/repair.",
        "# TYPE table_layout_json_repair_failure_total counter",
        f"table_layout_json_repair_failure_total {REPAIR_FAILURE_COUNT}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(
    file: UploadFile = File(...),
    priority: bool = Form(False)
):
    try:
        buffer = bytearray()
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max upload size of {MAX_UPLOAD_MB}MB."
                )

        json_data = json.loads(buffer.decode("utf-8"))

        if isinstance(json_data, dict): json_data = [json_data]

        job_id = str(uuid.uuid4())
        total = len(json_data)
        priority_flag = 1 if priority else 0
        file_name = file.filename or "unknown.json"
        created_at = time.time()

        with get_db() as conn:
            conn.execute("""
                INSERT INTO jobs (id, status, input_data, completed_tables, total_tables, created_at, priority, input_file_name, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, 'queued', json.dumps(json_data), 0, total, created_at, priority_flag, file_name, MODEL_NAME))

        return PredictResponse(
            status="queued",
            job_id=job_id,
            priority=priority,
            input_file_name=file_name,
            model_name=MODEL_NAME,
            created_at=format_timestamp(created_at),
            message="Priority job added." if priority else "Job added to queue."
        )

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")
    except Exception:
        logger.exception("Failed to enqueue job from upload '%s'", file.filename)
        raise HTTPException(status_code=500, detail="Failed to process upload.")


@app.get("/status/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_key)])
async def check_status(job_id: str):
    with get_db() as conn:
        cur = conn.execute("SELECT id, status, completed_tables, total_tables, time_taken, error, priority, input_file_name, model_name, created_at FROM jobs WHERE id = ?", (job_id,))
        job = cur.fetchone()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = dict(job)
    job_dict["priority"] = bool(job_dict["priority"])
    job_dict["created_at"] = format_timestamp(job_dict["created_at"])

    if job_dict["total_tables"] > 0:
        job_dict["progress"] = round((job_dict["completed_tables"] / job_dict["total_tables"]) * 100, 2)
    else:
        job_dict["progress"] = 0.0

    return job_dict


@app.get("/results/{job_id}", dependencies=[Depends(require_api_key)])
async def get_results(job_id: str):
    with get_db() as conn:
        cur = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        job = cur.fetchone()

        if not job:
            raise HTTPException(status_code=404, detail="Job ID not found")
        if job["status"] in ["queued", "processing", "paused"]:
            raise HTTPException(status_code=400, detail=f"Job is currently {job['status']}.")
        if job["status"] in ["failed", "cancelled"]:
            raise HTTPException(status_code=500, detail=f"Job was {job['status']}.")

        rows = conn.execute(
            "SELECT result_json FROM job_table_results WHERE job_id = ? ORDER BY table_index ASC",
            (job_id,)
        ).fetchall()

    return [json.loads(row["result_json"]) for row in rows]


@app.get("/jobs", response_model=JobsListResponse, dependencies=[Depends(require_api_key)])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    with get_db() as conn:
        if status:
            cur = conn.execute("""
                SELECT id, status, completed_tables, total_tables, time_taken, priority, input_file_name, model_name, created_at
                FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (status, limit, offset))
        else:
            cur = conn.execute("""
                SELECT id, status, completed_tables, total_tables, time_taken, priority, input_file_name, model_name, created_at
                FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, offset))

        jobs = [dict(row) for row in cur.fetchall()]
        for j in jobs:
            j["priority"] = bool(j["priority"])
            j["created_at"] = format_timestamp(j["created_at"])
            j["progress"] = round((j["completed_tables"] / j["total_tables"]) * 100, 2) if j["total_tables"] > 0 else 0.0

    return {"count": len(jobs), "jobs": jobs}


@app.put("/jobs/{job_id}/priority", response_model=ActionResponse, dependencies=[Depends(require_api_key)])
async def update_job_priority(job_id: str, priority: bool):
    with get_db() as conn:
        cur = conn.execute("UPDATE jobs SET priority = ? WHERE id = ?", (1 if priority else 0, job_id))
        if cur.rowcount == 0: raise HTTPException(status_code=404, detail="Job ID not found.")
    return {"status": "success", "message": f"Job {job_id} priority updated."}


class JobAction(str, Enum):
    pause = "pause"
    resume = "resume"
    cancel = "cancel"


@app.put("/jobs/{job_id}/action", response_model=ActionResponse, dependencies=[Depends(require_api_key)])
async def manage_job_state(
    job_id: str,
    action: JobAction = Query(..., description="Select the action to perform on the job")
):
    action_str = action.value

    with get_db() as conn:
        status_check = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not status_check:
            raise HTTPException(status_code=404, detail="Job ID not found.")

        current_status = status_check[0]

        if action_str == "resume":
            if current_status not in ["paused", "cancelled"]:
                raise HTTPException(status_code=400, detail=f"Cannot resume a job that is {current_status}.")

            conn.execute("UPDATE jobs SET status = 'queued', error = NULL WHERE id = ?", (job_id,))
            return {"status": "success", "message": f"Job {job_id} resumed and placed back into the queue."}

        elif action_str in ["pause", "cancel"]:
            new_status = "paused" if action_str == "pause" else "cancelled"

            if current_status in ["completed", "failed", "cancelled"]:
                raise HTTPException(status_code=400, detail=f"Cannot {action_str} a job that is already {current_status}.")

            conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (new_status, job_id))

            for active_task in ACTIVE_TASKS.get(job_id, set()):
                active_task.cancel()

            return {"status": "success", "message": f"Job {job_id} {new_status} successfully."}


@app.delete("/jobs/{job_id}", response_model=ActionResponse, dependencies=[Depends(require_api_key)])
async def delete_job(job_id: str):
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status = 'cancelled' WHERE id = ? AND status = 'processing'", (job_id,))
        cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.execute("DELETE FROM job_table_results WHERE job_id = ?", (job_id,))

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job ID not found.")

    for active_task in ACTIVE_TASKS.get(job_id, set()):
        active_task.cancel()

    return {"status": "success", "message": f"Job {job_id} permanently deleted."}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    status: str
    role: str
    content: str


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat_with_model(request: ChatRequest):
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(LLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            ai_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return ChatResponse(status="success", role="assistant", content=ai_message)

    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
