# Training a Small LLM for Financial Table Analysis (Qwen3.5-4B)

This guide walks through, phase by phase, how we turned raw financial DOCX
filings into a fine-tuned, quantized, self-hosted 4B-parameter model that
classifies and structures financial tables - and how we proved it works by
benchmarking it against GPT-4.1 on documents it never saw during training.

The short version: **parse documents -> label tables with a big commercial
LLM -> use those labels to fine-tune a small local LLM -> shrink and serve
the small LLM -> check that it still agrees with the big LLM on new data.**

Everything below maps to a numbered folder under `code/` in this same
directory, so you can open the actual script while reading the explanation.

---

## Phase 1 - Turning DOCX filings into labeled table JSON

**Script:** `code/01_raw_export/export_table_classification_inputs_filterfin.py`
**Input:** a folder of `.doc`/`.docx` financial filings
**Output:** `output/<job_id>/classify_<table_id>_input.json` / `_prompt.json` / `_response.json`

Every filing contains dozens of tables, and most of them are not financial
(tables of contents, glossaries, bullet lists dressed up as tables, etc).
Before spending money on GPT-4.1 calls or polluting the training set, this
script:

1. **Parses the DOCX** into a structured document (paragraphs, headings,
   tables) using the IntelligentDocumentParser app's own parsing stage.
2. **Filters to financial tables only**, using a deterministic scorer
   (`is_financial_table`) - no LLM call needed for this step. It fast-rejects
   obvious non-financial sections (table of contents, risk factors, glossary,
   management bios, ...), page-number tables, and bullet lists, then scores
   the rest on financial keywords (in 20+ languages), currency symbols/codes,
   the ratio of numeric cells, accounting-style negative numbers in
   parentheses, percentage signs, and year-like column headers. A table needs
   a score of 7+ to be kept.
3. **Builds a row/column/cell summary** for each surviving table (labels,
   sample values, which rows/columns still need classification) plus
   surrounding heading context, and turns that into the classification
   prompt (`classify_<id>_prompt.json`) and its structured input payload
   (`classify_<id>_input.json`).
4. **Sends each table to GPT-4.1 (Azure OpenAI)** when run with `--classify`,
   asking it to label:
   - `table_type` (e.g. "balance sheet", "income statement", ...)
   - per-row `row_type`, `direction`, sign, note references
   - per-column `column_type`
   - per-cell `reporting_period`, `concept_id`, `concept_meaning`, `unit`,
     `scale`, `scale_multiplier`

   The response is saved as `classify_<id>_response.json` right next to the
   input, so every table has a matching (input, label) pair on disk.
5. Writes a `job.json` manifest per document (table counts, whether
   classification ran).

Everything is keyed by `job_id` (one folder per document) and `table_id`
(`tbl-<n>` within that document), which is the join key every later phase
uses to line files up.

**Note:** this script only runs from inside the **IntelligentDocumentParser**
repo - it imports that app's parsing/classification/AI-client code directly.
The copy in this folder is for reading, not standalone execution.

---

## Phase 2 - Building and cleaning the training dataset

**Scripts:** `code/02_dataset_build/dataprep_140726.py`, `dataset_clean.py`,
`canonicalize_dataset.py`, `analyze_dataset_labels.py`

The raw `_input.json` / `_response.json` pairs from Phase 1 are not yet a
training set - they need to become `{"prompt": ..., "completion": ...}`
records, and the labels need to be cleaned up.

**2a. `dataprep_140726.py` - assemble the JSONL**
Walks every job folder, pairs each `classify_<id>_input.json` with its
`classify_<id>_response.json`, and wraps them in a simple instruction
template:

```
### Instruction:
You are a financial table structure analyst. Analyze the provided table
JSON (row/column/cell matrix) and output JSON object only.
### Input:
<the table JSON>

### Response:
```

The paired label becomes the `completion` (compact single-line JSON). Result:
one big `.jsonl` file, one line per table.

**2b. `dataset_clean.py` - fix labeling inconsistencies**
GPT-4.1 didn't always answer in exactly the same JSON shape. This step:
- Expands an abbreviated key schema (`tt`/`r`/`c`/`cs`, `v`/`cf`, ...) that
  ~100 records used, back to the full field names, so the model only ever
  sees one output format.
- Repairs a handful of records where cell data leaked out as stray
  top-level numeric keys instead of living inside `cells`.
- Drops genuinely broken records (null `table_type`, row/column counts that
  don't match the input).
- **Deduplicates**: a number of tables were accidentally labeled twice by
  two independent passes. Only the higher-confidence completion per unique
  prompt is kept (scored by mean confidence across all fields), so the
  model never sees "the same input, two different correct answers."
- Light whitespace/case cleanup on `table_type` strings only - it
  deliberately does **not** merge semantically-equivalent labels (e.g.
  "balance_sheet" vs "statement of financial position") since that's a
  taxonomy decision, not a bug fix.

A full report of what was changed/dropped is written to
`dataset_clean_report.txt`.

**2c. `canonicalize_dataset.py` - fold vocabulary variants (optional, feeds `main_v2.py`)**
A second, independent normalization pass over the *values* the labels use
(not the schema): folds case/whitespace/underscore variants of
`table_type`, `concept_id`, and `unit` down to their single most common
spelling, and applies a small manual alias map for known unit synonyms
(e.g. "percentage" -> "percent"). `concept_meaning` per `concept_id` is only
auto-collapsed if one wording already dominates (>=60% of occurrences) -
otherwise it's left alone and flagged as "ambiguous" in the report, since
guessing wrong there would silently corrupt a real distinction. Output is
a second, canonical JSONL used by the newer training script.

**2d. `analyze_dataset_labels.py` - QA, not required for training**
Produces distribution stats (label imbalance, near-duplicate labels,
concept_id/meaning consistency) purely for human review of dataset quality.

---

## Phase 3 - Fine-tuning Qwen3.5-4B with LoRA

**Scripts:** `code/03_finetune/main.py` (original run) and `main_v2.py` (improved)
**Input:** the cleaned JSONL from Phase 2
**Output:** a LoRA adapter directory (a few hundred MB, not a full model)

We use **PEFT / LoRA** rather than full fine-tuning: it trains a small set
of low-rank adapter weights instead of all 4B parameters, which is
dramatically cheaper in GPU memory and time while still adapting the
model's behavior on this narrow task.

Key choices, common to both scripts:

- **Base model:** `Qwen/Qwen3.5-4B`
- **LoRA config:** rank `r=32`, `alpha=64`, dropout `0.05`, applied to
  `all-linear` modules, no bias training - a fairly high-capacity adapter
  setup chosen because the output schema (nested JSON with many field
  types) is more complex than typical chat fine-tuning.
- **Loss masking:** every example is tokenized as `prompt + completion`, but
  the labels for the prompt tokens are set to `-100` (ignored by the loss).
  The model is only ever trained to predict the JSON completion, not to
  reproduce the input it was given.
- **3 epochs**, effective batch size 48 (`per_device_batch=3` x
  `gradient_accumulation=16`), learning rate `5e-5` with cosine decay and
  warmup, `adamw_torch` optimizer, gradient checkpointing to fit in memory.
- Runs on a cloud GPU box (Lightning/Teamspace-style), reading the dataset
  from and writing the adapter back to a shared `/teamspace/uploads/` path.

**`main.py`** (the run that actually produced the adapter currently
deployed, `teamspace_uploads_Qwen3.5-4B_v16v2_clean_dataset_210726`) trains
on the whole dataset with no held-out split, fixed 1024-token padding for
every example, and fp16.

**`main_v2.py`** is a later, more careful version of the same idea:
- Splits off 5% of the data as an eval set and tracks `eval_loss` +
  token-level accuracy during training.
- Uses **dynamic padding** (pad each batch only to its own longest example,
  rounded to a multiple of 8) instead of always padding to 1024 - faster
  and less wasteful.
- Truncates from the *front* of the prompt (never the completion) when an
  example is too long, so the training signal is never the part that gets
  cut.
- Adds early stopping (patience 4) and checkpoint-resume support.
- Trains on the *canonical* dataset from step 2c, and uses bf16 instead of
  fp16.

Both scripts finish by saving the adapter + tokenizer, and dumping
`training_args.json`, `trainer_state.json`, `training_metrics.json`,
`dataset_info.json`, `lora_config.json`, and `environment.json` alongside
it - so every adapter directory is self-documenting about exactly how it
was produced.

---

## Phase 4 - Merging the LoRA adapter into a standalone model

**Script:** `code/04_merge_adapters/merge_model.py`
**Input:** base model + LoRA adapter directory from Phase 3
**Output:** one standalone merged model directory (full-size, no PEFT needed to load it)

A LoRA adapter only contains the *delta* weights - to serve the model
efficiently (and to feed it into the GGUF/llama.cpp toolchain in Phase 5),
the adapter needs to be folded back into the base model's weights:

1. Load the base model (`Qwen/Qwen3.5-4B`) onto CPU in fp16, with
   `low_cpu_mem_usage=True` (loading a 4B model onto a GPU just to merge and
   throw the loader away is unnecessary; this keeps it to a CPU-RAM-bound
   step instead).
2. Load the LoRA adapter on top via `PeftModel.from_pretrained`.
3. Call `merge_and_unload()` - this bakes the adapter's low-rank deltas
   directly into the base model's weight matrices and returns a plain
   `AutoModelForCausalLM`, no PEFT wrapper needed anymore.
4. Save the merged model + tokenizer to disk as a normal Hugging Face model
   directory (e.g. `merged_Qwen3.5-4B_v16v2_clean_dataset_210726`).

This step is CPU- and RAM-heavy (expect the CPU to spike while the weight
matrices are combined) but doesn't need a GPU at all.

---

## Phase 5 - Shrinking the model: GGUF conversion & quantization

**Assets:** `code/05_quantize/Containerfile`, `original_quantize_serve_notes.md`
**Input:** the merged model directory from Phase 4
**Output:** `.gguf` model files at various precisions

The merged Hugging Face model is still in fp16/bf16 - great for a GPU
training box, impractical for cheap, always-on inference serving. We
convert it to **GGUF**, the format used by `llama.cpp`, which supports
efficient CPU/GPU inference and quantized weights:

1. **Build the toolchain:** the `Containerfile` builds an Ubuntu image that
   clones and compiles `llama.cpp` from source (so `convert_hf_to_gguf.py`
   and `llama-quantize` are both available).
2. **Convert to GGUF (fp16)** - `--no-mtp` skips Qwen's multi-token-prediction
   head, which the conversion script doesn't need for standard autoregressive
   serving:

```
python3 convert_hf_to_gguf.py <merged_model_dir> --outfile <name>_fp16.gguf --outtype f16 --no-mtp
```

3. **Quantize** - we produced both `Q6_K` (smaller, more lossy) and `Q8_0`
   (larger, closer to fp16 accuracy) variants:

```
llama-quantize <name>_fp16.gguf <name>_fp16-Q8_0.gguf Q8_0
```

**Q8_0 was chosen for production** - it's still roughly half the size of
fp16 while keeping accuracy loss minimal, which matters a lot for a model
whose whole job is emitting exact structured JSON.

Resulting artifact sizes for this project's model, for reference: fp16
~8.4 GB, Q6_K ~3.5 GB, Q8_0 ~4.5 GB.

---

## Phase 6 - Serving the quantized model

**Assets:** `code/06_serve/api_GPU.py`
**Input:** the `.gguf` file from Phase 5
**Output:** a running HTTP API that other scripts can send tables to

Two layers sit in front of the model:

1. **`llama-server`** (from `llama.cpp`, either the Podman build or the
   prebuilt Windows binaries) loads the `.gguf` file directly and exposes a
   raw completion API:

```
llama-server.exe -m <name>_fp16-Q8_0.gguf --port 8080 --threads 4 --ctx-size 8192
```

2. **`api_GPU.py`**, a small FastAPI app, sits in front of it on port 8000
   and adds the pieces `llama-server` doesn't provide on its own: an
   async job queue (`/predict`, `/status/{job_id}`, `/results/{job_id}`) so
   many tables can be submitted as one batch job and polled for progress,
   JSON repair on the model's raw text output (`json_repair`, since a small
   quantized model occasionally emits near-valid JSON), and a health check
   that confirms `llama-server` is actually reachable at startup.

This is the same shape of interface the raw-export pipeline in Phase 1
used against GPT-4.1 - which is what makes the apples-to-apples comparison
in Phase 7 possible: both models are handed the *exact same*
`classify_<id>_input.json` files and asked to produce the same structured
output.

---

## Phase 7 - Evaluating the small LLM against GPT-4.1

**Scripts:** `code/07_evaluate/generate_customllm_responses.py`,
`batch_generate_customllm_responses.py`, `compare_llm_responses.py`,
`batch_compare_and_visualize.py`, `visualize_comparison.py`, `visualize_table.py`

The whole point of this project is a small model that's "good enough" to
replace calling a commercial LLM for this task. To measure that:

1. **Replay the same inputs.** For a set of documents held out from
   training (never seen during Phase 3), the `classify_<id>_input.json`
   files already exist from Phase 1 (along with GPT-4.1's answers in
   `classify_<id>_response.json`). `generate_customllm_responses.py` takes
   those *same* input files, submits them to `api_GPU.py`, and saves the
   quantized model's answers as `classify_<id>_response_customllm.json`
   right next to GPT-4.1's - so both models' answers for the same table
   sit side by side. `batch_generate_customllm_responses.py` does this
   across every job folder in a directory, one job at a time (deliberately
   sequential, since they'd all compete for the same GPU slots anyway).

2. **Diff the two answer sets.** `compare_llm_responses.py` treats
   GPT-4.1's answer as the reference and the custom LLM's answer as the
   candidate, and compares field by field:
   - `table_type` - exact match, plus lexical/embedding similarity as a
     softer signal (via `sentence-transformers`, optional - falls back to
     lexical-only matching if it's not installed).
   - Rows - `row_type`, direction, sign, note references.
   - Columns - `column_type`.
   - Cells - `reporting_period` (with calendar-aware normalization, e.g.
     "FY2023" and "year ended Dec 31 2023" should match), `concept_id`,
     `concept_meaning`, `unit`, `scale`, `scale_multiplier`.

   Results are written as `compare/compare_<table_id>.json` (per table) and
   `compare/compare_summary.json` (per document) inside each job folder.

3. **Visualize.** `batch_compare_and_visualize.py` / `visualize_comparison.py`
   / `visualize_table.py` render those comparison files as charts and
   annotated table views, for spot-checking specific documents.

This produced two evaluation runs worth knowing about:
- A **mixed** run (in-sample + held-out documents together), to sanity
  check the model isn't wildly overfit.
- An **unseen-only** run, using only the held-out documents, as the
  honest measure of real-world generalization.

---

## Phase 8 - Turning results into reports

**Scripts:** `code/08_reporting/analyze.py`, `chart_gen.py`,
`train_chart_gen.py`, `analyze_eval.py`, `analyze_eval_unseen.py`,
`eval_chart_gen.py`, `eval_chart_gen_unseen.py`, `build_docx.py`

The `compare/` folders from Phase 7 are still hundreds of small per-table
JSON files - not something you'd hand to a stakeholder. The reporting
scripts:

1. **Aggregate** every job's `compare_summary.json` / `compare_<id>.json`
   into overall accuracy metrics, an in-sample-vs-held-out generalization
   check, best/worst-performing documents and tables, and a check for
   whether table size (row/column count) correlates with accuracy
   (`analyze_eval.py` for the mixed run, `analyze_eval_unseen.py` for the
   held-out-only run) - written out as `eval_stats.json`.
2. **Chart** those stats into PNGs (`eval_chart_gen.py` /
   `eval_chart_gen_unseen.py`), following one fixed convention: best results
   are always shown first / at the top, and every "where the model
   struggles" chart is paired with a "where it performs well" chart, never
   shown alone.
3. **Render to Word.** `build_docx.py <markdown_report> <output.docx>`
   parses a markdown report (headings, bullet points, tables, code blocks,
   embedded chart images) and lays it out as a formatted `.docx` - this is
   literally the same script used to generate the guide you're reading now.

The three finished reports this pipeline has produced so far live at the
top of `sllm_training_v2/`:

| Report | Covers |
|---|---|
| `Dataset_and_Training_Report.docx` | Training corpus stats + the LoRA training run (loss curve, LR schedule, compute) |
| `Evaluation_and_Metrics_Report.docx` | Quantized model vs GPT-4.1 on a mixed in-sample + held-out set, plus a deployment cost/capacity analysis |
| `Unseen_Evaluation_and_Metrics_Report.docx` | The same evaluation, restricted to only the held-out documents - the true generalization number |

---

## End-to-end artifact map

| Stage | Produces | Typical location on this machine |
|---|---|---|
| 1. Raw export | `classify_*_input/prompt/response.json`, `job.json` | `.../IntelligentDocumentParser/output/<job_id>/` |
| 2. Dataset build | `dataset_*.jsonl` (raw / clean / clean.canonical) | `D:\Dev\sllm_training_v2\` |
| 3. Fine-tune | LoRA adapter dir + training metadata JSONs | `teamspace_uploads_Qwen3.5-4B_...\` |
| 4. Merge | Standalone merged HF model dir | `D:\Dev\qwen_podman\models\v2\merged_...\` |
| 5. Quantize | `.gguf` files (fp16 / Q6_K / Q8_0) | `D:\Dev\qwen_podman\models\v2\*.gguf` |
| 6. Serve | HTTP endpoints on `:8080` (llama-server) and `:8000` (api_GPU.py) | wherever the server process runs |
| 7. Evaluate | `classify_*_response_customllm.json`, `compare/*.json` | alongside each job's other classify_* files |
| 8. Report | `eval_stats*.json`, chart PNGs, final `.docx` reports | `D:\Dev\sllm_training_v2\report_assets\` and repo root |
