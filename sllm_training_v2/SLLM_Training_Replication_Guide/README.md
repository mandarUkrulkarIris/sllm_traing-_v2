# SLLM Training Replication Guide

How the Qwen3.5-4B "financial table analyst" small LLM was built, end to end:
raw document parsing -> labeled dataset -> LoRA fine-tune -> merge -> quantize
-> serve -> evaluate against GPT-4.1.

**Read `SLLM_Training_Pipeline_Guide.docx` first** - it explains every phase
in plain language with inputs, outputs, and the reasoning behind each step.
This README is the terse, operational companion: what's in this folder and
which command to run where.

## Folder map

```
SLLM_Training_Replication_Guide/
  README.md                          <- you are here
  SLLM_Training_Pipeline_Guide.docx  <- the full walkthrough (start here)
  guide_source.md                    <- markdown source of the .docx
  build_guide_docx.py                <- regenerates the .docx from the .md
  requirements.txt                   <- all phases' pip deps, grouped
  code/
    01_raw_export/     - DOCX -> financial-table JSON + GPT-4.1 labels
    02_dataset_build/  - JSON pairs -> cleaned prompt/completion JSONL
    03_finetune/       - LoRA fine-tuning on Qwen3.5-4B
    04_merge_adapters/ - merge LoRA weights into a standalone model
    05_quantize/       - convert to GGUF, quantize to Q8_0
    06_serve/          - FastAPI queue wrapper in front of llama-server
    07_evaluate/        - run held-out docs through the custom LLM & diff vs GPT-4.1
    08_reporting/      - aggregate metrics, generate charts, build the .docx reports
```

## Prerequisites per phase

| Phase | Runs where | Needs |
|---|---|---|
| 01 raw_export | Windows box with Word installed | The **IntelligentDocumentParser** repo (`app.*` package) - this script only runs from inside that repo; it is copied here for reference, not standalone. Azure OpenAI (GPT-4.1) credentials for `--classify`. |
| 02 dataset_build | Any machine, CPU only | Just Python stdlib. |
| 03 finetune | GPU VM (this project used a cloud Lightning/Teamspace GPU box) | `torch`, `transformers`, `datasets`, `peft`, `accelerate`. A CUDA GPU with enough VRAM for a 4B model + LoRA (bf16/fp16). |
| 04 merge_adapters | CPU machine with ~16GB+ RAM | Same `torch`/`transformers`/`peft` stack. |
| 05 quantize | CPU machine (Podman/Docker) | `llama.cpp` - either build via `Containerfile` or grab prebuilt Windows binaries. |
| 06 serve | GPU or CPU box | `llama-server` (from llama.cpp) running the `.gguf`, plus `fastapi`/`uvicorn`/`httpx`/`json_repair` for `api_GPU.py`. |
| 07 evaluate | Any machine that can reach `api_GPU.py` | `requests`, optional `sentence-transformers`, `matplotlib`. Also needs the **tablelayout_Generalized_v2** repo's `api/api_GPU.py` actually running. |
| 08 reporting | Any machine, CPU only | `matplotlib`, `numpy`, `python-docx`. |

## Running it, phase by phase

```bash
# 01 - export financial tables + GPT-4.1 labels (run FROM the IntelligentDocumentParser repo)
python scripts/export_table_classification_inputs_filterfin.py --input <docs_dir> --classify

# 02 - build -> clean -> (optional) canonicalize the training set
python code/02_dataset_build/dataprep_140726.py
python code/02_dataset_build/dataset_clean.py
python code/02_dataset_build/canonicalize_dataset.py     # optional, feeds main_v2.py
python code/02_dataset_build/analyze_dataset_labels.py    # optional QA report

# 03 - fine-tune (on the GPU box)
python code/03_finetune/main.py       # original run (fixed-length padding, fp16)
# or the improved variant with eval/early-stopping/dynamic padding:
python code/03_finetune/main_v2.py

# 04 - merge LoRA adapter into a standalone model
python code/04_merge_adapters/merge_model.py

# 05 - convert + quantize to GGUF (inside the llama.cpp Podman image)
podman build -t qwen-llama-cpu code/05_quantize/
podman run --rm -v <models_dir>:/models qwen-llama-cpu \
  python3 /app/convert_hf_to_gguf.py /models/<merged_model> --outfile /models/<name>_fp16.gguf --outtype f16 --no-mtp
podman run --rm -v <models_dir>:/models qwen-llama-cpu \
  llama-quantize /models/<name>_fp16.gguf /models/<name>_fp16-Q8_0.gguf Q8_0

# 06 - serve: llama-server first, then the FastAPI queue wrapper in front of it
llama-server.exe -m <name>_fp16-Q8_0.gguf --port 8080 --threads 4 --ctx-size 8192
MODEL_PATH=<name>_fp16-Q8_0.gguf python code/06_serve/api_GPU.py   # listens on :8000

# 07 - replay held-out docs through the custom LLM, then diff vs GPT-4.1
python code/07_evaluate/batch_generate_customllm_responses.py --input-dir <held_out_jobs_dir> --base-url http://127.0.0.1:8000
python code/07_evaluate/batch_compare_and_visualize.py --input-dir <held_out_jobs_dir>

# 08 - aggregate everything and generate the Word reports
python code/08_reporting/analyze_eval.py
python code/08_reporting/eval_chart_gen.py
python code/08_reporting/build_docx.py evaluation_report.md Evaluation_and_Metrics_Report.docx
```

## Important caveats

- **Phase 01 and 06 are not standalone here.** `export_table_classification_inputs_filterfin.py`
  imports from an `app` package that lives in the IntelligentDocumentParser
  repo, and `api_GPU.py` is the serving layer from the tablelayout_Generalized_v2
  repo. Both are copied into this folder purely for reference/reading - to
  actually run them you need those repos checked out.
- Every script has hardcoded paths/dates at the top (`MODEL_ID`, `DATASET_PATH`,
  `OUTPUT_MODEL_DIR`, `LORA_MODEL_PATH`, etc.) reflecting the specific run
  that produced the current production model. Edit those constants before
  each new run rather than treating them as configuration-free.
- `main.py` (fixed-length padding, no eval split) is what actually produced
  the LoRA adapter currently merged/quantized/deployed. `main_v2.py` is a
  later, better-engineered version (train/eval split, dynamic padding,
  early stopping) - prefer it for future runs, but its output has not yet
  gone through merge/quantize/eval itself.
- Finished evaluation reports already exist at the top of `sllm_training_v2/`
  (`Dataset_and_Training_Report.docx`, `Evaluation_and_Metrics_Report.docx`,
  `Unseen_Evaluation_and_Metrics_Report.docx`) - re-run phase 08 only if the
  underlying data changes.
