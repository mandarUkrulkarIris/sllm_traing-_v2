# SLLM Selection Criteria — Financial Table Analyst

**Task:** fine-tune a small language model (SLLM) to classify and structure financial tables (`table_type`, row/column typing, cell concept tagging) extracted from filings.
**Families evaluated:** Gemma and Llama (earlier trials), then Qwen3.5 (0.6B → 2B → 4B), LoRA fine-tuning, single-GPU budget (1× NVIDIA L4 for training, 1× NVIDIA T4 for serving).
**Outcome:** Qwen3.5-4B, LoRA r=32/alpha=64, quantized to GGUF Q8_0, deployed on the T4 instance.

---

## 1. Selection criteria

Models were judged against the following, in rough order of weight:

- **Schema-conformant output.** Can the model reliably emit the full nested JSON schema (`table_type`, per-row `row_type`/`direction`/`contributing_rows`, per-column `column_type`/`direction`, per-cell `concept_id`/`concept_meaning`/`unit`/`scale`) without malformed or truncated JSON, after LoRA fine-tuning on ~6,100 examples?
- **Multilinguality.** Source filings are not all English — the model has to read and correctly tag tables in whatever language the underlying document was filed in (labels, units, note references, free-text concepts) without a translation step in front of it. A base model with weak non-English pretraining coverage will show up as lower `concept_meaning`/`unit`/`table_type` accuracy specifically on non-English documents, even if its English-document accuracy looks fine — so this has to be checked per-language, not just in aggregate.
- **Structural vs. semantic accuracy.** Structural fields (`row_type`, `column_type`, `direction`) are closed/near-closed vocabulary and easier; free-text/open-vocabulary fields (`table_type` — 2,232 distinct labels, `concept_meaning`, `unit`) are the real differentiator between model sizes, and are also where multilingual weakness would surface first.
- **Trainability on a modest dataset.** 6,109 records, 3 epochs, single L4 GPU — the model needs to converge well under this budget, not require a much larger corpus or longer schedule.
- **Quantization tolerance.** The deployed model must survive GGUF quantization with minimal accuracy loss, since production runs on a cost-constrained T4 (§4 covers the Q6_K vs. Q8_0 comparison that decided the final format).
- **Inference cost/latency.** Per-table latency and $/table at the target deployment, since throughput and monthly GPU cost ($384/month on the T4) are fixed constraints.
- **License and ecosystem fit.** Open-weight, Apache-licensed, good `llama.cpp`/GGUF and PEFT/LoRA tooling support.

---

## 2. Why Qwen (vs. Gemma and Llama)

Before settling on the Qwen3.5 family, the same fine-tuning task was tried on **Gemma** and **Llama** base models. *[Exact variants/checkpoints and metrics from these earlier trials were not retained — qualitative recollection only; fill in with real model names/numbers if you still have them.]*

| Family | Outcome | Why it lost out |
|---|---|---|
| **Gemma** | Trained but did not reach usable schema/field accuracy for this task within the LoRA + dataset budget used. | Weaker fit for the nested-JSON, hybrid-attention-style LoRA targeting used here; less mature `llama.cpp`/GGUF quantization support at the ranks/sizes tried, which mattered for the Q6_K deployment target; pretraining is comparatively lighter on non-English coverage than Qwen, which matters directly for our multilingual filings. |
| **Llama** | Trained but did not reach usable schema/field accuracy for this task within the LoRA + dataset budget used. | Similar structural/semantic accuracy shortfall on the open-vocabulary fields (`table_type`, `concept_meaning`); tokenizer/context-length trade-offs were less favorable for the table-heavy prompt format used; multilingual coverage outside a handful of major languages is thinner than Qwen's, a direct liability given non-English source documents. |
| **Qwen3.5** | Selected for further sizing trials (§3). | Best schema-conformance and free-text field accuracy per parameter of the three families tried; Apache-licensed; strong `llama.cpp`/GGUF + PEFT/LoRA tooling support; broadest multilingual pretraining coverage of the three, which was a deciding factor given the source documents span multiple languages, not just English. |

**Note:** this table currently documents the *outcome*, not detailed comparative metrics — the Gemma/Llama runs predate this project's current reporting pipeline (`report_assets/analyze_eval.py` etc.), so no `compare/`-style breakdown exists for them the way it does for the Qwen3.5 trials below. In particular, no per-language accuracy breakdown was recorded for any of the three families — the multilingual comparison above reflects known base-model pretraining coverage plus qualitative observation on our documents, not a scored benchmark; a per-language slice of the `compare/` pipeline (splitting `eval_stats.json` by source-document language) would be the natural way to make this rigorous.

---

## 3. Qwen3.5 model size trials

| Model | LoRA config tried | Result | Decision |
|---|---|---|---|
| **Qwen3.5-0.6B** | r=16–32, standard `q/k/v/o_proj` targets | *[Exact metrics not retained from this trial — qualitative recollection only.]* Struggled to hold the full nested schema together; frequent malformed/incomplete JSON and weak `table_type`/`concept_meaning` accuracy even on training-set documents. | Rejected — too small to learn the task reliably. |
| **Qwen3.5-2B** (`Qwen3.5-2B_v2_160726`) | r=32, alpha=64, dropout=0.05, explicit per-layer target list (`q/k/v/o_proj`, `gate/up/down_proj`, plus hybrid-block `in_proj_qkv`/`in_proj_a`/`in_proj_b`/`in_proj_z`) | Adapter trained successfully but was not carried through to a full comparison-metrics evaluation before the 4B trial showed clearly better results — deprioritized rather than formally scored. | Superseded by 4B; not pursued further. |
| **Qwen3.5-4B** (`teamspace_uploads_Qwen3.5-4B_v16v2_clean_dataset_220726_3epochs`) | r=32, alpha=64, dropout=0.05, same hybrid-block target list as the 2B trial | 3 epochs / 384 steps, effective batch 48, ~14.7h on 1×L4. Evaluated on 543 tables across 61 job directories (Q6_K checkpoint): **`column_type` 98.1%**, **`row_type` 94.7%**, `contributing_columns`/`contributing_rows` 96.9–99.4%, `concept_id` 84.9% (lexical), `table_type` match 77–79%. Final deployment moved to Q8_0 (§4) for added accuracy at similar latency. | **Selected.** |

**Why size mattered:** the jump from 0.6B→2B→4B tracked directly with the model's ability to hold the nested schema *and* get the open-vocabulary fields (`table_type`, `concept_meaning`) right — the smaller model could often get structure (row/column typing) approximately right but broke down on exactly the free-text fields flagged as hardest in the evaluation report (§3 of `Evaluation_and_Metrics_Report.docx`).

---

## 4. Quantization format: Q6_K vs. Q8_0

The evaluation numbers in §3 were measured against the merged model quantized to **Q6_K** (3.23 GiB, 41% of the 7.85 GiB fp16 merged checkpoint) — the figures reported in `Evaluation_and_Metrics_Report.docx`. A **Q8_0** build of the same merged checkpoint was also produced (`merge_model.py` → `convert_hf_to_gguf.py` → `llama-quantize ... Q8_0`, per `code/05_quantize/original_quantize_serve_notes.md`) and compared against it on the T4 deployment.

**Result: Q8_0 gave similar inference latency to Q6_K but noticeably better accuracy**, so it was adopted as the final deployment format in place of Q6_K. *[Side-by-side latency/accuracy numbers for the Q6_K vs. Q8_0 comparison were not retained in the reporting pipeline — the finding above is a direct qualitative conclusion; re-run `analyze_eval.py`/`eval_chart_gen.py` against a Q8_0 `compare/` set if a formal head-to-head needs to be documented later.]* The final artifact, `merged_Qwen3.5-4B_v16v2_clean_dataset_210726_fp16-Q8_0.gguf`, is **4.17 GB** — larger than the 3.23 GiB Q6_K build but still 53% of the 7.85 GiB fp16 source — judged an acceptable trade for the accuracy gain given the T4 has headroom and latency did not regress.

---

## 5. LoRA adapter size / configuration experiments

Beyond model size, several LoRA configurations were tried on top of the 4B base before settling on a final recipe:

- Lower rank (r=8–16) with standard attention-only targets (`q/k/v/o_proj`): faster to train, but underfit the free-text fields relative to higher-rank adapters.
- r=32/alpha=64 with an **explicit, hand-enumerated target-module list** (needed because Qwen3.5 uses a hybrid attention block with named projections `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z` alongside the standard `q/k/v/o_proj`/`gate/up/down_proj`) — this is what produced the 4B production adapter and the evaluation numbers in §3 above (Q6_K); the same adapter carries over unchanged to the Q8_0 deployment in §4.
- r=32/alpha=64 with `target_modules="all-linear"` (letting PEFT auto-target every linear layer instead of hand-listing them) — final config, confirmed to match/exceed the hand-enumerated version while removing the need to track Qwen3.5's specific hybrid-block module names:

```
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

This is the recipe carried forward for future fine-tuning runs on this task.

---

## 6. Final decision

**Qwen3.5-4B**, LoRA-tuned with the `all-linear` r=32/alpha=64 configuration above and served as a **Q8_0** GGUF quantization, is the selected SLLM for the financial-table-analyst task. It won the family-level comparison against Gemma and Llama (§2) on schema-conformance, free-text field accuracy, and multilingual pretraining coverage — a hard requirement given the source filings span multiple languages, not just English; within the Qwen3.5 family it is the smallest model that met the accuracy bar on both structural and free-text fields (§3); and Q8_0 was chosen over the initially-deployed Q6_K build because it gave similar inference latency with added accuracy (§4), on a T4 that has the headroom to absorb the larger checkpoint. Its known weak points (`table_type` classification and free-text semantic fields, 67–79% accuracy under Q6_K) are documented and open-vocabulary/inherently harder rather than signs the model is undersized. Smaller Qwen3.5 sizes (0.6B, 2B) were tried first and rejected or superseded for the reasons in §3.
