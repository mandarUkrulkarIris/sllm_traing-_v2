import gc
import math
import os

# Must be set before CUDA is initialized (i.e. before `import torch`) to take effect.
# Reduces the fragmentation that caused the OOM (large reserved-but-unallocated blocks).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. SETUP & MODEL SELECTION
# ==========================================
MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_PATH = "/teamspace/uploads/dataset_v16v2_200726.clean.canonical.jsonl"
OUTPUT_MODEL_DIR = "./teamspace/uploads/Qwen3.5-4B_v16v2_clean_dataset_210726_v2"
CHECKPOINT_DIR = "./training_checkpoints_v2"

MAX_SEQ_LEN = 1024
EVAL_FRACTION = 0.05
SEED = 42

set_seed(SEED)

print("Loading Dataset...")
hf_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
hf_dataset = hf_dataset.shuffle(seed=SEED)

split_dataset = hf_dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
train_raw = split_dataset["train"]
eval_raw = split_dataset["test"]

print(f"Train examples: {len(train_raw):,} | Eval examples: {len(eval_raw):,}")

print("Initializing Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. JSON-ONLY MASKING (variable length — no fixed padding here)
# ==========================================
def tokenize_batch(examples):
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        completion_tokens = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        if len(completion_tokens) >= MAX_SEQ_LEN:
            # Completion alone doesn't fit. Keep its tail (with EOS) so the
            # training signal survives; drop the prompt rather than the label.
            completion_tokens = completion_tokens[-(MAX_SEQ_LEN - 1):]
            prompt_tokens = []
        elif len(prompt_tokens) + len(completion_tokens) > MAX_SEQ_LEN:
            # Truncate from the FRONT of the prompt so the completion (the
            # only part carrying a training signal) is never cut off.
            keep = MAX_SEQ_LEN - len(completion_tokens)
            prompt_tokens = prompt_tokens[-keep:]

        input_ids = prompt_tokens + completion_tokens
        labels = [-100] * len(prompt_tokens) + completion_tokens
        attention_mask = [1] * len(input_ids)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


print("Tokenizing Dataset with custom masking...")
tokenized_train = train_raw.map(tokenize_batch, batched=True, remove_columns=["prompt", "completion"])
tokenized_eval = eval_raw.map(tokenize_batch, batched=True, remove_columns=["prompt", "completion"])


class DynamicPaddingCollator:
    """Pads each batch to its own longest sequence instead of a fixed length."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


data_collator = DynamicPaddingCollator(tokenizer)

# ==========================================
# 3. LOAD MODEL TO GPU
# ==========================================
print(f"Loading {MODEL_ID} to GPU (bfloat16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

print("Applying Optimized LoRA...")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.enable_input_require_grads()
model.config.use_cache = False

model.print_trainable_parameters()

# ==========================================
# 4. METRICS (eval loss + token accuracy)
# ==========================================
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # Argmax immediately so full-vocab logits never get accumulated across the eval set.
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    correct = (predictions == labels) & mask
    denom = mask.sum()
    accuracy = (correct.sum() / denom).item() if denom > 0 else 0.0
    return {"token_accuracy": accuracy}


# ==========================================
# 5. TRAINING ARGS
# ==========================================
EFFECTIVE_BATCH = 2 * 24
steps_per_epoch = math.ceil(len(tokenized_train) / EFFECTIVE_BATCH)
eval_steps = max(10, steps_per_epoch // 3)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=24,

    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_torch",

    max_grad_norm=1.0,
    bf16=True,
    fp16=False,

    gradient_checkpointing=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,

    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,

    eval_strategy="steps",
    eval_steps=eval_steps,
    eval_accumulation_steps=8,

    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
    seed=SEED,
)

class MemoryEfficientTrainer(Trainer):
    """Clears the CUDA allocator cache around eval so leftover eval-phase
    reservations don't fragment the pool the next training step allocates into."""

    def evaluate(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        metrics = super().evaluate(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        return metrics


trainer = MemoryEfficientTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

last_checkpoint = None
if os.path.isdir(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

print("Starting High-Capacity JSON-Strict Training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. SAVE (best checkpoint, since load_best_model_at_end=True)
# ==========================================
print("Saving Final Model...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Done! Model saved to {OUTPUT_MODEL_DIR}")

import json

SAVE_DIR = OUTPUT_MODEL_DIR

training_args_json = training_args.to_dict()
with open(os.path.join(SAVE_DIR, "training_args.json"), "w") as f:
    json.dump(training_args_json, f, indent=4)
print("training_args.json saved")

trainer.state.save_to_json(os.path.join(SAVE_DIR, "trainer_state.json"))
print("trainer_state.json saved")

metrics = trainer.state.log_history
with open(os.path.join(SAVE_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print("training_metrics.json saved")

final_eval_metrics = trainer.evaluate()
with open(os.path.join(SAVE_DIR, "final_eval_metrics.json"), "w") as f:
    json.dump(final_eval_metrics, f, indent=4)
print("final_eval_metrics.json saved:", final_eval_metrics)

dataset_info = {
    "dataset_path": DATASET_PATH,
    "total_samples": len(hf_dataset),
    "train_samples": len(train_raw),
    "eval_samples": len(eval_raw),
    "eval_fraction": EVAL_FRACTION,
    "max_sequence_length": MAX_SEQ_LEN,
}
with open(os.path.join(SAVE_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=4)
print("dataset_info.json saved")

lora_info = {
    "r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "lora_dropout": lora_config.lora_dropout,
    "target_modules": str(lora_config.target_modules),
    "bias": lora_config.bias,
    "task_type": str(lora_config.task_type),
}
with open(os.path.join(SAVE_DIR, "lora_config.json"), "w") as f:
    json.dump(lora_info, f, indent=4)
print("lora_config.json saved")

env_info = {
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "device": str(next(model.parameters()).device),
    "dtype": str(next(model.parameters()).dtype),
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    env_info["gpu_name"] = torch.cuda.get_device_name(0)
with open(os.path.join(SAVE_DIR, "environment.json"), "w") as f:
    json.dump(env_info, f, indent=4)
print("environment.json saved")

print("\nAll training information has been saved successfully.")

import shutil

zip_path = OUTPUT_MODEL_DIR.rstrip("/") + ".zip"
print("Creating ZIP...")
shutil.make_archive(OUTPUT_MODEL_DIR.rstrip("/"), "zip", OUTPUT_MODEL_DIR)
print(f"ZIP saved to: {zip_path}")
print(f"ZIP Size: {os.path.getsize(zip_path)/1024/1024:.2f} MB")
