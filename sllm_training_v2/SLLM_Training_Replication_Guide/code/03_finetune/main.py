import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. SETUP & MODEL SELECTION
# ==========================================
# If you want to use Llama-3.2-1B, change this to "meta-llama/Llama-3.2-1B-Instruct"
# (Requires huggingface-cli login)
MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_PATH = "/teamspace/uploads/dataset_v16v2_200726.clean.canonical.jsonl"
OUTPUT_MODEL_DIR = "./teamspace/uploads/Qwen3.5-4B_v16v2_clean_dataset_210726"
print("Loading Dataset...")

DATASET_FRACTION = 1

hf_dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

hf_dataset = hf_dataset.shuffle(seed=42)

total_size = len(hf_dataset)
subset_size = int(total_size * DATASET_FRACTION)

hf_dataset = hf_dataset.select(range(subset_size))

print(f"Using {subset_size:,} / {total_size:,} samples ({DATASET_FRACTION*100:.1f}%)")

print("Initializing Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

# Ensure the model has a padding token (crucial for Llama/Qwen)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. CUSTOM JSON-ONLY MASKING
# ==========================================
def tokenize_data(examples):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_tokens = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = prompt_tokens + completion_tokens

        # Mask the prompt so the 1.5B model focuses entirely on JSON rules
        labels = [-100] * len(prompt_tokens) + completion_tokens

        if len(input_ids) > 1024:
            input_ids = input_ids[:1024]
            labels = labels[:1024]

        pad_len = 1024 - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * (1024 - pad_len) + [0] * pad_len
        labels = labels + [-100] * pad_len

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs

print("Tokenizing Dataset with custom masking...")
tokenized_dataset = hf_dataset.map(tokenize_data, batched=True, remove_columns=["prompt", "completion"])

# ==========================================
# 3. LOAD MODEL TO GPU
# ==========================================
print(f"Loading {MODEL_ID} to GPU (float16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Applying Optimized LoRA...")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Required for gradient checkpointing
model.enable_input_require_grads()
model.config.use_cache = False

model.print_trainable_parameters()

# ==========================================
# 4. MEMORY-OPTIMIZED TRAINING ARGS
# ==========================================
training_args = TrainingArguments(
    output_dir="./training_checkpoints",

    ############################
    # Training
    ############################
    num_train_epochs=3,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=16,

    ############################
    # Optimizer
    ############################
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_torch",

    ############################
    # Stability
    ############################
    max_grad_norm=1.0,
    fp16=True,
    bf16=False,

    ############################
    # Memory
    ############################
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,

    ############################
    # Logging
    ############################
    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,

    ############################
    # Saving
    ############################
    save_strategy="epoch",
    save_total_limit=2,

    ############################
    # Misc
    ############################
    report_to="none",
    seed=42
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

import os

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs("./training_checkpoints", exist_ok=True)

print("Starting High-Capacity JSON-Strict Training...")
trainer.train()

# ==========================================
# 5. SAVE
# ==========================================
print("Saving Final Model...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Done! Model saved to {OUTPUT_MODEL_DIR}")

import os
import json
from transformers import trainer_utils

SAVE_DIR = OUTPUT_MODEL_DIR

# ==========================================
# 1. Save Training Arguments
# ==========================================
training_args_json = training_args.to_dict()

with open(os.path.join(SAVE_DIR, "training_args.json"), "w") as f:
    json.dump(training_args_json, f, indent=4)

print("✓ training_args.json saved")

# ==========================================
# 2. Save Trainer State
# ==========================================
trainer.state.save_to_json(
    os.path.join(SAVE_DIR, "trainer_state.json")
)

print("✓ trainer_state.json saved")

# ==========================================
# 3. Save Training Metrics
# ==========================================
metrics = trainer.state.log_history

with open(os.path.join(SAVE_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("✓ training_metrics.json saved")

# ==========================================
# 4. Save Dataset Information
# ==========================================
dataset_info = {
    "dataset_path": DATASET_PATH,
    "dataset_fraction": DATASET_FRACTION,
    "total_samples": total_size,
    "used_samples": subset_size,
    "max_sequence_length": 1024,
}

with open(os.path.join(SAVE_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=4)

print("✓ dataset_info.json saved")

# ==========================================
# 5. Save LoRA Configuration
# ==========================================
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

print("✓ lora_config.json saved")

# ==========================================
# 6. Save Environment Information
# ==========================================
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

print("✓ environment.json saved")

print("\nAll training information has been saved successfully.")

import shutil
import os

zip_path = "./teamspace/uploads/Qwen3.5-4B_v16v2_clean_dataset_210726.zip"

print("Creating ZIP...")

shutil.make_archive(
    "./teamspace/uploads/Qwen3.5-4B_v16v2_clean_dataset_210726",
    "zip",
    OUTPUT_MODEL_DIR
)

print(f"ZIP saved to: {zip_path}")
print(f"ZIP Size: {os.path.getsize(zip_path)/1024/1024:.2f} MB")