import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
LORA_MODEL_PATH = r"D:\Dev\sllm_training_v2_gitrepo\sllm_traing-_v2\sllm_training_v2\adapters\Qwen3.5-9B_v16v2_clean_dataset_220726_v3"
MERGED_OUTPUT_DIR = r"D:\Dev\sllm_training_v2_gitrepo\sllm_traing-_v2\sllm_training_v2\adapters\Merged_Qwen3.5-9B_v16v2_clean_dataset_220726_v3"

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("1. Loading Base Model to CPU RAM... (This will take a few minutes)")
    # REMOVED device_map to stop the accelerate library from trying to chunk the model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,   # CHANGED to float16 to save 50% RAM
        low_cpu_mem_usage=True
    )

    print("2. Loading LoRA Adapters...")
    # ADDED offload_folder as a safety net just in case the OS spikes
    model = PeftModel.from_pretrained(
        base_model, 
        LORA_MODEL_PATH,
        offload_folder="temporary_offload_dir"
    )

    print("3. Merging Weights... (Your CPU will spike to 100% here. Please be patient!)")
    merged_model = model.merge_and_unload()

    print("4. Saving the Unified Model to disk...")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("5. Saving the Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
        print(f"\nSuccess! Your standalone model is now saved in: {MERGED_OUTPUT_DIR}")
    except Exception as e:
        print(f"\n[!] Failed to save tokenizer: {e}")

    # Clean up the temporary offload folder if it was created
    try:
        shutil.rmtree("temporary_offload_dir", ignore_errors=True)
    except:
        pass