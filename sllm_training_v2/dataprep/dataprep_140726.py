import os
import json
import re

# ==========================================
# CONFIGURATION
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Since everything is in a flat directory, we only need one root directory path
DATASET_DIR = r"D:\Dev\git_repo\TFS\IntelligentDocumentParser\output"
DATASET_SAVE_PATH = os.path.join(ROOT_DIR, "dataset_v16v2_200726.jsonl")

SYSTEM_PROMPT = """You are a financial table structure analyst. Analyze the provided table JSON (row/column/cell matrix) and output JSON object only."""


def build_local_dataset():
    print(f"Scanning for files in '{DATASET_DIR}' and its subdirectories...")
    data = []
    
    # Use os.walk to search recursively through all subdirectories
    for root, dirs, files in os.walk(DATASET_DIR):
        # Filter out all input files in the current folder (root)
        input_files = [f for f in files if f.endswith('_input.json')]
        
        for input_filename in input_files:
            # Match the prefix (e.g. 'classify_tbl-1' from 'classify_tbl-1_input.json')
            prefix = input_filename.rsplit('_input.json', 1)[0]
            response_filename = f"{prefix}_response.json"
            
            # Form complete paths relative to the subdirectory they were found in
            input_path = os.path.join(root, input_filename)
            label_path = os.path.join(root, response_filename)
            
            # Check if the paired response file exists in the same subdirectory
            if response_filename not in files:
                print(f"[!] Missing response label for {input_path}. Skipping...")
                continue
                
            with open(input_path, 'r', encoding='utf-8') as f:
                inp_data = json.load(f)
            with open(label_path, 'r', encoding='utf-8') as f:
                outp_data = json.load(f)
                
            # (No key deletion occurs here anymore - the original outp_data structure is preserved)
            
            # Normalize into iterable list streams
            if isinstance(inp_data, dict):
                inp_data = [inp_data]
            if isinstance(outp_data, dict):
                outp_data = [outp_data]
                
            if isinstance(inp_data, list) and isinstance(outp_data, list):
                for single_input, single_output in zip(inp_data, outp_data):
                    inp_string = json.dumps(single_input, ensure_ascii=False)
                    
                    # Combine prompt block with optimized technical constraints
                    prompt = (
                        f"### Instruction:\n{SYSTEM_PROMPT}\n"
                        f"### Input:\n{inp_string}\n\n"
                        f"### Response:\n"
                    )
                    
                    # Force response to be single-line compact JSON (highly token efficient)
                    response = json.dumps(single_output, ensure_ascii=False)
                    
                    data.append({"prompt": prompt, "completion": response})

    # Save cleanly to JSONL
    print(f"Extracted {len(data)} examples. Saving...")
    with open(DATASET_SAVE_PATH, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Success! '{DATASET_SAVE_PATH}' is compiled and ready.")

if __name__ == "__main__":
    build_local_dataset()