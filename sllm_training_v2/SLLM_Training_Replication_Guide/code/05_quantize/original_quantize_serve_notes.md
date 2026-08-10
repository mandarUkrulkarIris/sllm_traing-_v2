Build image 
- podman build -t qwen-llama-cpu .



1. Re-convert the model (ignoring the MTP head):

podman run --rm -v ./models:/models qwen-llama-cpu python3 /app/convert_hf_to_gguf.py /models/v2/merged_Qwen3.5-4B_v16v2_clean_dataset_210726 --outfile /models/v2/merged_Qwen3.5-4B_v16v2_clean_dataset_210726_fp16.gguf --outtype f16 --no-mtp

2. Re-quantize the fixed file:

podman run --rm -v ./models:/models qwen-llama-cpu llama-quantize /models/v2/merged_Qwen3.5-4B_v16v2_clean_dataset_210726_fp16.gguf /models/v2/merged_Qwen3.5-4B_v16v2_clean_dataset_210726_fp16-Q8_0.gguf Q8_0 



Step 1: Download the Native Windows Binaries
Instead of compiling it yourself, the llama.cpp team provides heavily optimized binaries for Windows.

Go to the llama.cpp Releases page on GitHub.

Scroll down to the Assets section of the latest release.

Look for the Windows CPU zip file. It is usually named something like llama-bXXXX-bin-win-x64.zip or llama-bXXXX-bin-win-cpu-x64.zip.

Download and extract this ZIP file to a new folder (e.g., D:\Dev\llama_cpp).

(Note: If you run into a silent crash later, you may need to install the latest Visual C++ Redistributable from Microsoft, as the newest binaries require it).

Step 2: Organize Your Files
For simplicity, grab your qwen2b-q4_k_m.gguf file and move it directly into that newly extracted llama_cpp folder alongside the .exe files.

Step 3: Run the Local Server (or CLI)
Open PowerShell, navigate to your extracted folder, and you can run it immediately.

To run it as an API server (like we did in Podman):

.\llama-server.exe -m qwen2b-q4_k_m.gguf --port 8080 --threads 4 --ctx-size 8192