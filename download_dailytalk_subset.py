import os
import json
from huggingface_hub import hf_hub_download
from tqdm import tqdm

repo_id = "kyutai/DailyTalkContiguous"
repo_type = "dataset"
output_dir = "/files/moshi-finetune/data/dailytalk_subset"
audio_dir = os.path.join(output_dir, "data_stereo")
jsonl_path = os.path.join(output_dir, "dailytalk_subset.jsonl")

os.makedirs(audio_dir, exist_ok=True)

# Download the main jsonl file to get the list of files and durations
print("Downloading dailytalk.jsonl...")
main_jsonl = hf_hub_download(repo_id=repo_id, filename="dailytalk.jsonl", repo_type=repo_type)

# Read the first N entries
num_files_to_download = 200
entries = []
with open(main_jsonl, 'r') as f:
    for i, line in enumerate(f):
        if i >= num_files_to_download:
            break
        entries.append(json.loads(line))

print(f"Downloading {num_files_to_download} audio and json files...")
with open(jsonl_path, 'w') as f_out:
    for entry in tqdm(entries):
        # entry["path"] is like "data_stereo/0.wav"
        wav_filename = entry["path"]
        json_filename = wav_filename.replace(".wav", ".json")
        
        # Download wav
        local_wav = hf_hub_download(repo_id=repo_id, filename=wav_filename, repo_type=repo_type, local_dir=output_dir)
        # Download json
        local_json = hf_hub_download(repo_id=repo_id, filename=json_filename, repo_type=repo_type, local_dir=output_dir)
        
        # Write to our new jsonl with absolute path
        abs_wav_path = os.path.abspath(local_wav)
        f_out.write(json.dumps({"path": abs_wav_path, "duration": entry["duration"]}) + "\n")

print(f"\nDone! Subset saved to {output_dir}")
print(f"You can now use {jsonl_path} in your config.yaml")
