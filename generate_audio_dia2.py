import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import subprocess

try:
    from dia2 import Dia2, GenerationConfig, SamplingConfig
except Exception as e:
    print(f"Warning: dia2 module not found or broken. Exception: {e}")
    print("Attempting to automatically clone and install dia2 in editable mode...")
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    dia2_dir = os.path.join(parent_dir, "dia2")
    
    if not os.path.exists(dia2_dir):
        print(f"Cloning dia2 into {dia2_dir}...")
        subprocess.run(["git", "clone", "https://github.com/nari-labs/dia2.git", dia2_dir], check=True)
    else:
        print(f"Found existing dia2 directory at {dia2_dir}.")
        
    print("Installing dia2 in editable mode...")
    # Use the same python executable to run uv pip install
    subprocess.run(["uv", "pip", "install", "-e", dia2_dir], check=True)
    
    print("\n\n✅ dia2 has been successfully installed in editable mode!")
    print("Please click 'Generate Audio & Timestamps' again to restart the process.")
    sys.exit(0)
    print("Warning: dia2 module not found. Please ensure it is installed via pyproject.toml.")
    sys.exit(0)

def process_transcript(transcript, dia_model, config, output_audio_path, output_text_path):
    # Look for prefix files in the dia2 directory to stabilize voices
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    default_prefix1 = os.path.join(parent_dir, "dia2", "example_prefix1.wav")
    default_prefix2 = os.path.join(parent_dir, "dia2", "example_prefix2.wav")
    
    current_prefix1 = default_prefix1 if os.path.exists(default_prefix1) else None
    current_prefix2 = default_prefix2 if os.path.exists(default_prefix2) else None
        
    all_left_channels = []
    all_right_channels = []
    all_timestamps = []
    cumulative_time = 0.0
    sample_rate = 24000 # Default fallback
    
    for i, turn in enumerate(transcript):
        speaker = turn["speaker"] # "A" or "B"
        text = turn["text"]
        speaker_tag = "[S1]" if speaker == "A" else "[S2]"
        dia_script = f"{speaker_tag} {text}"
        
        print(f"  Turn {i+1}/{len(transcript)}: {dia_script[:50]}...")
        
        temp_wav = f"temp_turn_{i}.wav"
        
        kwargs = {"include_prefix": False}
        if current_prefix1:
            kwargs["prefix_speaker_1"] = current_prefix1
        if current_prefix2:
            kwargs["prefix_speaker_2"] = current_prefix2
            
        result = dia_model.generate(
            dia_script, 
            config=config, 
            output_wav=temp_wav, 
            verbose=False,
            **kwargs
        )
        
        # Load the generated turn audio
        audio_data, sample_rate = sf.read(temp_wav)
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)
        waveform = torch.from_numpy(audio_data).T.float()
        
        turn_duration = waveform.shape[1] / sample_rate
        
        # Update the prefix for the next turn (only if it's long enough to be a good prefix)
        if turn_duration > 1.5:
            if speaker == "A":
                current_prefix1 = temp_wav
            else:
                current_prefix2 = temp_wav
                
        # Process timestamps
        for j, (gen_word, start_time) in enumerate(result.timestamps):
            # Estimate end time
            if j + 1 < len(result.timestamps):
                end_time = result.timestamps[j+1][1]
                end_time = min(end_time, start_time + 1.0)
            else:
                end_time = start_time + 0.5
                
            speaker_label = "SPEAKER_MAIN" if speaker == "A" else "SPEAKER_USER"
            all_timestamps.append([
                gen_word,
                [round(cumulative_time + start_time, 3), round(cumulative_time + end_time, 3)],
                speaker_label
            ])
            
        # Add to channels
        left_channel = torch.zeros_like(waveform)
        right_channel = torch.zeros_like(waveform)
        
        if speaker == "A":
            left_channel = waveform
        else:
            right_channel = waveform
            
        all_left_channels.append(left_channel)
        all_right_channels.append(right_channel)
        
        cumulative_time += turn_duration
        
    # Combine all turns
    final_left = torch.cat(all_left_channels, dim=1)
    final_right = torch.cat(all_right_channels, dim=1)
    stereo_waveform = torch.cat([final_left, final_right], dim=0)
    
    # Save final outputs
    with open(output_text_path, "w") as f:
        json.dump({"alignments": all_timestamps}, f, indent=2)
        
    sf.write(output_audio_path, stereo_waveform.T.numpy(), sample_rate)
    
    # Cleanup temp files
    for i in range(len(transcript)):
        temp_wav = f"temp_turn_{i}.wav"
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
    return cumulative_time

def main():
    input_file = "data/custom_dataset/raw_transcripts.json"
    audio_dir = "data/custom_dataset/audio"
    dataset_jsonl = "data/custom_dataset/dataset.jsonl"
    
    os.makedirs(audio_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run generate_transcripts.py first.")
        return
        
    with open(input_file, "r") as f:
        transcripts = json.load(f)
        
    print(f"Loaded {len(transcripts)} transcripts.")
    
    # Extract valid UUIDs to clean up orphaned audio files later
    valid_ids = set()
    for t in transcripts:
        if isinstance(t, dict) and "id" in t:
            valid_ids.add(t["id"])
        else:
            # Fallback for old format
            pass
    
    try:
        print("Loading Dia2 Model...")
        dia_model = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")
        config = GenerationConfig(
            cfg_scale=3.0,
            audio=SamplingConfig(temperature=0.7, top_k=50),
            use_cuda_graph=True,
        )
    except NameError:
        print("Skipping model load (Dia2 not imported). Exiting.")
        return
        
    with open(dataset_jsonl, "w") as f_jsonl:
        for i, transcript_obj in enumerate(transcripts):
            if isinstance(transcript_obj, list):
                transcript_id = f"legacy_{i+1:03d}"
                transcript = transcript_obj
                valid_ids.add(transcript_id)
            else:
                transcript_id = transcript_obj["id"]
                transcript = transcript_obj["dialogue"]
                
            audio_path = os.path.join(audio_dir, f"{transcript_id}.wav")
            text_path = os.path.join(audio_dir, f"{transcript_id}.json")
            
            if os.path.exists(audio_path) and os.path.exists(text_path):
                print(f"Skipping {transcript_id} (Audio already exists)...")
                # Get duration from existing file
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
            else:
                print(f"Processing {transcript_id}...")
                duration = process_transcript(transcript, dia_model, config, audio_path, text_path)
            
            # Write to jsonl
            abs_audio_path = os.path.abspath(audio_path)
            f_jsonl.write(json.dumps({"path": abs_audio_path, "duration": duration}) + "\n")
        
    # Cleanup orphaned files (e.g., if a user deleted a row in the UI)
    print("\nCleaning up orphaned audio files...")
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav") or filename.endswith(".json"):
            file_id = os.path.splitext(filename)[0]
            if file_id not in valid_ids:
                file_path = os.path.join(audio_dir, filename)
                os.remove(file_path)
                print(f"Deleted orphaned file: {filename}")

    print("\nAll audio and timestamps generated successfully!")
    print(f"Dataset ready at {dataset_jsonl}")

if __name__ == "__main__":
    main()
