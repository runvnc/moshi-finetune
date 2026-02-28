import os
import json
import sys
import base64
import torch
import numpy as np
import soundfile as sf
import io
from elevenlabs.client import ElevenLabs

def process_transcript(transcript, client, output_audio_path, output_text_path, voice_a, voice_b):
    inputs = []
    for turn in transcript:
        speaker = turn["speaker"]
        text = turn["text"]
        voice_id = voice_a if speaker == "A" else voice_b
        inputs.append({
            "text": text,
            "voice_id": voice_id
        })
        
    print(f"Generating audio for {len(inputs)} turns via ElevenLabs...")
    
    response = client.text_to_dialogue.convert_with_timestamps(
        inputs=inputs
    )
    
    # Decode audio
    audio_bytes = base64.b64decode(response.audio_base_64)
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=1)
    waveform = torch.from_numpy(audio_data).T.float()
    
    # Process timestamps
    alignment = response.alignment
    chars = alignment.characters
    starts = alignment.character_start_times_seconds
    ends = alignment.character_end_times_seconds
    
    # Map character index to speaker
    char_to_speaker = ["A"] * len(chars)
    for segment in response.voice_segments:
        speaker = "A" if segment.voice_id == voice_a else "B"
        for i in range(segment.character_start_index, segment.character_end_index + 1):
            if i < len(char_to_speaker):
                char_to_speaker[i] = speaker
                
    # Group characters into words
    moshi_timestamps = []
    current_word = ""
    current_start = None
    current_end = None
    current_speaker = None
    
    for i, (char, start, end) in enumerate(zip(chars, starts, ends)):
        speaker = char_to_speaker[i]
        
        if char.strip() == "":
            if current_word:
                speaker_label = "SPEAKER_MAIN" if current_speaker == "A" else "SPEAKER_USER"
                moshi_timestamps.append([
                    current_word,
                    [round(current_start, 3), round(current_end, 3)],
                    speaker_label
                ])
                current_word = ""
                current_start = None
        else:
            if not current_word:
                current_start = start
                current_speaker = speaker
            current_word += char
            current_end = end
            
    if current_word:
        speaker_label = "SPEAKER_MAIN" if current_speaker == "A" else "SPEAKER_USER"
        moshi_timestamps.append([
            current_word,
            [round(current_start, 3), round(current_end, 3)],
            speaker_label
        ])
        
    # Save Moshi JSON
    with open(output_text_path, "w") as f:
        json.dump({"alignments": moshi_timestamps}, f, indent=2)
        
    # Create Stereo WAV (Left=A, Right=B)
    left_channel = torch.zeros_like(waveform)
    right_channel = torch.zeros_like(waveform)
    
    current_speaker = None
    turn_start_time = 0.0
    
    for i, item in enumerate(moshi_timestamps):
        speaker = "A" if item[2] == "SPEAKER_MAIN" else "B"
        start_time = item[1][0]
        end_time = item[1][1]
        
        if speaker != current_speaker:
            current_speaker = speaker
            turn_start_time = start_time
            
        start_sample = int(start_time * sample_rate)
        end_sample = int((end_time + 0.2) * sample_rate) 
        end_sample = min(end_sample, waveform.shape[1])
        
        if speaker == "A":
            left_channel[0, start_sample:end_sample] = waveform[0, start_sample:end_sample]
        else:
            right_channel[0, start_sample:end_sample] = waveform[0, start_sample:end_sample]
            
    stereo_waveform = torch.cat([left_channel, right_channel], dim=0)
    sf.write(output_audio_path, stereo_waveform.T.numpy(), sample_rate)
    
    return waveform.shape[1] / sample_rate

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
    
    valid_ids = set()
    for t in transcripts:
        if isinstance(t, dict) and "id" in t:
            valid_ids.add(t["id"])
            
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set.")
        return
        
    client = ElevenLabs(api_key=api_key)
    
    # Default voices (can be overridden via env vars or UI later)
    voice_a = os.getenv("ELEVENLABS_VOICE_A", "9BWtsMINqrJLrRacOk9x") # Aria
    voice_b = os.getenv("ELEVENLABS_VOICE_B", "IKne3meq5aSn9XLyUdCD") # Charlie
    
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
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
            else:
                print(f"Processing {transcript_id}...")
                duration = process_transcript(transcript, client, audio_path, text_path, voice_a, voice_b)
            
            abs_audio_path = os.path.abspath(audio_path)
            f_jsonl.write(json.dumps({"path": abs_audio_path, "duration": duration}) + "\n")
        
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
