import os
import json
import sys
import base64
import hashlib
import torch
import numpy as np
import soundfile as sf
import io
from elevenlabs.client import ElevenLabs
from elevenlabs.types import DialogueInput, ModelSettingsResponseModel
import random

VOICE_CACHE_FILE = "data/custom_dataset/voice_cache.json"

def load_voice_cache():
    if os.path.exists(VOICE_CACHE_FILE):
        try:
            with open(VOICE_CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_voice_cache(cache):
    os.makedirs(os.path.dirname(VOICE_CACHE_FILE), exist_ok=True)
    with open(VOICE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def design_voice(client, voice_description):
    """Create or retrieve a cached ElevenLabs voice from a text description."""
    cache = load_voice_cache()
    key = hashlib.md5(voice_description.encode()).hexdigest()
    if key in cache:
        print(f"  Using cached voice for: {voice_description[:50]}...")
        return cache[key]

    print(f"  Designing new voice: {voice_description[:60]}...")
    previews = client.text_to_voice.create_previews(
        voice_description=voice_description,
        auto_generate_text=True,
    )
    preview = previews.previews[0]
    voice_name = f"AutoVoice_{key[:8]}"
    saved = client.text_to_voice.create(
        voice_name=voice_name,
        voice_description=voice_description,
        generated_voice_id=preview.generated_voice_id,
    )
    voice_id = saved.voice_id
    cache[key] = voice_id
    save_voice_cache(cache)
    print(f"  Created voice '{voice_name}' -> {voice_id}")
    return voice_id

def process_transcript(transcript, client, output_audio_path, output_text_path, voice_a, voice_b, system_prompt):
    inputs = []
    for turn in transcript:
        speaker = turn["speaker"]
        text = turn["text"]
        voice_id = voice_a if speaker == "A" else voice_b
        inputs.append(DialogueInput(text=text, voice_id=voice_id))
        
    print(f"Generating audio for {len(inputs)} turns via ElevenLabs...")
    
    response = client.text_to_dialogue.convert_with_timestamps(
        inputs=inputs,
        model_id="eleven_v3",
        settings=ModelSettingsResponseModel(stability=0.1)
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
    output_data = {"alignments": moshi_timestamps}
    if system_prompt:
        output_data["text_conditions"] = {"description": system_prompt}
        
    with open(output_text_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    # Create Stereo WAV (Left=A, Right=B)
    left_channel = torch.zeros_like(waveform)
    right_channel = torch.zeros_like(waveform)
    
    # Group words into speaker turns and copy turn-level segments
    # This avoids word-by-word channel switching artifacts
    if moshi_timestamps:
        current_speaker = "A" if moshi_timestamps[0][2] == "SPEAKER_MAIN" else "B"
        turn_start = moshi_timestamps[0][1][0]
        turn_end = moshi_timestamps[0][1][1]

        def flush_turn(spk, t_start, t_end):
            s = int(t_start * sample_rate)
            e = min(int(t_end * sample_rate), waveform.shape[1])
            if spk == "A":
                left_channel[0, s:e] = waveform[0, s:e]
            else:
                right_channel[0, s:e] = waveform[0, s:e]

        for item in moshi_timestamps[1:]:
            spk = "A" if item[2] == "SPEAKER_MAIN" else "B"
            if spk != current_speaker:
                flush_turn(current_speaker, turn_start, turn_end)
                current_speaker = spk
                turn_start = item[1][0]
            turn_end = item[1][1]
        flush_turn(current_speaker, turn_start, turn_end)
            
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
                print(f"Processing {transcript_id}...")
                # Design unique voices for each conversation based on prompts
                # Speaker A is User, Speaker B is Agent
                user_voice_prompt = transcript_obj.get("user_voice_prompt", "A casual male voice, slightly deep.")
                agent_voice_prompt = transcript_obj.get("agent_voice_prompt", "A professional female customer service agent with a clear American accent.")
                voice_a = design_voice(client, user_voice_prompt)
                voice_b = design_voice(client, agent_voice_prompt)
                system_prompt = transcript_obj.get("system_prompt", "") if isinstance(transcript_obj, dict) else ""
                duration = process_transcript(transcript, client, audio_path, text_path, voice_a, voice_b, system_prompt)
            
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
