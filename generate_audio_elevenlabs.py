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


def delete_voice(client, voice_id):
    """Delete a voice from ElevenLabs and remove from cache."""
    try:
        client.voices.delete(voice_id)
        print(f"  Deleted ElevenLabs voice: {voice_id}")
        return True
    except Exception as e:
        print(f"  Warning: could not delete voice {voice_id}: {e}")
        return False

def cleanup_voices(client, used_voice_ids):
    """Delete all AutoVoice voices from ElevenLabs that were used in this run."""
    cache = load_voice_cache()
    # Find cache entries whose voice_id is in used_voice_ids
    keys_to_remove = [k for k, v in cache.items() if v in used_voice_ids]
    for key in keys_to_remove:
        voice_id = cache[key]
        if delete_voice(client, voice_id):
            del cache[key]
    save_voice_cache(cache)
    print(f"  Cleaned up {len(keys_to_remove)} voices from ElevenLabs.")


# Curated list of ElevenLabs public voice IDs with rough gender/style labels
# These are stable built-in voices that don't count against voice creation limits
_PUBLIC_VOICES_MALE = [
    "IKne3meq5aSn9XLyUdCD",  # Charlie
    "N2lVS1w4EtoT3dr4eOWO",  # Callum
    "ODq5zmih8GrVes37Dizd",  # Patrick
    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "VR6AewLTigWG4xSOukaG",  # Arnold
    "pNInz6obpgDQGcFmaJgB",  # Adam
    "yoZ06aMxZJJ28mfd3POQ",  # Sam
]
_PUBLIC_VOICES_FEMALE = [
    "9BWtsMINqrJLrRacOk9x",  # Aria
    "EXAVITQu4vr4xnSDxMaL",  # Bella
    "MF3mGyEYCl7XYWbV9V6O",  # Elli
    "ThT5KcBeYPX3keUQqHPh",  # Dorothy
    "XB0fDUnXU5powFXDhCwa",  # Charlotte
    "jBpfuIE2acCO8z3wKNLl",  # Gigi
    "oWAxZDx7w5VEj9dCyTzz",  # Grace
]

def pick_public_voice(voice_description):
    """Pick a random public ElevenLabs voice based on gender hint in description."""
    desc_lower = voice_description.lower()
    if any(w in desc_lower for w in ["female", "woman", "girl", "lady"]):
        pool = _PUBLIC_VOICES_FEMALE
    elif any(w in desc_lower for w in ["male", "man", "boy", "guy"]):
        pool = _PUBLIC_VOICES_MALE
    else:
        pool = _PUBLIC_VOICES_MALE + _PUBLIC_VOICES_FEMALE
    chosen = random.choice(pool)
    print(f"  Using public voice {chosen} for: {voice_description[:50]}")
    return chosen

def design_voice(client, voice_description):
    """Create or retrieve a cached ElevenLabs voice from a text description."""
    cache = load_voice_cache()
    key = hashlib.md5(voice_description.encode()).hexdigest()
    if key in cache:
        print(f"  Using cached voice for: {voice_description[:50]}...")
        return cache[key]

    print(f"  Designing new voice: {voice_description[:60]}...")
    previews = client.text_to_voice.design(
        voice_description=voice_description,
        auto_generate_text=True,
        model_id="eleven_ttv_v3",
        guidance_scale=5,
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
    
    stability_val = 0.5
    print(f"  Using stability={stability_val}")
    response = client.text_to_dialogue.convert_with_timestamps(
        inputs=inputs,
        model_id="eleven_v3",
        settings=ModelSettingsResponseModel(stability=stability_val)
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
    
    # Use voice_segments for exact speaker turn boundaries
    for seg in response.voice_segments:
        spk = "A" if seg.voice_id == voice_a else "B"
        s = int(seg.start_time_seconds * sample_rate)
        e = min(int(seg.end_time_seconds * sample_rate), waveform.shape[1])
        if spk == "A":
            left_channel[0, s:e] = waveform[0, s:e]
        else:
            right_channel[0, s:e] = waveform[0, s:e]
            
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
                # Speaker A is Agent (main), Speaker B is User
                agent_voice_prompt = transcript_obj.get("agent_voice_prompt", "A professional female customer service agent with a clear American accent.")
                user_voice_prompt = transcript_obj.get("user_voice_prompt", "A casual male voice, slightly deep.")
                voice_a = pick_public_voice(agent_voice_prompt)
                voice_b = pick_public_voice(user_voice_prompt)
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
