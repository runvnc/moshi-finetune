import gradio as gr
import os
import json
import subprocess
import glob
import google.generativeai as genai
import pandas as pd
import uuid
import sys

CONFIG_FILE = "ui_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def load_transcripts_df():
    file_path = "data/custom_dataset/raw_transcripts.json"
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["ID", "Turns", "Preview", "Full JSON"])
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        rows = []
        for item in data:
            if isinstance(item, list):
                item_id = str(uuid.uuid4())
                dialogue = item
                full_obj = {"dialogue": dialogue}
            else:
                item_id = item.get("id", str(uuid.uuid4()))
                dialogue = item.get("dialogue", [])
                full_obj = {k: v for k, v in item.items() if k != "id"}
                
            turns = len(dialogue)
            preview = dialogue[0].get("text", "")[:50] + "..." if turns > 0 else "Empty"
            rows.append([item_id, turns, preview, json.dumps(full_obj)])
        return pd.DataFrame(rows, columns=["ID", "Turns", "Preview", "Full JSON"])
    except Exception as e:
        print(f"Error loading transcripts: {e}")
        return pd.DataFrame(columns=["ID", "Turns", "Preview", "Full JSON"])

def save_transcripts_df(df):
    file_path = "data/custom_dataset/raw_transcripts.json"
    try:
        data = []
        for row_id, json_str in zip(df["ID"], df["Full JSON"]):
            obj = json.loads(json_str)
            if isinstance(obj, list):
                obj = {"dialogue": obj}
            obj["id"] = row_id
            data.append(obj)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise gr.Error(f"Failed to save dataset: Invalid JSON format. {str(e)}")

def generate_transcripts(api_key, model_name, system_prompt, num_samples, num_turns):
    if not api_key:
        yield "Error: Please provide a Gemini API Key.", gr.update()
        return
    if not system_prompt:
        yield "Error: Please provide a system/conditioning prompt.", gr.update()
        return
        
    genai.configure(api_key=api_key)
    
    # Auto-wrap in <system> tags if not already present
    cleaned = system_prompt.strip()
    if not (cleaned.startswith("<system>") and cleaned.endswith("<system>")):
        system_prompt = f"<system> {cleaned} <system>"

    prompt = f"""
    You are an expert dialogue writer. Generate {num_samples} unique, short dialogue transcripts.
    
    The AI Agent (Speaker B) will be conditioned on a system/persona prompt during training.
    Here is the base system prompt to use as a reference:
    ---
    {system_prompt}
    ---
    For each conversation, generate a unique "system_prompt" field that is a variation of the above.
    If the base prompt contains scenario-specific information such as an agent name, company name, or
    other variable fields, vary those naturally across conversations (e.g. different agent names,
    different company names). If the base prompt is generic or contains no variable fields, you may
    use it as-is or with only minor natural phrasing variations. Either way, every conversation object
    MUST include a "system_prompt" field. The dialogue content must be consistent with the
    system_prompt used for that specific conversation (e.g. the agent introduces themselves with the
    correct name from their system_prompt).
    
    Format Requirements:
    - Speaker A is the User.
    - Speaker B is the AI Agent.
    - Follow the scenario description closely for who speaks first and the tone of the conversation.
    - IMPORTANT: You MUST inject ElevenLabs v3 emotion/audio tags into BOTH Speaker A and Speaker B lines to make them sound natural and human.
      Examples of valid tags: [laughs], [sighs], [angry], [cheerful], [whispering], [shouting], [sad], [excited], [nervous], [clears throat], [hmm], [gasp], [surprised], [confused], [hesitant].
      Use these tags inline for BOTH speakers, for example: "[hmm] Yeah, I think so..." or "[sighs] Let me check on that."
      Both speakers should have varied emotional delivery. Do not overdo it, but ensure both speakers sound expressive and human, not robotic.
    - Also include ambient/background audio event tags occasionally in both speakers' lines, such as [typing in background], [phone ringing in distance], [papers shuffling]. These add realism.
    - Use ellipses (...) for natural pauses and trailing thoughts, and dashes (--) for interruptions or hesitations. This improves pacing and realism.
    - Each conversation should be approximately {num_turns} turns long, including a natural back-and-forth exchange.
    - Output strictly as a JSON array of conversation objects.
    - Each conversation object MUST contain:
      1. "system_prompt": A variation of the base system prompt as described above, wrapped in <system> tags like: "<system> ... <system>"
      2. "agent_voice_prompt": A short description of the AI Agent's voice (e.g., "A professional female customer service agent with a clear American accent.")
      3. "user_voice_prompt": A short description of the User's voice (e.g., "A casual male voice, slightly deep.")
      4. "dialogue": An array of turn objects.
    - For agent_voice_prompt: ALWAYS include these exact elements: "clear call recording", "is a customer service representative", and "Occasional background sounds like keyboards and faint other calls."
      Vary the age, gender, accent, and personality traits (e.g. sweet, sarcastic, warm, no-nonsense, bubbly, tired, etc.).
      Example: "Clear call recording of an older woman with a thick Southern accent. She is a customer service representative. She is sweet and sarcastic. Occasional background sounds like keyboards and faint other calls."
    - For user_voice_prompt: ALWAYS include "clear call recording" and "Occasional background sounds like keyboards and faint other calls."
      Vary age, gender, accent, and mood (e.g. impatient, polite, confused, businesslike, etc.).
      Example: "Clear call recording of a middle-aged man with a mild Midwestern accent. Occasional background sounds like keyboards and faint other calls."
    
    Example Output:
    [
      {{
        "system_prompt": "<system> You are on an outgoing phone call. Your name: Jordan. Company name: TechVerify Inc. <system>",
        "agent_voice_prompt": "A cheerful young female voice with a light Southern US accent.",
        "user_voice_prompt": "An elderly male voice with a mild British accent.",
        "dialogue": [
          {{"speaker": "A", "text": "Hello?"}},
          {{"speaker": "B", "text": "[cheerful] Hi, how can I help you today?"}}
        ]
      }}
    ]
    
    Generate exactly {num_samples} conversations in this exact JSON format.
    """
    
    try:
        yield "Connecting to Gemini API...\n", gr.update()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
            ),
            stream=True
        )
        
        full_text = ""
        for chunk in response:
            full_text += chunk.text
            yield full_text, gr.update()
            
        # Clean up markdown formatting if present
        clean_text = full_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3].strip()
            
        new_data = json.loads(clean_text)
        
        # Wrap new data in UUID objects
        new_data_with_ids = []
        for convo in new_data:
            if isinstance(convo, list):
                new_data_with_ids.append({"id": str(uuid.uuid4()), "dialogue": convo})
            else:
                convo["id"] = str(uuid.uuid4())
                new_data_with_ids.append(convo)
        
        os.makedirs("data/custom_dataset", exist_ok=True)
        output_file = "data/custom_dataset/raw_transcripts.json"
        
        # Append to existing data
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
                
        combined_data = existing_data + new_data_with_ids
        
        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=2)
            
        yield full_text + f"\n\n\n✅ Successfully generated {len(new_data_with_ids)} new transcripts! Total dataset size: {len(combined_data)}.", load_transcripts_df()
        
    except Exception as e:
        yield f"Failed to generate transcripts: {str(e)}", gr.update()

def run_command(command, env=None):
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        
        output = ""
        for line in process.stdout:
            output += line
            yield output 
            
        process.wait()
        if process.returncode == 0:
            yield output + "\n\n✅ Process completed successfully!"
        else:
            yield output + f"\n\n❌ Process failed with exit code {process.returncode}"
            
    except Exception as e:
        yield f"Error executing command: {str(e)}"

def generate_audio():
    yield "Starting Dia2 Audio Generation...\n"
    for log in run_command("uv run generate_audio_dia2.py"):
        yield log

def download_dailytalk(num_samples):
    yield f"Downloading {num_samples} samples from DailyTalk...\n"
    
    script = f"""import os
import json
from huggingface_hub import hf_hub_download
from tqdm import tqdm

repo_id = 'kyutai/DailyTalkContiguous'
repo_type = 'dataset'
output_dir = 'data/dailytalk_subset'
audio_dir = os.path.join(output_dir, 'data_stereo')
jsonl_path = os.path.join(output_dir, 'dailytalk_subset.jsonl')

os.makedirs(audio_dir, exist_ok=True)

print('Downloading dailytalk.jsonl...')
main_jsonl = hf_hub_download(repo_id=repo_id, filename='dailytalk.jsonl', repo_type=repo_type)

entries = []
with open(main_jsonl, 'r') as f:
    for i, line in enumerate(f):
        if i >= {num_samples}:
            break
        entries.append(json.loads(line))

print(f'Downloading {num_samples} audio and json files...')
with open(jsonl_path, 'w') as f_out:
    for entry in tqdm(entries):
        wav_filename = entry['path']
        json_filename = wav_filename.replace('.wav', '.json')
        local_wav = hf_hub_download(repo_id=repo_id, filename=wav_filename, repo_type=repo_type, local_dir=output_dir)
        local_json = hf_hub_download(repo_id=repo_id, filename=json_filename, repo_type=repo_type, local_dir=output_dir)
        abs_wav_path = os.path.abspath(local_wav)
        f_out.write(json.dumps({{'path': abs_wav_path, 'duration': entry['duration']}}) + '\n')

print(f'\nDone! Subset saved to {{output_dir}}')
"""
    with open("download_dailytalk_temp.py", "w") as f:
        f.write(script)
        
    for log in run_command("uv pip install huggingface_hub tqdm && uv run download_dailytalk_temp.py"):
        yield log

def run_training(base_model, hf_token, max_steps, batch_size, learning_rate, mix_dailytalk, lora_rank, lora_scaling, duration_sec):
    import subprocess, sys, importlib
    yield "Installing personaplex moshi...\n"
    result = subprocess.run(["pip3", "install", "-e", "/workspace/personaplex/moshi/", "-q"], capture_output=True, text=True)
    if result.returncode != 0:
        yield f"ERROR: Failed to install personaplex moshi:\n{result.stderr}\nAborting training.\n"
        return
    # Verify by asking pip3 show where moshi is installed
    check = subprocess.run(["pip3", "show", "moshi"], capture_output=True, text=True)
    moshi_location = ""
    for line in check.stdout.splitlines():
        if line.startswith("Location:"):
            moshi_location = line.split(":", 1)[1].strip()
    if "personaplex" not in moshi_location:
        yield f"ERROR: moshi is installed at {moshi_location} (not personaplex).\nAborting training.\n"
        return
    yield f"OK: Personaplex moshi confirmed at: {moshi_location}\n"
    yield "Starting LoRA Fine-Tuning...\n"
    
    # Clear existing run dir to avoid conflict
    import shutil
    run_dir = "output/custom_model"
    if os.path.exists(run_dir):
        yield f"Removing existing run dir: {run_dir}\n"
        shutil.rmtree(run_dir)

    train_data = "'data/custom_dataset/dataset.jsonl'"
    if mix_dailytalk:
        train_data = "'data/custom_dataset/dataset.jsonl:0.7,data/dailytalk_subset/dailytalk_subset.jsonl:0.3'"
    
    # Update config.yaml
    config_content = f"""# data
data:
  eval_data: ''
  shuffle: true
  train_data: {train_data}

# model
moshi_paths: 
  hf_repo_id: "{base_model}"

full_finetuning: false
lora:
  enable: true
  rank: {lora_rank}
  scaling: {lora_scaling}
  ft_embed: false

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

# optim
duration_sec: {duration_sec}
batch_size: {batch_size}
max_steps: {max_steps}
gradient_checkpointing: true
optim:
  lr: {learning_rate}
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 0
log_freq: 1
eval_freq: 100
do_eval: false
do_ckpt: true
ckpt_freq: 100

save_adapters: true

run_dir: "output/custom_model"
"""
    with open("config_custom.yaml", "w") as f:
        f.write(config_content)
    
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    cmd = "pip3 install -e . -q && torchrun --nproc-per-node 1 -m train config_custom.yaml"
    for log in run_command(cmd, env=env):
        yield log

def export_model():
    yield "Finding latest LoRA checkpoint...\n"
    
    checkpoints = glob.glob(os.path.join(os.getcwd(), "output/custom_model/checkpoints/checkpoint_*"))
    
    valid_checkpoints = []
    for c in checkpoints:
        try:
            step = int(c.split("_")[-1])
            valid_checkpoints.append((step, c))
        except ValueError:
            pass

    if not valid_checkpoints:
        yield f"❌ Error: No training checkpoints found in {os.path.abspath('output/custom_model/checkpoints/')}"
        return
        
    latest_checkpoint = sorted(valid_checkpoints)[-1][1]
    step_name = os.path.basename(latest_checkpoint)
    lora_path = os.path.join(latest_checkpoint, "consolidated", "lora.safetensors")
    adapter_dir = os.path.join(latest_checkpoint, "consolidated")
    
    yield f"Found latest checkpoint: {step_name}\n"
    yield "\nWith LoRA, no complex export is needed! The adapters are already saved as safetensors.\n"
    yield f"\n📁 Checkpoint directory:\n  {os.path.abspath(latest_checkpoint)}\n"
    yield f"\n📁 Adapter directory:\n  {os.path.abspath(adapter_dir)}\n"
    if os.path.exists(lora_path):
        size_mb = os.path.getsize(lora_path) / (1024 * 1024)
        yield f"\n🎉 ALL DONE! LoRA adapter ({size_mb:.0f} MB):\n  {os.path.abspath(lora_path)}\n"
    else:
        yield f"\n⚠️ lora.safetensors not found at expected path:\n  {os.path.abspath(lora_path)}\n"
        yield f"  Files in adapter dir:\n"
        if os.path.exists(adapter_dir):
            for f in os.listdir(adapter_dir):
                yield f"    {f}\n"
    lora_abs = os.path.abspath(lora_path)
    yield "\n\n--- Deploy with PersonaPlex server (supports voice cloning) ---"
    yield f"\n  cd /path/to/personaplex"
    yield f"\n  git pull"
    yield f"\n  pip install -e moshi/"
    yield f"\n  python -m moshi.server \\"
    yield f"\n    --hf-repo nvidia/personaplex-7b-v1 \\"
    yield f"\n    --lora-weight {lora_abs} \\"
    yield f"\n    --lora-rank 32 --lora-scaling 2.0 \\"
    yield f"\n    --host 0.0.0.0 --port 8998"

# --- Gradio UI Layout ---
config = load_config()

with gr.Blocks(title="Moshi Fine-Tuning Studio") as app:
    gr.Markdown("# 🎛️ Moshi / PersonaPlex Fine-Tuning Studio")
    gr.Markdown("Generate synthetic dialogue data and fine-tune your audio language model end-to-end.")
    
    with gr.Tab("1. Dataset Generation & Management"):
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="Gemini API Key", type="password", value=config.get("api_key", ""))
                model_name = gr.Dropdown(choices=["gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-1.5-pro"], value=config.get("model_name", "gemini-3-flash-preview"), label="Gemini Model")
                system_prompt = gr.Textbox(
                    label="System Prompt / Conditioning Prompt", 
                    lines=8, 
                    value=config.get("system_prompt", "You are a helpful assistant."),
                    info="Dialogues will be generated aligned to this. Can optionally contain example fields like agent or company name."
                )
                num_samples = gr.Slider(minimum=1, maximum=500, value=config.get("num_samples", 50), step=1, label="Number of Conversations")
                num_turns = gr.Slider(minimum=2, maximum=20, value=config.get("num_turns", 6), step=1, label="Target Turns per Conversation")
                gen_text_btn = gr.Button("Generate Transcripts", variant="primary")
            
            with gr.Column():
                text_output = gr.Textbox(label="Live Generation Stream", lines=15)
                
        gr.Markdown("### 📂 Current Dataset Manager")
        gr.Markdown("This table shows all conversations currently saved in your dataset. You can edit or delete rows here, and it will automatically save to `raw_transcripts.json`.")
        dataset_df = gr.Dataframe(value=load_transcripts_df, headers=["ID", "Turns", "Preview", "Full JSON"], interactive=True, wrap=True)

        api_key.change(lambda x: save_config("api_key", x), inputs=[api_key])
        model_name.change(lambda x: save_config("model_name", x), inputs=[model_name])
        system_prompt.change(lambda x: save_config("system_prompt", x), inputs=[system_prompt])
        num_samples.change(lambda x: save_config("num_samples", x), inputs=[num_samples])
        num_turns.change(lambda x: save_config("num_turns", x), inputs=[num_turns])

        gen_text_btn.click(generate_transcripts, inputs=[api_key, model_name, system_prompt, num_samples, num_turns], outputs=[text_output, dataset_df])
        dataset_df.change(save_transcripts_df, inputs=[dataset_df])

    with gr.Tab("2. Generate Audio"):
        gr.Markdown("Generate audio and timestamps based on the transcripts generated in Step 1.")
        
        with gr.Row():
            audio_engine = gr.Radio(choices=["Dia2 (Local)", "ElevenLabs (API)"], value=config.get("audio_engine", "Dia2 (Local)"), label="Audio Engine")
            elevenlabs_api_key = gr.Textbox(label="ElevenLabs API Key", type="password", value=config.get("elevenlabs_api_key", ""), visible=config.get("audio_engine", "Dia2 (Local)") == "ElevenLabs (API)")
            
        gen_audio_btn = gr.Button("Generate Audio & Timestamps", variant="primary")
        audio_output = gr.Textbox(label="Terminal Output", lines=15)
        
        def update_audio_engine_ui(engine):
            save_config("audio_engine", engine)
            is_el = engine == "ElevenLabs (API)"
            return gr.update(visible=is_el)
            
        audio_engine.change(update_audio_engine_ui, inputs=[audio_engine], outputs=[elevenlabs_api_key])
        elevenlabs_api_key.blur(lambda x: save_config("elevenlabs_api_key", x), inputs=[elevenlabs_api_key])
        
        def generate_audio_wrapper(engine, el_api_key):
            yield f"Starting Audio Generation using {engine}...\n"
            if engine == "Dia2 (Local)":
                for log in run_command(f"{sys.executable} -u generate_audio_dia2.py"):
                    yield log
            else:
                if not el_api_key:
                    yield "Error: ElevenLabs API Key is required.\n"
                    return
                env = os.environ.copy()
                env["ELEVENLABS_API_KEY"] = el_api_key
                for log in run_command(f"{sys.executable} -u generate_audio_elevenlabs.py", env=env):
                    yield log
                    
        gen_audio_btn.click(generate_audio_wrapper, inputs=[audio_engine, elevenlabs_api_key], outputs=audio_output)

    with gr.Tab("3. Download DailyTalk (Optional)"):
        gr.Markdown("Download a subset of the DailyTalk dataset to mix with your synthetic data. This prevents catastrophic forgetting and keeps the voice sounding natural.")
        with gr.Row():
            with gr.Column():
                dt_samples = gr.Slider(minimum=50, maximum=2000, value=config.get("dt_samples", 200), step=50, label="Number of DailyTalk Samples to Download")
                download_dt_btn = gr.Button("Download DailyTalk Subset", variant="secondary")
            with gr.Column():
                dt_output = gr.Textbox(label="Download Logs", lines=10)
                
        dt_samples.change(lambda x: save_config("dt_samples", x), inputs=[dt_samples])
        download_dt_btn.click(download_dailytalk, inputs=[dt_samples], outputs=dt_output)

    with gr.Tab("4. Train Model (LoRA)"):
        with gr.Row():
            with gr.Column():
                base_model = gr.Dropdown(choices=["nvidia/personaplex-7b-v1"], value="nvidia/personaplex-7b-v1", label="Base Model", info="Base model for fine-tuning.")
                hf_token = gr.Textbox(label="HuggingFace Token (HF_TOKEN)", type="password", value=config.get("hf_token", ""), info="Required for gated models like nvidia/personaplex-7b-v1. Get yours at huggingface.co/settings/tokens")
                max_steps = gr.Slider(minimum=10, maximum=2000, value=config.get("max_steps", 50), step=10, label="Max Training Steps", info="For 100 examples, 50 steps is usually enough.")
                batch_size = gr.Slider(minimum=4, maximum=64, value=config.get("batch_size", 16), step=4, label="Batch Size", info="Increase to 32 or 48 if you have an H200/A100. Keep at 8-16 for 24GB GPUs.")
                lr = gr.Textbox(label="Learning Rate", value=config.get("lr", "2e-6"), info="Default is 2e-6.")
                mix_dailytalk = gr.Checkbox(label="Mix with DailyTalk Subset (70/30 split)", value=config.get("mix_dailytalk", True), info="Requires downloading the subset in Step 3 first.")
                
                with gr.Accordion("Advanced LoRA & Memory Parameters", open=False):
                    lora_rank = gr.Slider(minimum=8, maximum=256, value=config.get("lora_rank", 32), step=8, label="LoRA Rank", info="Lower rank (e.g., 32) uses less memory and prevents overfitting on small datasets.")
                    lora_scaling = gr.Slider(minimum=0.5, maximum=4.0, value=config.get("lora_scaling", 2.0), step=0.5, label="LoRA Scaling (Alpha)", info="Multiplier for the LoRA weights.")
                    duration_sec = gr.Slider(minimum=10, maximum=100, value=config.get("duration_sec", 100), step=10, label="Max Audio Duration (sec)", info="Reduce to 30-50 if you run out of VRAM on 24GB GPUs.")

                train_btn = gr.Button("Start Fine-Tuning", variant="primary")
            
            with gr.Column():
                train_output = gr.Textbox(label="Training Logs", lines=20)
                
        base_model.change(lambda x: save_config("base_model", x), inputs=[base_model])
        max_steps.change(lambda x: save_config("max_steps", x), inputs=[max_steps])
        batch_size.change(lambda x: save_config("batch_size", x), inputs=[batch_size])
        lr.change(lambda x: save_config("lr", x), inputs=[lr])
        mix_dailytalk.change(lambda x: save_config("mix_dailytalk", x), inputs=[mix_dailytalk])
        lora_rank.change(lambda x: save_config("lora_rank", x), inputs=[lora_rank])
        lora_scaling.change(lambda x: save_config("lora_scaling", x), inputs=[lora_scaling])
        duration_sec.change(lambda x: save_config("duration_sec", x), inputs=[duration_sec])

        train_btn.click(run_training, inputs=[base_model, hf_token, max_steps, batch_size, lr, mix_dailytalk, lora_rank, lora_scaling, duration_sec], outputs=train_output)

    with gr.Tab("5. Export Model"):
        gr.Markdown("**Final Step:** Locate the final LoRA adapter weights for deployment.")
        export_btn = gr.Button("Export Final Model", variant="primary")
        export_output = gr.Textbox(label="Export Logs", lines=15)
        
        export_btn.click(export_model, outputs=export_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
