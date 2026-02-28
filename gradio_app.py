import gradio as gr
import os
import json
import subprocess
import glob
import google.generativeai as genai
import pandas as pd
import uuid

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
            else:
                item_id = item.get("id", str(uuid.uuid4()))
                dialogue = item.get("dialogue", [])
                
            turns = len(dialogue)
            preview = dialogue[0].get("text", "")[:50] + "..." if turns > 0 else "Empty"
            rows.append([item_id, turns, preview, json.dumps(dialogue)])
        return pd.DataFrame(rows, columns=["ID", "Turns", "Preview", "Full JSON"])
    except Exception as e:
        print(f"Error loading transcripts: {e}")
        return pd.DataFrame(columns=["ID", "Turns", "Preview", "Full JSON"])

def save_transcripts_df(df):
    file_path = "data/custom_dataset/raw_transcripts.json"
    try:
        data = [{"id": row_id, "dialogue": json.loads(dialogue_str)} for row_id, dialogue_str in zip(df["ID"], df["Full JSON"])]
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise gr.Error(f"Failed to save dataset: Invalid JSON format. {str(e)}")

def generate_transcripts(api_key, model_name, scenario, num_samples, num_turns):
    if not api_key:
        yield "Error: Please provide a Gemini API Key.", gr.update()
        return
    if not scenario:
        yield "Error: Please provide a scenario description.", gr.update()
        return
        
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are an expert dialogue writer. Generate {num_samples} unique, short dialogue transcripts.
    
    Scenario:
    {scenario}
    
    Format Requirements:
    - Speaker A is the User.
    - Speaker B is the AI Agent.
    - Follow the scenario description closely for who speaks first and the tone of the conversation.
    - Each conversation should be approximately {num_turns} turns long, including a natural back-and-forth exchange.
    - Output strictly as a JSON array of conversations. Each conversation is an array of turn objects.
    
    Example Output:
    [
      [
        {{"speaker": "A", "text": "Hello?"}},
        {{"speaker": "B", "text": "Hi, how can I help you today?"}}
      ]
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
        new_data_with_ids = [{"id": str(uuid.uuid4()), "dialogue": convo} for convo in new_data]
        
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
            env=env
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
output_dir = '/files/moshi-finetune/data/dailytalk_subset'
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

def run_training(base_model, max_steps, batch_size, learning_rate, mix_dailytalk, lora_rank, lora_scaling, duration_sec):
    yield "Starting LoRA Fine-Tuning...\n"
    
    train_data = "'/files/moshi-finetune/data/custom_dataset/dataset.jsonl'"
    if mix_dailytalk:
        train_data = "'/files/moshi-finetune/data/custom_dataset/dataset.jsonl:0.7,/files/moshi-finetune/data/dailytalk_subset/dailytalk_subset.jsonl:0.3'"
    
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

run_dir: "/files/moshi-finetune/output/custom_model"
"""
    with open("config_custom.yaml", "w") as f:
        f.write(config_content)
    
    cmd = "uv pip install -e . && uv run torchrun --nproc-per-node 1 -m train config_custom.yaml"
    for log in run_command(cmd):
        yield log

def export_model():
    yield "Finding latest LoRA checkpoint...\n"
    
    checkpoints = glob.glob("output/custom_model/checkpoints/checkpoint_*")
    
    valid_checkpoints = []
    for c in checkpoints:
        try:
            step = int(c.split("_")[-1])
            valid_checkpoints.append((step, c))
        except ValueError:
            pass

    if not valid_checkpoints:
        yield "❌ Error: No training checkpoints found in output/custom_model/"
        return
        
    latest_checkpoint = sorted(valid_checkpoints)[-1][1]
    step_name = os.path.basename(latest_checkpoint)
    
    yield f"Found latest checkpoint: {step_name}\n"
    yield "\nWith LoRA, no complex export is needed! The adapters are already saved as safetensors.\n"
    yield f"\n\n🎉 ALL DONE! Your final deployable LoRA adapter is located at:\n{os.path.abspath(latest_checkpoint)}/consolidated/lora.safetensors"
    yield "\nYou can load this into the Moshi server using the --lora-weight argument."

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
                scenario = gr.Textbox(
                    label="Dialogue Scenario Description", 
                    lines=5, 
                    value=config.get("scenario", ""),
                    placeholder="e.g., Agent Alexis Kim calling a business to verify pre-employment background checks..."
                )
                system_prompt = gr.Textbox(
                    label="System Prompt (Persona)", 
                    lines=3, 
                    value=config.get("system_prompt", "<system> You are a helpful assistant. <system>"),
                    info="This is the exact prompt the model will be conditioned on during training. Must be wrapped in <system> tags."
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
        scenario.change(lambda x: save_config("scenario", x), inputs=[scenario])
        system_prompt.change(lambda x: save_config("system_prompt", x), inputs=[system_prompt])
        num_samples.change(lambda x: save_config("num_samples", x), inputs=[num_samples])
        num_turns.change(lambda x: save_config("num_turns", x), inputs=[num_turns])

        gen_text_btn.click(generate_transcripts, inputs=[api_key, model_name, scenario, num_samples, num_turns], outputs=[text_output, dataset_df])
        dataset_df.change(save_transcripts_df, inputs=[dataset_df])

    with gr.Tab("2. Generate Audio (Dia2)"):
        gr.Markdown("This step uses Dia2 to generate the audio and timestamps based on the transcripts generated in Step 1.")
        gen_audio_btn = gr.Button("Generate Audio & Timestamps", variant="primary")
        audio_output = gr.Textbox(label="Terminal Output", lines=15)
        
        gen_audio_btn.click(generate_audio, outputs=audio_output)

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
                base_model = gr.Dropdown(choices=["kyutai/moshiko-pytorch-bf16", "nvidia/personaplex-7b-v1"], value=config.get("base_model", "kyutai/moshiko-pytorch-bf16"), label="Base Model", info="Select the base Moshi model to fine-tune.")
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

        train_btn.click(run_training, inputs=[base_model, max_steps, batch_size, lr, mix_dailytalk, lora_rank, lora_scaling, duration_sec], outputs=train_output)

    with gr.Tab("5. Export Model"):
        gr.Markdown("**Final Step:** Locate the final LoRA adapter weights for deployment.")
        export_btn = gr.Button("Export Final Model", variant="primary")
        export_output = gr.Textbox(label="Export Logs", lines=15)
        
        export_btn.click(export_model, outputs=export_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
