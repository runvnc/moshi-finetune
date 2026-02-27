import gradio as gr
import os
import json
import subprocess
import glob
import google.generativeai as genai

def generate_transcripts(api_key, model_name, scenario, num_samples, num_turns):
    if not api_key:
        return "Error: Please provide a Gemini API Key."
    if not scenario:
        return "Error: Please provide a scenario description."
        
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are an expert dialogue writer. Generate {num_samples} unique, short dialogue transcripts.
    
    Scenario:
    {scenario}
    
    Format Requirements:
    - Speaker A is the User (the person answering the phone).
    - Speaker B is the Agent (the AI caller).
    - CRITICAL: Speaker A MUST speak first in every single conversation. The conversation must always start with the User answering the phone (e.g., "Hello?", "Acme Corp, how can I help you?").
    - Speaker B (the Agent) speaks second, introducing themselves and stating the purpose of the call.
    - Each conversation should be approximately {num_turns} turns long, including a natural back-and-forth exchange.
    - Output strictly as a JSON array of conversations. Each conversation is an array of turn objects.
    
    Example Output:
    [
      [
        {{"speaker": "A", "text": "Hello?"}},
        {{"speaker": "B", "text": "Hi, this is Alexis. I'm calling about..."}}
      ]
    ]
    
    Generate exactly {num_samples} conversations in this exact JSON format.
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
            )
        )
        
        data = json.loads(response.text)
        
        os.makedirs("data/outbound", exist_ok=True)
        output_file = "data/outbound/raw_transcripts.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
        preview = json.dumps(data[0], indent=2) if data else "No data generated."
        return f"Successfully generated {len(data)} transcripts! Saved to {output_file}.\n\nPreview of first transcript:\n{preview}"
        
    except Exception as e:
        return f"Failed to generate transcripts: {str(e)}"

def run_command(command):
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
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

def run_training(max_steps, batch_size, learning_rate, mix_dailytalk):
    yield "Starting LoRA Fine-Tuning...\n"
    
    train_data = "'/files/moshi-finetune/data/outbound/dataset.jsonl'"
    if mix_dailytalk:
        train_data = "'/files/moshi-finetune/data/outbound/dataset.jsonl:0.7,/files/moshi-finetune/data/dailytalk_subset/dailytalk_subset.jsonl:0.3'"
    
    # Update config.yaml
    config_content = f"""# data
data:
  eval_data: ''
  shuffle: true
  train_data: {train_data}

# model
moshi_paths: 
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.
  ft_embed: false

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

# optim
duration_sec: 100
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

run_dir: "/files/moshi-finetune/output/personaplex-custom"
"""
    with open("config_custom.yaml", "w") as f:
        f.write(config_content)
    
    cmd = "uv pip install -e . && uv run torchrun --nproc-per-node 1 -m train config_custom.yaml"
    for log in run_command(cmd):
        yield log

def export_model():
    yield "Finding latest LoRA checkpoint...\n"
    
    checkpoints = glob.glob("output/personaplex-custom/checkpoints/checkpoint_*")
    
    if not checkpoints:
        yield "❌ Error: No training checkpoints found in output/personaplex-custom/"
        return
        
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]
    step_name = os.path.basename(latest_checkpoint)
    
    yield f"Found latest checkpoint: {step_name}\n"
    yield "\nWith LoRA, no complex export is needed! The adapters are already saved as safetensors.\n"
    yield f"\n\n🎉 ALL DONE! Your final deployable LoRA adapter is located at:\n{os.path.abspath(latest_checkpoint)}/consolidated/lora.safetensors"
    yield "\nYou can load this into the Moshi server using the --lora-weight argument."

# --- Gradio UI Layout ---
with gr.Blocks(title="Moshi Fine-Tuning Studio") as app:
    gr.Markdown("# 🎛️ Moshi / PersonaPlex Fine-Tuning Studio")
    gr.Markdown("Generate synthetic dialogue data and fine-tune your audio language model end-to-end.")
    
    with gr.Tab("1. Generate Transcripts"):
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="Gemini API Key", type="password")
                model_name = gr.Dropdown(choices=["gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-1.5-pro"], value="gemini-3-flash-preview", label="Gemini Model")
                scenario = gr.Textbox(
                    label="Dialogue Scenario Description", 
                    lines=5, 
                    placeholder="e.g., Agent Alexis Kim calling a business to verify pre-employment background checks..."
                )
                num_samples = gr.Slider(minimum=1, maximum=500, value=50, step=1, label="Number of Conversations")
                num_turns = gr.Slider(minimum=2, maximum=20, value=6, step=1, label="Target Turns per Conversation")
                gen_text_btn = gr.Button("Generate Transcripts", variant="primary")
            
            with gr.Column():
                text_output = gr.Textbox(label="Output / Preview", lines=15)
                
        gen_text_btn.click(generate_transcripts, inputs=[api_key, model_name, scenario, num_samples, num_turns], outputs=text_output)

    with gr.Tab("2. Generate Audio (Dia2)"):
        gr.Markdown("This step uses Dia2 to generate the audio and timestamps based on the transcripts generated in Step 1.")
        gen_audio_btn = gr.Button("Generate Audio & Timestamps", variant="primary")
        audio_output = gr.Textbox(label="Terminal Output", lines=15)
        
        gen_audio_btn.click(generate_audio, outputs=audio_output)

    with gr.Tab("3. Download DailyTalk (Optional)"):
        gr.Markdown("Download a subset of the DailyTalk dataset to mix with your synthetic data. This prevents catastrophic forgetting and keeps the voice sounding natural.")
        with gr.Row():
            with gr.Column():
                dt_samples = gr.Slider(minimum=50, maximum=2000, value=200, step=50, label="Number of DailyTalk Samples to Download")
                download_dt_btn = gr.Button("Download DailyTalk Subset", variant="secondary")
            with gr.Column():
                dt_output = gr.Textbox(label="Download Logs", lines=10)
                
        download_dt_btn.click(download_dailytalk, inputs=[dt_samples], outputs=dt_output)

    with gr.Tab("4. Train Model (LoRA)"):
        with gr.Row():
            with gr.Column():
                max_steps = gr.Slider(minimum=10, maximum=2000, value=50, step=10, label="Max Training Steps", info="For 100 examples, 50 steps is usually enough.")
                batch_size = gr.Slider(minimum=4, maximum=64, value=16, step=4, label="Batch Size", info="Increase to 32 or 48 if you have an H200/A100. Keep at 8-16 for 24GB GPUs.")
                lr = gr.Textbox(label="Learning Rate", value="2e-6", info="Default is 2e-6.")
                mix_dailytalk = gr.Checkbox(label="Mix with DailyTalk Subset (70/30 split)", value=True, info="Requires downloading the subset in Step 3 first.")
                train_btn = gr.Button("Start Fine-Tuning", variant="primary")
            
            with gr.Column():
                train_output = gr.Textbox(label="Training Logs", lines=20)
                
        train_btn.click(run_training, inputs=[max_steps, batch_size, lr, mix_dailytalk], outputs=train_output)

    with gr.Tab("5. Export Model"):
        gr.Markdown("**Final Step:** Locate the final LoRA adapter weights for deployment.")
        export_btn = gr.Button("Export Final Model", variant="primary")
        export_output = gr.Textbox(label="Export Logs", lines=15)
        
        export_btn.click(export_model, outputs=export_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
