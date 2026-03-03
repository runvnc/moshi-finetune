# Moshi Synthetic Fine-Tuning Studio

This repository provides an end-to-end pipeline for fine-tuning `nvidia/personaplex-7b-v1` on custom synthetic dialogue datasets using LoRA.

It is a fork of [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune), extended with a synthetic data generation pipeline, Gradio UI, and PersonaPlex-specific fixes.

---

## Quick Reference

| Repo | Local | RunPod | GitHub |
|------|-------|--------|--------|
| moshi-finetune | /files/moshi-finetune | /workspace/moshi-finetune | github.com/runvnc/moshi-finetune |
| personaplex | /files/personaplex | /workspace/personaplex | github.com/runvnc/personaplex |

---

## Fresh RunPod Setup

```bash
cd /workspace
git clone https://github.com/runvnc/moshi-finetune.git
git clone https://github.com/runvnc/personaplex.git
```

### Install training venv

```bash
cd /workspace/moshi-finetune

# openai-whisper requires these before uv sync
uv pip install 'setuptools<70' wheel

uv sync
```

---

## Training

### Launch the Gradio UI

```bash
cd /workspace/moshi-finetune
uv run gradio_app.py --system
# Opens on port 7860
```

### Workflow

1. **Tab 1 – Transcripts**: Generate dialogue scripts (requires Gemini API key). Describe your scenario and persona.
2. **Tab 2 – Audio**: Generate stereo WAV + word-level timestamps (requires ElevenLabs API key). The system prompt / persona is saved as `text_conditions` in each JSON file.
3. **Tab 3 – DailyTalk** *(optional)*: Download regularization data to prevent catastrophic forgetting.
4. **Tab 4 – Train**: Select `nvidia/personaplex-7b-v1`, enter HF token, click Start. Default LoRA rank=32, scaling=2.0.
5. **Tab 5 – Export**: Shows the exact path to your `lora.safetensors` and the server command.

### Text / Persona Conditioning During Training

The `text_conditions` field (your persona/system prompt) is saved in each JSON file by the audio generation step. During training, these tokens are prepended to the text channel of each training sequence — mirroring exactly what happens at inference time when `step_system_prompts_async` injects the system prompt before the conversation starts.

This means the LoRA learns to generate responses **conditioned on the persona text**.

### Checkpoint location

```
output/custom_model/checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors
```

For real training use 500–2000 steps. The smoke-test default is 10 steps.

---

## Running the Server with a Trained LoRA

> **Important**: Both `moshi-finetune` (upstream kyutai moshi) and `personaplex` install into the same `moshi` Python namespace. You must install the personaplex fork into the venv before serving, then restore for training.

### Step 1 – Switch venv to personaplex moshi

```bash
cd /workspace/moshi-finetune
uv pip install -e /workspace/personaplex/moshi/
```

### Step 2 – Run the server

```bash
.venv/bin/python3 /workspace/personaplex/moshi/moshi/server.py \
  --hf-repo nvidia/personaplex-7b-v1 \
  --lora-weight /workspace/moshi-finetune/output/custom_model/checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors \
  --lora-rank 32 \
  --lora-scaling 2.0 \
  --host 0.0.0.0 \
  --port 8998
```

Replace `checkpoint_XXXXXX` with your actual checkpoint number (Tab 5 in the UI shows the exact path).

The server downloads model weights from HF on first run (~14GB). Access the web UI at `http://<host>:8998`.

### Step 3 – Restore training venv (when done serving)

```bash
cd /workspace/moshi-finetune
uv sync
```

### Server flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-weight` | None | Path to `lora.safetensors` |
| `--lora-rank` | 32 | Must match value used during training |
| `--lora-scaling` | 2.0 | Must match value used during training |
| `--hf-repo` | nvidia/personaplex-7b-v1 | HF repo for base weights |
| `--host` | localhost | Use `0.0.0.0` for RunPod |
| `--port` | 8998 | Port to serve on |
| `--wait-for-user` | False | Wait for user to speak first (outbound call mode) |
| `--voice-prompt-dir` | auto | Directory of voice prompt files; auto-downloads from HF if omitted |
| `--cpu-offload` | False | Offload layers to CPU if GPU VRAM is tight |

### Text / Persona Conditioning at Inference

PersonaPlex conditions on the persona via **text token injection** — not a separate conditioning module. The system prompt is tokenized and fed into the model’s context before the conversation starts.

The client sends the persona via the `text_prompt` WebSocket query parameter. In the web UI this is the "System Prompt" field. Example value:

```
You are Alex, a friendly outbound sales agent for Acme Corp. You are calling to follow up on a recent inquiry about our cloud services. Be professional, concise, and helpful.
```

The server wraps it in `<system> ... <system>` tags automatically.

---

## Architecture Notes

### Why two moshi forks?

| | Upstream kyutai moshi | PersonaPlex fork |
|---|---|---|
| Used for | Training | Inference |
| Has | FSDP, LoRA training infra | Voice cloning, `dep_q=16`, `get_lora_moshi()` with key translation |

### Key architecture difference

PersonaPlex uses `dep_q=16` (vs base moshi `dep_q=8`). The full set of defaults is injected at training time via `_PERSONAPLEX_LM_DEFAULTS` in `train.py`.

### LoRA key translation

Upstream moshi uses `.in_projs.0.` / `.out_projs.0.` key names. PersonaPlex uses `.in_proj.` / `.out_proj.`. The `get_lora_moshi()` function in the personaplex fork translates automatically on load.

---

## Environment

- RunPod: RTX 5090, CUDA driver 12.0.90
- Python 3.12
- torch 2.10+cu128 (in moshi-finetune venv — required; system Python has cu130 which needs driver 13.0+)
- uv 0.8.5

---

## Troubleshooting

**`uv sync` fails on openai-whisper**
```bash
uv pip install 'setuptools<70' wheel
uv sync
```

**Server crashes immediately on model load**
Make sure you installed personaplex moshi into the venv first:
```bash
uv pip install -e /workspace/personaplex/moshi/
```

**LoRA weights fail to load (unexpected_keys error)**
The key translation in `get_lora_moshi()` handles `.in_projs.0.` → `.in_proj.` automatically. If you see unexpected keys, check that `--lora-rank` and `--lora-scaling` match the values used during training (default: rank=32, scaling=2.0).

**Port not accessible on RunPod**
Use `--host 0.0.0.0` and expose port 8998 in the RunPod pod configuration.

**Want to run training and serving on the same pod**
Switch between them by installing the appropriate moshi fork:
- For training: `uv sync` (restores upstream moshi)
- For serving: `uv pip install -e /workspace/personaplex/moshi/`
