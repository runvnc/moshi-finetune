# Engineering Handoff: Moshi-Finetune + PersonaPlex

**Date:** 2026-03-03  
**Context:** Fine-tuning PersonaPlex (nvidia/personaplex-7b-v1) with LoRA for outbound call scenarios.

---

## Repos

| Repo | Local Path | RunPod Path | GitHub |
|------|-----------|-------------|--------|
| moshi-finetune | /files/moshi-finetune | /workspace/moshi-finetune | github.com/runvnc/moshi-finetune |
| personaplex | /files/personaplex | /workspace/personaplex | github.com/runvnc/personaplex |

---

## Architecture Overview

**Training:** Uses upstream kyutai moshi (pinned to commit `061cc4c`) via the moshi-finetune venv. The personaplex architecture defaults (dep_q=16, etc.) are injected at runtime in `train.py` since the personaplex HF config.json only contains `{"model_type": "personaplex", "version": "7b-v1"}`.

**Inference:** Uses the personaplex moshi fork (`/workspace/personaplex/moshi/`) which has voice cloning support. The LoRA weights from training use Kyutai key format; `get_lora_moshi()` in the personaplex fork translates them automatically (`.in_projs.0.` -> `.in_proj.`).

**Why two different moshi forks?**
- Upstream moshi has FSDP, CheckpointInfo, proper LoRA training infrastructure
- PersonaPlex fork has voice cloning, the right dep_q=16 architecture, and `get_lora_moshi()` with key translation
- The key translation in personaplex's `get_lora_moshi()` was specifically designed for this cross-fork workflow

---

## Key Architecture Differences: PersonaPlex vs Base Moshi

PersonaPlex uses `dep_q=16` (vs base moshi `dep_q=8`). All other arch params are the same. The full set of defaults is in:
- `train.py`: `_PERSONAPLEX_LM_DEFAULTS` dict (injected into lm_config at training time)
- `configs/personaplex.json`: same values, used for inference with upstream moshi server

---

## Training Setup (moshi-finetune)

### Install
```bash
cd /workspace/moshi-finetune
git pull
uv pip install 'setuptools<70' wheel  # required before uv sync (openai-whisper build bug)
uv sync
```

### Launch UI
```bash
uv run gradio_app.py --system
# Opens on port 7860
```

### Training Flow
1. Tab 1: Generate transcripts (Gemini API key required)
2. Tab 2: Generate audio (ElevenLabs API key required) - saves stereo WAV + JSON timestamps + `text_conditions` (system prompt)
3. Tab 3: Download DailyTalk subset (optional, prevents catastrophic forgetting)
4. Tab 4: Train - select `nvidia/personaplex-7b-v1`, enter HF token, click Start
5. Tab 5: Export - shows exact paths and server command

### Training Notes
- `text_conditions` (persona/system prompt) is saved in the JSON files and read by the interleaver
- The base personaplex model has `condition_provider=None` when loaded via upstream moshi, so text conditioning is silently dropped during training (the model learns persona from audio patterns only)
- LoRA rank=32, scaling=2.0 are the defaults
- Checkpoint saved to `output/custom_model/checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors`

---

## Inference Setup (PersonaPlex Server)

### The Namespace Conflict Problem

Both `moshi` (upstream, from git) and `moshi-personaplex` (personaplex fork) install into the `moshi` Python namespace. When both are present, they conflict. The upstream moshi's `server.py` doesn't have `--lora-rank`/`--lora-scaling` args; the personaplex one does.

**Current workaround:** Run the personaplex server.py directly by full path using the moshi-finetune venv (which has the right torch version - 2.10+cu128):

```bash
cd /workspace/moshi-finetune

# Install personaplex moshi into the finetune venv (overrides upstream moshi)
uv pip install -e /workspace/personaplex/moshi/

# Run server directly by path to avoid namespace ambiguity
.venv/bin/python3 /workspace/personaplex/moshi/moshi/server.py \
  --hf-repo nvidia/personaplex-7b-v1 \
  --lora-weight output/custom_model/checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors \
  --lora-rank 32 --lora-scaling 2.0 \
  --host 0.0.0.0 --port 8188
```

**After serving, restore training venv:**
```bash
uv sync  # restores upstream moshi for training
```

### Why not use system Python for serving?
The system Python has `torch 2.10+cu130` which requires CUDA driver 13.0+. The RunPod instance has driver 12.0.90 (too old). The moshi-finetune venv has `torch 2.10+cu128` which works.

### Voice Cloning
The personaplex server supports voice cloning via `--voice-prompt-dir`. Voice prompts are downloaded automatically from the HF repo to `~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/.../voices/`. Pass a custom voice prompt dir with `--voice-prompt-dir /path/to/voices/`.

### Client UI
The personaplex server serves its own static client from the HF repo cache. Access at `http://<host>:8188` after server starts.

---

## Key Fixes Made (2026-03-03)

| Commit | Fix |
|--------|-----|
| ac05871 | torch>=2.8 override, pin moshi git commit, fix openai-whisper build |
| 453cd3b | Add wheel to openai-whisper extra-build-dependencies |
| fa6556e | Add soundfile dependency |
| a2ae7ae | Remove non-existent mimi_config arg from get_mimi call |
| 772ea90 | Guard condition_provider.prepare() when model has no condition_provider |
| 1f0c4a9 | Pass HF token to training subprocess env |
| 0d152de | Export tab shows full paths and file sizes |
| 33591cc | Add configs/personaplex.json with full arch params |
| 2d2b2be | Export tab shows personaplex server command |

**personaplex repo:**
| Commit | Fix |
|--------|-----|
| 602898e | Wire --lora-weight to get_lora_moshi, add --lora-rank/--lora-scaling args, remove torch<2.5 constraint |

---

## Known Issues / TODO

1. **Namespace conflict**: Running `python3 -m moshi.server` picks up whichever moshi is first in sys.path. Use full path to server.py as workaround (see above). Proper fix: create a dedicated serving venv or rename the personaplex package's module.

2. **Text conditioning not used during training**: The personaplex model loaded via upstream moshi has `condition_provider=None` (the HF config doesn't specify conditioner params). The persona text from `text_conditions` is silently dropped. The model learns persona from audio patterns only. To fix: would need to add conditioner config to the personaplex HF config or inject it in training code.

3. **Only 10 training steps tested**: The test run used 10 steps. Real training needs 500-2000 steps with a proper dataset.

4. **openai-whisper manual pre-install**: On fresh installs, must run `uv pip install 'setuptools<70' wheel` before `uv sync`. Documented in README.

---

## Data Pipeline

```
Gemini API -> transcripts (JSON with system_prompt, turns)
    -> ElevenLabs API -> stereo WAV (agent=left, user=right) + timestamps JSON
    -> dataset.jsonl (paths + durations)
    -> [optional] mix with DailyTalk 70/30
    -> training
```

Data lives in `data/custom_dataset/`. The `text_conditions` field in the timestamp JSON carries the system prompt for conditioning.

---

## Environment

- RunPod: RTX 5090, CUDA driver 12.0.90
- Python 3.12
- torch 2.10+cu128 (in moshi-finetune venv)
- uv 0.8.5
- moshi-finetune venv: `/workspace/moshi-finetune/.venv`
