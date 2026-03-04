# Engineering Handoff - Moshi PersonaPlex Fine-Tuning Studio

## Overview

This repo (`moshi-finetune`) is a fine-tuning studio for `nvidia/personaplex-7b-v1`, a voice AI model based on Kyutai Moshi. It provides a Gradio UI for generating synthetic dialogue data, training LoRA adapters, and deploying them.

## Repo Locations

| Repo | Local | RunPod | GitHub |
|------|-------|--------|--------|
| moshi-finetune | /files/moshi-finetune | /workspace/moshi-finetune | github.com/runvnc/moshi-finetune |
| personaplex | /files/personaplex | /workspace/personaplex | github.com/runvnc/personaplex |

## Architecture

### Two Moshi Forks

| | Upstream Kyutai Moshi | PersonaPlex Fork |
|---|---|---|
| Used for | Training | Inference |
| Has | CheckpointInfo, FSDP, LoRA training infra | Voice cloning, dep_q=16, get_lora_moshi() |
| Installed | Via pyproject.toml git dep in moshi-finetune venv | pip install -e /workspace/personaplex/moshi/ |
| Python module name | `moshi` | `moshi` (CONFLICT - see below) |

### Key Architecture Difference

PersonaPlex uses `dep_q=16` and `depformer_weights_per_step=True`. The upstream kyutai moshi uses `dep_q=8` by default but the training code patches this via `_PERSONAPLEX_LM_DEFAULTS` in `train.py`.

### LoRA Key Translation Problem (PARTIALLY FIXED)

The upstream kyutai moshi stores weights_per_step attention as a **ModuleList** of N separate linears:
- `depformer.layers.0.self_attn.in_projs.0.lora_B.weight` shape [3072, 32]
- `depformer.layers.0.self_attn.in_projs.1.lora_B.weight` shape [3072, 32]
- ... through in_projs.15

PersonaPlex stores it as one **fused linear**:
- `depformer.layers.0.self_attn.in_proj.lora_B.weight` shape [49152, 32] = 16 * 3072

The translation in `personaplex/moshi/moshi/models/loaders.py` `get_lora_moshi()` now:
1. Groups all `in_projs.N` keys by prefix
2. Stacks `lora_B` tensors along dim=0 to form fused tensor
3. Takes `lora_A` from index 0 only (shared down-projection)

## Current Status (as of 2026-03-04)

### What Works
- Training runs correctly with dep_q=16 and depformer_weights_per_step=True (verified via debug log)
- LoRA checkpoint shapes are now correct (in_projs.N structure)
- Base model loads without crashing (fixed meta tensor .to() bug in get_moshi_lm)
- LoRA key translation stacks in_projs.N -> in_proj correctly

### What Is Broken / In Progress

**CURRENT BUG**: `get_lora_moshi()` in personaplex loaders.py line ~461 calls `model.to(dtype=dtype, device=device)` AFTER `load_state_dict(..., assign=True)`. This crashes with:
```
NotImplementedError: Cannot copy out of meta tensor
```
because `transformer.layers.*.self_attn.in_proj.weight` (the base model weights, not LoRA) are still meta tensors - they were missing from the base model safetensors and left as meta. The LoRA only fills the lora_A/lora_B sub-weights, not the frozen_W.

**Fix needed**: In `get_lora_moshi()`, remove or replace the `model = model.to(dtype=dtype, device=device)` call at line ~461. Same fix as was applied to `get_moshi_lm()` - just return model without calling .to().

File: `/files/personaplex/moshi/moshi/models/loaders.py`

Look for:
```python
        model = model.to(dtype=dtype, device=device)
        if fuse_lora:
            replace_lora_with_linear(model)
    return model
```

Change to:
```python
        # NOTE: do NOT call model.to() - assign=True already placed tensors on device
        # and remaining meta tensors (frozen_W for in_proj) will be filled by LoRA forward
        if fuse_lora:
            replace_lora_with_linear(model)
    return model
```

However, there may be a deeper issue: the `frozen_W` inside each `LoRALinear` for the `in_proj` layers may still be meta tensors since the base model weights for those layers are missing from the safetensors file. Need to verify this doesn't cause a runtime crash during inference.

### Why in_proj weights are missing from base model

The personaplex base model safetensors does NOT contain `transformer.layers.*.self_attn.in_proj.weight` keys. This is because the base model was saved with the upstream kyutai format (`in_projs.0` through `in_projs.N`), but the personaplex inference code expects the fused `in_proj` format. The `get_moshi_lm()` function handles this by logging "Missing" for those keys and leaving them as meta tensors, expecting LoRA to fill them.

But LoRA only fills `lora_A` and `lora_B`, not `frozen_W`. So `frozen_W` inside each `LoRALinear` for `in_proj` is a meta tensor. This will crash on first forward pass.

**Possible fix**: In `replace_all_linear_with_lora()`, when creating a LoRALinear to replace a meta-device Linear, initialize `frozen_W` as zeros on the target device rather than meta. Or: load the base model weights for in_proj by translating from the base model safetensors (which has in_projs.0 through in_projs.N).

## Running the Server (when working)

```bash
cd /workspace/personaplex
git pull
python3 -m moshi.server \
  --hf-repo nvidia/personaplex-7b-v1 \
  --lora-weight /workspace/moshi-finetune/output/custom_model/checkpoints/checkpoint_XXXXXX/consolidated/lora.safetensors \
  --lora-rank 32 \
  --lora-scaling 2.0 \
  --host 0.0.0.0 \
  --port 3030
```

## Running the Training UI

```bash
cd /workspace/moshi-finetune
git pull
uv run gradio_app.py --system
# Opens on port 7860
```

The training tab now defaults to nvidia/personaplex-7b-v1 only.

## Key Files

- `train.py` - Main training script. Contains `_PERSONAPLEX_LM_DEFAULTS` which patches dep_q=16 etc.
- `gradio_app.py` - Gradio UI
- `finetune/wrapped_model.py` - FSDP model wrapper, calls CheckpointInfo.get_moshi()
- `personaplex/moshi/moshi/models/loaders.py` - THE KEY FILE. Contains get_moshi_lm() and get_lora_moshi()
- `personaplex/moshi/moshi/modules/lora.py` - LoRALinear implementation

## Environment (RunPod)

- GPU: RTX 5090
- Python: 3.12 system
- Training: `uv run ... --system` (no venv, system Python)
- Inference: `python3 -m moshi.server` (system Python, personaplex moshi on path)
