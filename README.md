# PersonaPlex Outbound Call Fine-Tuning

This repository provides an end-to-end pipeline for fine-tuning the `nvidia/personaplex-7b-v1` model (based on the Kyutai Moshi architecture) to naturally initiate outbound sales and verification calls.

It is a fork of the official [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) repository, customized with a synthetic data generation pipeline and a Gradio UI.

## Features

*   **Synthetic Data Generation:** Uses Gemini to generate realistic outbound call transcripts.
*   **Audio & Timestamp Generation:** Uses Dia2 to generate stereo audio (Agent on left channel, User on right channel) and exact word-level timestamps.
*   **Data Mixing:** Automatically downloads a subset of the `DailyTalkContiguous` dataset to mix with your synthetic data, preventing catastrophic forgetting and maintaining natural voice quality.
*   **Lightweight LoRA Training:** Uses the official Kyutai LoRA fine-tuning implementation, allowing you to train on consumer GPUs (e.g., RTX 3090/4090) or high-end GPUs (H100/H200).
*   **Gradio UI:** A complete web interface (`gradio_app.py`) to manage the entire pipeline from data generation to model export.

## Quick Start

### 1. Install Dependencies

We recommend using `uv` for fast dependency management:

```bash
uv pip install -e .
uv pip install google-generativeai gradio huggingface_hub tqdm torchaudio
```

### 2. Launch the Studio UI

The easiest way to use this pipeline is through the Gradio web interface:

```bash
uv run gradio_app.py
```

This will open a web UI where you can:
1. Generate transcripts using your Gemini API key.
2. Generate audio and timestamps using Dia2.
3. Download a subset of the DailyTalk dataset for regularization.
4. Configure and launch the LoRA fine-tuning process.
5. Locate your final `lora.safetensors` adapter for deployment.

### 3. Command Line Pipeline

Alternatively, you can run the automated bash script:

```bash
export GEMINI_API_KEY="your_api_key_here"
./run_pipeline.sh
```

## Engineering Details

For a deep dive into the architecture, data formats, and deployment steps, please read the [ENGINEERING_HANDOFF.md](ENGINEERING_HANDOFF.md).

For details on the underlying Kyutai LoRA fine-tuning implementation, please refer to the original [README_KYUTAI.md](README_KYUTAI.md).
