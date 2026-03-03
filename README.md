# Moshi Synthetic Fine-Tuning Studio

This repository provides an end-to-end pipeline for fine-tuning Moshi-based models (like `kyutai/moshiko-pytorch-bf16` or `nvidia/personaplex-7b-v1`) on custom synthetic dialogue datasets.

It is a fork of the official [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) repository, customized with a synthetic data generation pipeline and a Gradio UI, allowing you to train Moshi for any conversational scenario (e.g., outbound sales, drive-thru attendant, pirate roleplay, etc.).

## Features

*   **Synthetic Data Generation:** Uses Gemini to generate realistic dialogue transcripts based on any scenario you describe.
*   **Audio & Timestamp Generation:** Uses Dia2 to generate stereo audio (Agent on left channel, User on right channel) and exact word-level timestamps.
*   **Data Mixing:** Automatically downloads a subset of the `DailyTalkContiguous` dataset to mix with your synthetic data, preventing catastrophic forgetting and maintaining natural voice quality.
*   **Lightweight LoRA Training:** Uses the official Kyutai LoRA fine-tuning implementation, allowing you to train on consumer GPUs (e.g., RTX 3090/4090) or high-end GPUs (H100/H200).
*   **Gradio UI:** A complete web interface (`gradio_app.py`) to manage the entire pipeline from data generation to model export.

## Quick Start

### 1. Install Dependencies

We recommend using `uv` for fast dependency management.

**Clone and sync:**

```bash
git clone https://github.com/runvnc/moshi-finetune.git
cd moshi-finetune
```

> **Note:** `openai-whisper` has broken build metadata and requires `setuptools<70` and `wheel` to be pre-installed in the venv before `uv sync` will succeed. Run:

```bash
uv pip install 'setuptools<70' wheel
uv sync
```

This is a known upstream issue with `openai-whisper==20240930`. The `[tool.uv.extra-build-dependencies]` workaround is present in `pyproject.toml` but the feature is experimental and unreliable in current uv versions, so the manual pre-install step is required.

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

## Tech Notes & Architecture

### Data Format Requirements
The Kyutai fine-tuning pipeline expects a specific data format:
1.  **Stereo WAV files:** The AI Agent (Moshi) must be on the **Left Channel** (Channel 0), and the User must be on the **Right Channel** (Channel 1).
2.  **JSON Timestamps:** Each `.wav` file must have a corresponding `.json` file with exact word-level timestamps for the AI Agent's speech, formatted like this:
    ```json
    {
        "alignments": [
            ["hello", [0.46, 1.52], "SPEAKER_MAIN"],
            ["how", [1.82, 2.04], "SPEAKER_MAIN"]
        ]
    }
    ```
3.  **Dataset Manifest:** A `dataset.jsonl` file containing the absolute paths to the generated audio files and their durations:
    ```jsonl
    {"path": "/absolute/path/to/data/custom_dataset/audio/001.wav", "duration": 12.5}
    ```

*Note: The `generate_audio_dia2.py` script handles all of this formatting automatically.*

### LoRA Adapters & Deployment
Because this pipeline uses Low-Rank Adaptation (LoRA), the base 7B model weights are frozen during training. The output of the training process is a small `lora.safetensors` file (usually a few hundred megabytes) located in `output/custom_model/checkpoints/checkpoint_XXXX/consolidated/`.

To deploy your fine-tuned model, you do not need to merge the weights. You can pass the adapter directly to the Moshi server:
```bash
python -m moshi.server \
  --lora-weight=/path/to/lora.safetensors \
  --config-path=/path/to/config.json
```

For details on the underlying Kyutai LoRA fine-tuning implementation, please refer to the original [README_KYUTAI.md](README_KYUTAI.md).
