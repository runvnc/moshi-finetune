#!/bin/bash

# Exit on error
set -e

echo "===================================================="
echo " PersonaPlex Outbound Call Fine-Tuning Pipeline"
echo "===================================================="

# 1. Generate Transcripts (Requires GEMINI_API_KEY)
echo "\n[1/3] Generating Transcripts with Gemini..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY is not set. Please export it first."
    exit 1
fi
uv pip install google-generativeai
uv run generate_transcripts.py

# 2. Generate Audio and Timestamps with Dia2
echo "\n[2/3] Generating Audio and Timestamps with Dia2..."
# Ensure torchaudio is installed for the stereo mixing
uv pip install torchaudio
uv run generate_audio_dia2.py

# 3. Training
echo "\n[3/3] Starting LoRA Fine-Tuning..."
echo "(This will take some time depending on your GPU)"

# Ensure dependencies are installed
uv pip install -e .
uv run torchrun --nproc-per-node 1 -m train config.yaml

echo "\n===================================================="
echo " Training Complete!"
echo " Run the post-processing steps in ENGINEERING_HANDOFF.md to deploy."
echo "===================================================="
