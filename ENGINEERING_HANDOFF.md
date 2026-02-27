# Engineering Handoff: PersonaPlex Outbound Call Fine-Tuning

## Project Objective
The goal of this project is to fine-tune the `nvidia/personaplex-7b-v1` model (which is based on the Kyutai Moshi architecture) to eliminate its strong bias for inbound customer service greetings (e.g., "Thank you for calling...") and train it to naturally initiate outbound sales/verification calls.

We will use the official `kyutai-labs/moshi-finetune` repository for lightweight LoRA training. The training data will be entirely synthetic, generated using an LLM for the transcripts and Dia2 for the audio.

---

## Phase 1: Synthetic Data Generation Pipeline

The fine-tuning repository requires a specific data format: Stereo WAV files and JSON transcripts with word-level timestamps.

### 1. Transcript Generation (LLM)
*   **Tool:** Gemini 3.1 Pro Preview (or similar high-tier LLM).
*   **Task:** Generate 50-100 short transcripts of successful outbound call initiations.
*   **Format:** The LLM should output JSON representing the dialogue turns.
*   **Scenario:** Agent (Alexis Kim) calling a business to verify pre-employment background checks.
*   **Flow:** 
    *   User: "Hello?" or "Directory Services, how can I help you?"
    *   Agent: "Hi, this is Alexis calling from Directory Services. I'm reaching out because..."

### 2. Audio Generation and Alignment (Dia2)
*   **Tool:** Dia2.
*   **Task:** Generate audio for both speakers and extract word-level timestamps simultaneously.
*   **Requirement:** The audio must be combined into a **single Stereo WAV file** per dialogue, and a corresponding JSON file with timestamps must be created.
    *   Left Channel: Speaker A (Moshi/Agent)
    *   Right Channel: Speaker B (User)
*   **Format Required:**
    ```json
    {
        "alignments": [
            ["hello", [0.46, 1.52], "SPEAKER_MAIN"],
            ["hi", [1.82, 2.04], "SPEAKER_USER"]
        ]
    }
    ```

---

## Phase 2: Data Preparation

The `generate_audio_dia2.py` script automatically generates a `dataset.jsonl` file containing the absolute paths to the generated audio files and their durations.

```jsonl
{"path": "/files/moshi-finetune/data/outbound/audio/001.wav", "duration": 12.5}
```

---

## Phase 3: Training

Run the lightweight LoRA training using the `kyutai-moshi-finetune` repository.
```bash
uv run torchrun --nproc-per-node 1 -m train config.yaml
```

---

## Phase 4: Deployment

The resulting LoRA adapters in `output/personaplex-outbound/checkpoints/` can be loaded directly into the Moshi server using the `--lora-weight` argument.

---

## Next Steps for the AI Agent
1. Run `run_pipeline.sh` to execute the full pipeline.
