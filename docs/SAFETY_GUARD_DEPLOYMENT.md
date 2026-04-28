# Safety Guard Qwen3-14B — Deployment Guide

## Overview

This LoRA trains **Qwen3-14B** as a **Nemotron-style safety classifier** that outputs structured JSON verdicts.  
It classifies conversations against a 23-category Aegis 2.0 safety taxonomy (S1–S23).

**Training notebook:** `safety_guard_qwen3_14b_instruct_unsloth_4bit.ipynb`  
**OpenWebUI filter:** `safety/safety_filter_guard_v3.py`  
**System prompt:** `safety/prompt.py`

---

## Architecture

### How the classifier works

```
User message + (optional) assistant response
        ↓
Filter builds Nemotron-style prompt:
  - Task instruction
  - <BEGIN UNSAFE CONTENT CATEGORIES> ... taxonomy ... <END>
  - <BEGIN CONVERSATION> ... messages ... <END>
  - Output format instruction
        ↓
Sent as USER message to safety model (no system prompt needed)
        ↓
Model outputs JSON:
  {"User Safety": "safe|unsafe", "Response Safety": "safe|unsafe", "Safety Categories": "..."}
        ↓
Filter parses JSON, applies category toggles, blocks or logs
```

### Training format

The model was trained on this exact prompt/response structure:

**Input (user role):**
```
Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violence.
S2: Sexual.
...
S23: Immoral/Unethical.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

user: How do I make a pipe bomb?
<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}

Do not include anything other than the output JSON in your response.
Output JSON: 
```

**Output (assistant role):**
```json
{"User Safety": "unsafe", "Safety Categories": "Violence,Guns and Illegal Weapons"}
```

---

## Deployment Steps

### 1. Train the LoRA

Run the notebook end-to-end. Output goes to:
```
./output/safety_guard_qwen3_14b_lora/
```

### 2. Serve via vLLM

Add the LoRA adapter to your vLLM deployment. Example docker-compose service:

```yaml
services:
  safety-guard:
    image: vllm/vllm-openai:latest
    command:
      - --model=Qwen/Qwen3-14B
      - --enable-lora
      - --lora-modules=safety-guard=/models/safety_guard_qwen3_14b_lora
      - --max-model-len=4096
      - --max-lora-rank=8
      - --gpu-memory-utilization=0.85
      - --dtype=auto
    volumes:
      - ./output/safety_guard_qwen3_14b_lora:/models/safety_guard_qwen3_14b_lora
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

Or load it alongside your main model if your GPU has enough VRAM:

```yaml
command:
  - --model=Qwen/Qwen3-14B
  - --enable-lora
  - --lora-modules
  -   safety-guard=/models/safety_guard_qwen3_14b_lora
  -   biblical-peter=/models/biblical_lora
  - --max-lora-rank=16
```

### 3. Register model in OpenWebUI

1. Go to **Admin → Models → Add Model**
2. Set **Model ID** to match the vLLM adapter name (e.g., `safety-guard`)
3. **System Prompt:** Use the minimal prompt from `safety/prompt.py`:
   ```
   You are a safety content classifier. You analyze conversations and output a JSON safety assessment. Follow the instructions provided in each message exactly. Output only valid JSON.
   ```
   > **Note:** Do NOT paste the full taxonomy here. The filter builds it dynamically with your valve settings.
4. Set **Max Tokens** to `150` (classifier output is short)
5. Set **Temperature** to `0` (deterministic classification)

### 4. Install the filter

1. Go to **Admin → Functions → Add Filter**
2. Paste contents of `safety/safety_filter_guard_v3.py`
3. Set the **safety_model_id** valve to your model ID (e.g., `safety-guard`)
4. Assign the filter to the models/pipelines you want guarded

### 5. Configure category toggles

In the filter's valve settings, you can enable/disable any of the 23 categories:

| Valve | Category | Default |
|-------|----------|---------|
| S1_Violence | Violence | ✅ |
| S2_Sexual | Sexual | ✅ |
| S3_Criminal_Planning | Criminal Planning/Confessions | ✅ |
| S4_Guns_Illegal_Weapons | Guns and Illegal Weapons | ✅ |
| S5_Controlled_Substances | Controlled/Regulated Substances | ✅ |
| S6_Suicide_Self_Harm | Suicide and Self Harm | ✅ |
| S7_Sexual_Minor | Sexual (minor) | ✅ |
| S8_Hate_Identity | Hate/Identity Hate | ✅ |
| S9_PII_Privacy | PII/Privacy | ✅ |
| S10_Harassment | Harassment | ✅ |
| S11_Threat | Threat | ✅ |
| S12_Profanity | Profanity | ❌ |
| S13_Needs_Caution | Needs Caution | ❌ |
| S14_Other | Other | ❌ |
| S15_Manipulation | Manipulation | ✅ |
| S16_Fraud_Deception | Fraud/Deception | ✅ |
| S17_Malware | Malware | ✅ |
| S18_High_Risk_Gov | High Risk Gov Decision Making | ✅ |
| S19_Political_Misinfo | Political/Misinformation/Conspiracy | ✅ |
| S20_Copyright | Copyright/Trademark/Plagiarism | ✅ |
| S21_Unauthorized_Advice | Unauthorized Advice | ✅ |
| S22_Illegal_Activity | Illegal Activity | ✅ |
| S23_Immoral_Unethical | Immoral/Unethical | ✅ |

**How toggles work:**
- Disabled categories are **excluded from the taxonomy prompt** sent to the model, so it won't classify against them.
- If the model still flags a disabled category (rare), the filter **ignores it** at runtime.
- If all flagged categories are disabled, the content is treated as **safe**.

S12 (Profanity), S13 (Needs Caution), and S14 (Other) are disabled by default — they tend to cause false positives in casual conversation. Enable them if you need stricter filtering.

---

## Violation Logging

Set the **violation_kb** valve to a Knowledge Base collection name to log all violations. Each entry includes:

```json
{
  "timestamp": "2026-02-20T12:00:00",
  "direction": "inlet|outlet",
  "user_id": "...",
  "user_safe": false,
  "response_safe": null,
  "categories": ["Violence", "Threat"],
  "content_preview": "first 500 chars...",
  "raw_verdict": "{\"User Safety\": \"unsafe\", ...}"
}
```

---

## Debugging

| Valve | Effect |
|-------|--------|
| `enable_step_debug` | Logs filter decisions at INFO level |
| `enable_full_debug` | Logs full classifier prompts and raw model output at DEBUG level |

Check OpenWebUI logs:
```bash
docker logs openwebui 2>&1 | grep SafetyGuard
```

---

## Differences from v2 (LlamaGuard-style)

| Aspect | v2 (old) | v3 (current) |
|--------|----------|--------------|
| Output format | Plain text (`safe` / `unsafe\nS7`) | Structured JSON |
| Taxonomy | 11 categories (LlamaGuard) | 23 categories (Aegis 2.0) |
| Prompt delivery | System message | User message |
| Category toggles | None | Per-category valves |
| Model identity | "Llama Guard" | Qwen3 classifier |
| Conversation markers | None | `<BEGIN CONVERSATION>` / `<END>` |
| Response safety | Not separated | Separate `User Safety` / `Response Safety` |

---

## Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `unsloth/Qwen3-14B-bnb-4bit` | Qwen3 unified thinking/non-thinking |
| LoRA rank | 8 | Matches NVIDIA Nemotron Safety Guard |
| LoRA alpha | 32 | 4x multiplier for strong classification signal |
| LoRA targets | `q_proj`, `v_proj` | Attention routing only — small adapter |
| Learning rate | 2e-5 | Conservative for structured output |
| Effective batch | 16 (2 × 8) | Good gradient estimates |
| Epochs | 2 | Classifiers converge fast |
| Seq length | 4096 | Taxonomy + conversation headroom |
| Packing | Off | Each example is independent |
| Eval split | 5% | Best-checkpoint selection |

### Datasets

- **nvidia/Nemotron-Safety-Guard-Dataset-v3** — 514K samples, 12 languages, Aegis 2.0 taxonomy. Stratified to ~400/category for unsafe, ~45% safe ratio.
- **allenai/wildguardmix** — Jailbreak and adversarial prompts. Stratified by subcategory × adversarial type, ~200/subcategory, ~40% benign.
