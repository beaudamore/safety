# Safety Guard — Deployment Guide

## Revision history

| Date | Change |
|---|---|
| 2026-04-28 | Written for the combined Qwen3-14B classifier (`archive/safety_guard_qwen3_14b_instruct_unsloth_4bit.ipynb`). |
| 2026-09-18 | 9B serving base switched to `AxionML/Qwen3.5-9B-NVFP4` (see `../README.md` §3 and the compose header). |
| 2026-09-18 | Updated for the split notebooks and the current bases. Notebook/filter paths corrected; Training Details table replaced with the values the split notebooks actually use (the 2026-04-28 table described r=8 / q,v-only / 2 epochs, which none of the surviving notebooks do); serving section notes the running vLLM servers' LoRA flags; filter code-matching fix recorded. |

## Overview

The Safety Guard LoRA is a **Nemotron-style content-safety classifier** that outputs structured
JSON verdicts against the 23-category Aegis 2.0 taxonomy (S1–S23). Since 2026-07-01 it is
trained **separately** from the prompt-injection classifier; see `../README.md` for the lineage
and the full notebook matrix.

**Training notebooks (content safety):**
- `../notebooks/safety_guard_lora_qwen3_14b.ipynb` — 2026-07-01, adapter trained
- `../notebooks/safety_guard_lora_gemma4_12b.ipynb` — 2026-07-21, adapter trained
- `../notebooks/safety_guard_lora_qwen35_9b.ipynb` — 2026-09-18, **not run yet**

**Open WebUI filters (live source, matches the DB as of 2026-09-18):**
- `/home/spark/projects/openwebui-safety-filters/content_safety/filter/safety_guard_filter_v3_latest.py` (v3.1.0)
- `/home/spark/projects/openwebui-safety-filters/policy_violation/filter/safety_filter_company_policy_violation_v1.py` (v1.1.0) — same classifier, adds company-policy KB context

The copies under `../filters/` and `../scripts/` are July/June 2026 snapshots and are historical.

**System prompt for the Open WebUI model entry:** `../prompts/Safety.md` (minimal; the filter builds the taxonomy).

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

Run the notebook for the base you serve, end-to-end, inside the `unsloth-notebook` container.
Output goes to `../output/<MODEL_NAME_BASE>/lora_adapters/`, for example:
```
../output/safety_guard_qwen35_9b_content_safety/lora_adapters/   # Qwen3.5 9B (2026-09-18 notebook)
../output/safety_guard_qwen3_14b_content_safety/lora_adapters/   # Qwen3-14B
```
The 2026-09-18 notebooks also write `complete.json` there when finished, and the adapter is
audited to contain only language-model tensors (no vision / MTP keys), which vLLM requires.

### 2. Serve via vLLM

Add the LoRA adapter to the vLLM server that runs the **same base** the adapter was trained on.
State of the running servers on 2026-09-18 (stacks live in Portainer; compose files on disk are reference only):

| Server | Base | LoRA flags today | Needed |
|---|---|---|---|
| `vllm-node-qwen35-9b-awq-kvdisk` (:8008, compose ref `compose/vllm-qwen35-9b-nvfp4-kvcache.yml`) | `AxionML/Qwen3.5-9B-NVFP4` since 2026-09-18 (ModelOpt NVFP4, MTP kept), `--language-model-only`, MTP spec-decode; served-model-name still `qwen3.5:9b-awq-mtp` | `--enable-lora --max-lora-rank 32 --max-loras 2` (reference compose, 2026-09-18) | after training, add `--lora-modules safety-guard=/training/safety/output/safety_guard_qwen35_9b_content_safety/lora_adapters prompt-injection=/training/safety/output/prompt_injection_qwen35_9b_detector/lora_adapters` |
| `vllm-node-qwen38-27b-nvfp4-kvdisk` (:8005) | `RadixArk/Qwen3.8-27B-NVFP4` | `--enable-lora --max-lora-rank 32 --max-loras 1` (slot used by `biblical_dpo`) | raise `--max-loras` and add the module |

Both servers already mount `/home/spark/projects/training` at `/training`, so no new volume is
needed. Edit the stack in Portainer, not the compose file. Example docker-compose service (historical, Qwen3-14B):

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
docker logs open-webui-beaudamore 2>&1 | grep SafetyGuard
```

**2026-09-18 fix:** the filter maps a returned category code to its name. Before this date it used a
prefix match, so a verdict of `S22` (Illegal Activity) was displayed as `Sexual` (matched `S2`), and
S10–S19 displayed as `Violence`. The match is now exact. If a block shows a category that does not fit
the prompt, check the `raw_verdict` in the violation log before blaming the model.

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

Values as set in the split notebooks (`../notebooks/safety_guard_lora_*.ipynb`); the Qwen3.5 9B
column is the 2026-09-18 notebook, which has not been run yet.

| Parameter | Qwen3-14B (2026-07-01) / Gemma 4 12B (2026-07-21) | Qwen3.5 9B (2026-09-18) | Rationale |
|-----------|------|------|-----------|
| Base model | `unsloth/Qwen3-14B-unsloth-bnb-4bit` / `unsloth/gemma-4-12b-it` | `Qwen/Qwen3.5-9B` (NF4 on load) | the checkpoint the served `AxionML/Qwen3.5-9B-NVFP4` was quantized from |
| LoRA rank / alpha | 32 / 32 | 32 / 32 | rank 32 = the vLLM servers' `--max-lora-rank` |
| LoRA targets | q,k,v,o + gate,up,down | same, scoped with `finetune_vision_layers=False` | excludes vision tower and MTP head, which vLLM rejects |
| Learning rate | 5e-5 | 5e-5 | conservative for structured output |
| Effective batch | 16 (2 × 8) | 16 (2 × 8) | |
| Epochs | 1 | 1 | classifiers converge fast |
| Seq length | 4096 | 4096 | taxonomy + conversation headroom |
| Packing | off | off (required: hybrid linear attention) | each example independent |
| Loss | all tokens | **assistant verdict only** | see `../README.md` |
| Eval split | 5% | 5%, eval every 100 steps | |
| Data | Nemotron v3, `jailbreaking` excluded, ≤450 unsafe/category, safe ratio 0.82 | same | content safety only; prompt injection is a separate LoRA |
| Thinking | n/a | `enable_thinking=False` in training and inference | classifier must answer immediately |

### Datasets

- **nvidia/Nemotron-Safety-Guard-Dataset-v3** — Aegis 2.0 taxonomy. English rows only, `jailbreaking` tag excluded, up to 450 unsafe rows per primary category, safe rows at 0.82× the unsafe count.
- **allenai/wildguardmix** — no longer used by the content-safety LoRA since the 2026-07-01 split. It feeds the prompt-injection LoRA (`../notebooks/prompt_injection_lora_*.ipynb`).
