# Safety Classifier LoRAs — `training/safety/`

**Last updated:** 2026-10-06
**Purpose:** two small classifier LoRAs that back the Open WebUI safety filters: a
**content-safety** classifier (Nemotron / Aegis 2.0, 23 categories, JSON verdict) and a
**prompt-injection** classifier (`SAFE` / `INJECTION: <reason>`). Both are SFT-only; a
classifier does not get a DPO stage (`training/docs/improvements-directive.md`).

Both classifiers have been trained on four bases (Qwen3-14B, Gemma 4 12B, Qwen3.5-9B,
Qwen3.8-27B). The Qwen3.8-27B pair is the one in production: attached to the DGX's vLLM
stack and called by the live filters in
[openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters).

This file is the index. Verify a notebook's current content before editing it.

---

## What it demonstrates

| Area | Specifics |
| --- | --- |
| **Classifier fine-tuning** | Two output contracts (structured JSON over a 23-category taxonomy; a two-token verdict with a reason), each trained as a LoRA on four different base families with per-task hyperparameters. |
| **Dataset curation** | Nemotron Safety Guard v3 with the `jailbreaking` tag carved out so the two classifiers do not overlap; WildGuardMix adversarial and benign rows capped per source; frozen curated rows and a manifest written next to every adapter. |
| **Training correctness** | Loss on the assistant verdict only (`train_on_responses_only`, mask verified), `enable_thinking=False` with a leak check, VLM-safe LoRA scoping with an assertion that no vision or MTP module was adapted, cold-reload audit of the saved adapter. |
| **Serving** | Adapters trained on the bf16 checkpoint and served over an NVFP4 ModelOpt quantization of the same architecture through vLLM `--enable-lora`, sharing one base with the biblical persona adapter. |
| **Production feedback loop** | Filter bugs found in production (a category-code prefix match) fixed in both the live filter repo and the snapshot here; filters reworked to be block-aware without a model call on block. |

---

## 1. Lineage — one combined LoRA became two (2026-07-01)

| Date | What | Where |
|---|---|---|
| 2026-04-28 | First deployment guide written for the combined Qwen3-14B classifier. | `docs/SAFETY_GUARD_DEPLOYMENT.md` |
| ≤ 2026-06-08 | **Combined** era: one LoRA trained on Nemotron **plus** WildGuardMix, emitting the Nemotron JSON for everything including jailbreaks. Three notebooks (Qwen3-14B ×2, Qwen3.5-27B). | `archive/` |
| 2026-07-01 | **Split.** Content safety keeps Nemotron with the `jailbreaking` tag excluded; prompt injection gets its own notebook on Nemotron `jailbreaking` + WildGuardMix adversarial, with its own text contract. Both on Qwen3-14B. | `notebooks/*_qwen3_14b.ipynb` |
| 2026-07-07 | Filter snapshots copied into this repo. **These are stale** — see §4. | `filters/` |
| 2026-07-21 | Gemma 4 12B variants of both split notebooks; both adapters trained 2026-07-22. | `notebooks/*_gemma4_12b.ipynb` |
| 2026-09-12 | Live filters reworked to be block-aware (static aggregated reply, no model call on block) in the `openwebui-safety-filters` repo. Not mirrored here. | [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) |
| 2026-09-18 | Live Safety Guard filter fixed: category codes matched exactly (S22 was rendering as "Sexual" via prefix match on S2). Policy filter now maps codes to names and lists every returned category. Same one-line fix applied to `filters/safety-guard-filter.py`. | this repo + live repo + Open WebUI DB |
| 2026-09-18 | **Qwen3.5 9B** notebooks written from the Gemma 4 pair plus the Stoic Qwen3.8 training skeleton; both adapters trained the same day. | `notebooks/*_qwen35_9b.ipynb` |
| 2026-09-19 | **Qwen3.8 27B** notebooks created from the 9B pair; both adapters trained. Standalone evaluation notebooks added for prompt injection (9B and 27B). | `notebooks/*_qwen38_27b*.ipynb` |
| 2026-09-20 | Prompt-injection notebook updated with training progress and inference results. | `notebooks/prompt_injection_lora_qwen38_27b.ipynb` |
| by 2026-10-06 | 27B adapters attached to the production Qwen3.8-27B vLLM stack as `content_safety_lora` and `prompt_injection_lora`, next to `biblical_dpo`. The separate 9B serving stack was retired (its compose files are in the compose archive). | Portainer stack, see §3 |

The adapters on disk record the split in their metadata: `purpose: content_safety` with
`excluded_tags: ["jailbreaking"]`, and `purpose: prompt_injection` with the text contract.

---

## 2. Notebook matrix and results

Steps and losses are read from each run's `trainer_state.json`. The Qwen3-14B and Gemma 4
runs averaged the loss over the whole ~1,500-character classifier prompt; the 9B and 27B
runs compute loss on the assistant verdict only, so losses are comparable within a row's
right half, not across the whole row.

| Task | Qwen3-14B | Gemma 4 12B | Qwen3.5 9B | **Qwen3.8 27B (served)** |
|---|---|---|---|---|
| Content safety | `safety_guard_lora_qwen3_14b.ipynb` — 2026-06-28, 694 steps, loss 2.25 → 0.42 | `safety_guard_lora_gemma4_12b.ipynb` — 2026-07-22, 694 steps, 0.53 → 0.049 | `safety_guard_lora_qwen35_9b.ipynb` — 2026-09-18, 694 steps, 0.64 → 0.057 | `safety_guard_lora_qwen38_27b.ipynb` — 2026-09-19, 694 steps, 0.82 → 0.068 |
| Prompt injection | `prompt_injection_lora_qwen3_14b.ipynb` — 2026-06-28, 742 steps, 2.72 → 0.73 | `prompt_injection_lora_gemma4_12b.ipynb` — 2026-07-22, 742 steps, 1.43 → 0.19 | `prompt_injection_lora_qwen35_9b.ipynb` — 2026-09-18, 742 steps, 0.49 → 0.013 | `prompt_injection_lora_qwen38_27b.ipynb` — 2026-09-19, 742 steps, 0.39 → 0.015 |

All runs are one epoch. Adapter shapes: content safety r=32/α=32 on the seven attention + MLP
projections, sequence 4096, LR 5e-5; prompt injection r=16/α=32 attention-only, sequence
2048, LR 5e-5.

Curated data (identical across the 9B and 27B runs, frozen under
`output/<MODEL_NAME_BASE>/train/curated_data/` with a manifest):

| Task | Train rows | Eval rows | Composition |
|---|---:|---:|---|
| Content safety | 11,100 | 585 | Nemotron Safety Guard v3, `jailbreaking` excluded, ≤ 450 unsafe rows per category (6,421 unsafe), safe ratio 0.82 (5,265 safe) |
| Prompt injection | 11,872 | 625 | 4,500 WildGuardMix adversarial + 3,500 Nemotron `jailbreaking` → `INJECTION`; 4,500 WildGuardMix benign → `SAFE`. Reasons: Jailbreak Technique 7,266, Privilege Escalation 644, Meta-Command Injection 90 |

Adapters live under `output/<MODEL_NAME_BASE>/lora_adapters/` with a task metadata JSON
(`safety_guard_metadata.json` / `prompt_injection_metadata.json`). Notebooks created from
2026-09-18 on also write `complete.json` (sentinel) and the frozen curated rows.

**Evaluation.** Section 13 of the 9B/27B notebooks, and the standalone
`prompt_injection_lora_qwen35_9b_eval.ipynb` / `prompt_injection_lora_qwen38_27b_eval.ipynb`,
load the saved adapter and score it on the held-out split (Nemotron test for content safety;
WildGuardMix `wildguardtest` plus Nemotron held-out `jailbreaking` rows for prompt injection),
reporting accuracy, false-positive and false-negative rates, invalid-output count and
per-category precision/recall, and writing `eval_*.json` under `output/<model>/train/`.
**No `eval_*.json` exists on disk yet**: the evaluation cells have not been run for any
adapter. Until they are, the losses above are the only measured training signal.

### What the 2026-09-18+ notebooks (9B and 27B) change

Data curation, prompts and output contracts are **copied unchanged** from the Gemma 4 12B pair.
The training skeleton follows `training/stoic/notebooks/loras/qwen3.8/stoic_qwen38_27b_sft.ipynb`,
the workspace reference for the `qwen3_5` architecture family, and the rules in
`training/docs/sft_notebook_guidelines.md` and `training/docs/multimodal_and_hybrid_base_models.md`:

- Base loaded from the bf16 checkpoint (`Qwen/Qwen3.5-9B`, `unsloth/Qwen3.8-27B`;
  `Qwen3_5ForConditionalGeneration`) and quantized to NF4 on load, so adapter module paths
  match the NVFP4 serving bases. The text-only `techwithsergiu/Qwen3.5-text-9B-bnb-4bit` used by
  the Stoic 9B notebook is `Qwen3_5ForCausalLM` with a different module layout and was not used.
- Processor unwrapped to its tokenizer; `finetune_vision_layers=False` scoping with an
  assertion that no `mtp.*` / `visual.*` module is adapted or trainable; the saved
  `adapter_model.safetensors` is audited again after reload.
- `packing=False` (hybrid linear attention), `use_gradient_checkpointing=True` (not `"unsloth"`),
  no allocator env vars, periodic `gc`/`empty_cache` callback — all GB10 rules.
- `enable_thinking=False` in every chat-template render, with a leak check on the formatted text.
- **Loss on the assistant verdict only** (`train_on_responses_only`, mask verified). The earlier
  Qwen3-14B / Gemma 4 runs averaged the loss over the repeated classifier prompt.
- Checkpoint auto-resume, `complete.json` sentinel, frozen curated rows with a manifest.
- Section 13 evaluation, described above.

The one open question when the 9B notebooks were written, whether Unsloth would load
`Qwen/Qwen3.5-9B` 4-bit on this container the way it loads `unsloth/Qwen3.8-27B`, was
settled by the 2026-09-18 run: it does.

---

## 3. Serving (state on 2026-10-06)

| Server | Container / port | Base | LoRA modules |
|---|---|---|---|
| Qwen3.8 27B, served as `qwen3.8:27b-nvfp4-mtp` | `vllm-node-qwen38-27b-nvfp4-kvdisk`, host :8005 | `RadixArk/Qwen3.8-27B-NVFP4` (ModelOpt NVFP4, MTP spec-decode) | `--enable-lora` with `biblical_dpo`, `content_safety_lora` → `safety/output/safety_guard_qwen38_27b_content_safety/lora_adapters`, `prompt_injection_lora` → `safety/output/prompt_injection_qwen38_27b_detector/lora_adapters` |
| Qwen3.5 9B (`vllm-node-qwen35-9b-*`) | retired | was `AxionML/Qwen3.5-9B-NVFP4` | The 9B adapters were trained for this stack; the stack was retired in favour of serving everything from the 27B. Compose files are kept under the compose archive. |

The container mounts `/home/spark/projects/training` at `/training`, so a trained adapter is
reachable without a new volume. The stack is managed in Portainer; the compose file in
`/home/spark/projects/compose/` is the reference copy. vLLM refuses to start if a
`--lora-modules` path does not exist, so add a module only after its `complete.json` exists.

The Open WebUI detector model entries (`safety-guard-detector`,
`prompt-safety-and-policy-violation-detector`, `prompt-injection-detector`) should point at
`content_safety_lora` and `prompt_injection_lora` with the minimal system prompts from
`prompts/`. Deployment steps: `docs/SAFETY_GUARD_DEPLOYMENT.md`.

**Training base vs served base.** The 27B notebooks train on `unsloth/Qwen3.8-27B`, the bf16
checkpoint the NVFP4 serving quantization was made from, quantized to bitsandbytes NF4 on load
(`adapter_config.json` records `unsloth/qwen3.8-27b-unsloth-bnb-4bit`). Unsloth's 4-bit training
path is bitsandbytes, so the NVFP4 file itself is not trained on. Architecture, module paths
(`model.language_model.layers.*`), chat template and tokenizer are identical between the two.
LoRA over a ModelOpt NVFP4 base of this architecture under vLLM is exactly what `biblical_dpo`
does on the same stack.

---

## 4. Filters — where the real ones are

The Open WebUI filters are **not maintained in this repo**. The live code is the `function`
table in the Open WebUI sqlite DB; the on-disk source of truth is
[openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters)
(`/home/spark/projects/openwebui-safety-filters/` locally):

| Filter | Live file (matches DB) | Copy in `training/safety/filters/` |
|---|---|---|
| Safety Guard v3 (content safety) | `content_safety/filter/safety_guard_filter_v3_latest.py` v3.1.0 | `safety-guard-filter.py` v3.0.2 — 2026-07-07 snapshot, pre-block-aware; has the 2026-09-18 exact-match fix applied but nothing else |
| Company Policy Violation | `policy_violation/filter/safety_filter_company_policy_violation_v1.py` v1.1.0 | none |
| Prompt Injection Protection v2 | `prompt_injection/filter/safety_filter_prompt_injection_v2.py` v2.1.0 | `prompt-injection-protection-filter.py` and `-v2.py` (byte-identical to each other) — 2026-07-07 snapshot, pre-block-aware |

What the 2026-09-12 block-aware rework added that the copies here lack: `FILTER_BLOCK_AWARE`,
`block_mode` (`message` vs `error`), `_record_block` / `_enforce_block_if_last` /
`_scrub_blocked_history`, the single aggregated static reply listing every filter's reason, and
(Safety Guard) removal of the `outlet` hook and `check_input` / `check_output` valves.

Treat `filters/` as historical. Edit filters in `openwebui-safety-filters/` and load them into
Open WebUI from there.

---

## 5. Other files

- `docs/SAFETY_GUARD_DEPLOYMENT.md` — deploy an adapter behind the filters (vLLM `--lora-modules`,
  Open WebUI model entry, valves). Revision history at the top.
- `docs/prompt-tests.md` — hand test prompts for the prompt-injection adapter.
- `prompts/` — the minimal system prompts the Open WebUI detector model entries should carry.
- `scripts/` — 2026-06-08 snapshots (`safety_filter_guard_v3.py` v3.0.0, `prompt.py`). Historical.
- `archive/` — the pre-split combined notebooks. Historical.

## Related

- [openwebui-safety-filters](https://github.com/beaudamore/openwebui-safety-filters) — the filters these adapters serve: content safety, policy violation, prompt injection, ClamAV antivirus
- [damore.ai: Preventing prompt injection with local LLMs](https://www.damore.ai/blog/preventing-prompt-injection-local-llms) and [Safety protection filters](https://www.damore.ai/blog/safety-protection-filters)
- [biblical](https://github.com/beaudamore/biblical) — shares the production vLLM stack and the VLM-safe training skeleton
