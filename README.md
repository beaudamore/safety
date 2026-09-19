# Safety Classifier LoRAs — `training/safety/`

**Last updated:** 2026-09-18
**Purpose:** two small classifier LoRAs that back the Open WebUI safety filters: a
**content-safety** classifier (Nemotron / Aegis 2.0, 23 categories, JSON verdict) and a
**prompt-injection** classifier (`SAFE` / `INJECTION: <reason>`). Both are SFT-only; a
classifier does not get a DPO stage (`training/docs/improvements-directive.md`).

This file is the index. Verify a notebook's current content before editing it.

---

## 1. Lineage — one combined LoRA became two (2026-07-01)

| Date | What | Where |
|---|---|---|
| 2026-04-28 | First deployment guide written for the combined Qwen3-14B classifier. | `docs/SAFETY_GUARD_DEPLOYMENT.md` |
| ≤ 2026-06-08 | **Combined** era: one LoRA trained on Nemotron **plus** WildGuardMix, emitting the Nemotron JSON for everything including jailbreaks. Three notebooks (Qwen3-14B ×2, Qwen3.5-27B). | `archive/` |
| 2026-07-01 | **Split.** Content safety keeps Nemotron with the `jailbreaking` tag excluded; prompt injection gets its own notebook on Nemotron `jailbreaking` + WildGuardMix adversarial, with its own text contract. Both on Qwen3-14B. | `notebooks/*_qwen3_14b.ipynb` |
| 2026-07-07 | Filter snapshots copied into this repo. **These are stale** — see §4. | `filters/` |
| 2026-07-21 | Gemma 4 12B variants of both split notebooks (last edits 2026-08-08 / 2026-09-11). | `notebooks/*_gemma4_12b.ipynb` |
| 2026-09-12 | Live filters reworked to be block-aware (static aggregated reply, no model call on block) in the `openwebui-safety-filters` repo. Not mirrored here. | `/home/spark/projects/openwebui-safety-filters/` |
| 2026-09-18 | Live Safety Guard filter fixed: category codes matched exactly (S22 was rendering as "Sexual" via prefix match on S2). Policy filter now maps codes to names and lists every returned category. Same one-line fix applied to `filters/safety-guard-filter.py`. | this repo + live repo + Open WebUI DB |
| 2026-09-18 | **Qwen3.5 9B variants** of both split notebooks added (not yet run). | `notebooks/*_qwen35_9b.ipynb` |
| 2026-09-18 | 9B serving base switched from `QuantTrio/Qwen3.5-9B-AWQ` to **`AxionML/Qwen3.5-9B-NVFP4`** (ModelOpt NVFP4, MTP kept, same format as the 27B) for FP4 speed on GB10; `--enable-lora` added; served-model-name kept. POC decision, may change. | `compose/vllm-qwen35-9b-nvfp4-kvcache.yml` + Portainer |

The adapters on disk record the split in their metadata: `purpose: content_safety` with
`excluded_tags: ["jailbreaking"]`, and `purpose: prompt_injection` with the text contract.

---

## 2. Notebook matrix

| Task | Qwen3-14B | Gemma 4 12B | **Qwen3.5 9B** | Qwen3.8 27B |
|---|---|---|---|---|
| Content safety | `safety_guard_lora_qwen3_14b.ipynb` (2026-07-01) — adapter trained | `safety_guard_lora_gemma4_12b.ipynb` (2026-07-21) — adapter trained | `safety_guard_lora_qwen35_9b.ipynb` (2026-09-18) — **not run yet** | planned |
| Prompt injection | `prompt_injection_lora_qwen3_14b.ipynb` (2026-07-01) — adapter trained | `prompt_injection_lora_gemma4_12b.ipynb` (2026-07-21) — adapter trained | `prompt_injection_lora_qwen35_9b.ipynb` (2026-09-18) — **not run yet** | planned |

Adapters live under `output/<MODEL_NAME_BASE>/lora_adapters/` with a task metadata JSON
(`safety_guard_metadata.json` / `prompt_injection_metadata.json`). Notebooks created from
2026-09-18 on also write `complete.json` (sentinel) and freeze their curated train/eval rows to
`output/<MODEL_NAME_BASE>/train/curated_data/`.

### What the 2026-09-18 Qwen3.5 9B notebooks change

Data curation, prompts and output contracts are **copied unchanged** from the Gemma 4 12B pair.
The training skeleton follows `training/stoic/notebooks/loras/qwen3.8/stoic_qwen38_27b_sft.ipynb`,
the workspace reference for the `qwen3_5` architecture family, and the rules in
`training/docs/sft_notebook_guidelines.md` and `training/docs/multimodal_and_hybrid_base_models.md`:

- Base `Qwen/Qwen3.5-9B` (bf16, `Qwen3_5ForConditionalGeneration`) quantized to NF4 on load —
  the checkpoint the served `AxionML/Qwen3.5-9B-NVFP4` was quantized from, so adapter module
  paths match the serving base. The text-only `techwithsergiu/Qwen3.5-text-9B-bnb-4bit` used by the Stoic 9B
  notebook is `Qwen3_5ForCausalLM` with a different module layout and was not used here.
- Processor unwrapped to its tokenizer; `finetune_vision_layers=False` scoping with an
  assertion that no `mtp.*` / `visual.*` module is adapted or trainable; the saved
  `adapter_model.safetensors` is audited again after reload.
- `packing=False` (hybrid linear attention), `use_gradient_checkpointing=True` (not `"unsloth"`),
  no allocator env vars, periodic `gc`/`empty_cache` callback — all GB10 rules.
- `enable_thinking=False` in every chat-template render, with a leak check on the formatted text.
- **Loss on the assistant verdict only** (`train_on_responses_only`, mask verified). The earlier
  Qwen3-14B / Gemma 4 runs averaged the loss over the ~1,500-character repeated classifier prompt.
- Checkpoint auto-resume, `complete.json` sentinel, frozen curated rows with a manifest.
- **Section 13, evaluation (added 2026-09-18):** loads the saved adapter and scores it on the held-out
  split (Nemotron test for content safety; WildGuardMix `wildguardtest` plus Nemotron held-out
  `jailbreaking` rows for prompt injection). Prints accuracy, false-positive / false-negative rates,
  invalid-output count and per-category precision/recall; writes `eval_*.json` and a per-row JSONL
  under `output/<model>/train/`. The prompt-injection notebook was mid-training when this was added,
  so its copy lives in `notebooks/prompt_injection_lora_qwen35_9b_eval.ipynb` until the main
  notebook is next regenerated. Run it only after `complete.json` exists.
- Hyperparameters carried over per task: content safety r32/α32 on the seven attention+MLP
  projections, LR 5e-5, seq 4096, 450 unsafe/category, safe ratio 0.82; prompt injection r16/α32
  attention-only, LR 5e-5, seq 2048, caps 3500/4500/4500.

**Unverified until the first run:** that Unsloth loads `Qwen/Qwen3.5-9B` 4-bit on this container
the way it loads `unsloth/Qwen3.8-27B`. Everything else in the skeleton has run on this machine.

---

## 3. Serving (state on 2026-09-18)

Source for the 9B: `/home/spark/projects/compose/vllm-qwen35-9b-nvfp4-kvcache.yml` (confirmed
against the running container's args). Stacks are managed in Portainer; the compose file is the
reference copy.

| Server | Container / port | Base | LoRA flags today |
|---|---|---|---|
| Qwen3.5 9B, served as `qwen3.5:9b-awq-mtp` (name kept for Open WebUI compatibility) | `vllm-node-qwen35-9b-awq-kvdisk`, host :8008, vLLM v0.29.0 | **`AxionML/Qwen3.5-9B-NVFP4`** since 2026-09-18 (ModelOpt NVFP4 W4A4, group 16; lm_head / conv1d / vision / MTP kept bf16; `--language-model-only`; MTP spec-decode). Was `QuantTrio/Qwen3.5-9B-AWQ` 2026-09-10..18. | `--enable-lora --max-lora-rank 32 --max-loras 2` in the reference compose; **`--lora-modules` still to add after training** |
| Qwen3.8 27B, served as `qwen3.8:27b-nvfp4-mtp` | `vllm-node-qwen38-27b-nvfp4-kvdisk`, host :8005 | `RadixArk/Qwen3.8-27B-NVFP4` | `--enable-lora --max-lora-rank 32 --max-loras 1`, slot used by `biblical_dpo` |

Both containers mount `/home/spark/projects/training` at `/training`, so a trained adapter is
reachable without a new volume. The reference compose already carries `--enable-lora`,
`--max-lora-rank 32`, `--max-loras 2`. Once both 9B adapters exist (each notebook writes
`complete.json`), append to the 9B stack's command in Portainer:

```text
--lora-modules safety-guard=/training/safety/output/safety_guard_qwen35_9b_content_safety/lora_adapters prompt-injection=/training/safety/output/prompt_injection_qwen35_9b_detector/lora_adapters
```

vLLM refuses to start if a `--lora-modules` path does not exist yet, which is why it is not in the
compose today.

Then point the Open WebUI model entries (`safety-guard-detector`,
`prompt-safety-and-policy-violation-detector`, `prompt-injection-detector`) at the adapter names
with the minimal system prompts from `prompts/`. Today those entries point at the **bare** 9B/27B
base with a system prompt, because no adapter exists for either current base — that is what the
9B notebooks are for. Deployment steps: `docs/SAFETY_GUARD_DEPLOYMENT.md`.

**Training base vs served base.** The notebooks train on `Qwen/Qwen3.5-9B`, the bf16 checkpoint
`AxionML/Qwen3.5-9B-NVFP4` was quantized from, quantized to bitsandbytes NF4 on load. Unsloth's
4-bit training path is bitsandbytes, so the NVFP4 file itself is not trained on. Architecture,
module paths (`model.language_model.layers.*`), chat template and tokenizer config are identical
between the two (template and tokenizer_config byte-compared 2026-09-18). LoRA over a ModelOpt
NVFP4 base of this architecture under vLLM is exactly what `biblical_dpo` does on the 27B stack.

---

## 4. Filters — where the real ones are

The Open WebUI filters are **not maintained in this repo**. The live code is the `function`
table in the Open WebUI sqlite DB; the on-disk source of truth is
`/home/spark/projects/openwebui-safety-filters/`:

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
