"""
Safety Guard system prompt for OpenWebUI model configuration.

IMPORTANT: This LoRA was trained WITHOUT a system prompt.
The full classifier instruction (taxonomy + conversation + output format)
is injected as the USER message by the safety_filter_guard_v3.py filter.

Set this as the model's system prompt in OpenWebUI to provide a minimal
identity context. It should NOT contain the taxonomy or output format —
the filter handles that dynamically (with per-category valve toggles).

If you set a verbose system prompt here, it will conflict with the
filter's classifier prompt and degrade classification accuracy.
"""

SAFETY_GUARD_SYSTEM_PROMPT = """You are a safety content classifier. You analyze conversations and output a JSON safety assessment. Follow the instructions provided in each message exactly. Output only valid JSON."""
