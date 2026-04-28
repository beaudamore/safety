"""
title: Safety Guard Filter v3 (Nemotron Classifier)
author: spark
version: 3.0.0
description: >
    OpenWebUI inlet/outlet filter for Qwen3-14B Safety Guard LoRA.
    Uses Nemotron-style structured JSON classifier format (23-category Aegis 2.0 taxonomy).
    Supports per-category enable/disable via valves. Logs violations to a Knowledge Base.
    
    Compatible with: safety_guard_qwen3_14b_unsloth_4bit LoRA
    Trained on: nvidia/Nemotron-Safety-Guard-Dataset-v3 + allenai/wildguardmix
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Full 23-category Aegis 2.0 / Nemotron taxonomy ─────────────────────────
SAFETY_TAXONOMY = {
    "S1": "Violence",
    "S2": "Sexual",
    "S3": "Criminal Planning/Confessions",
    "S4": "Guns and Illegal Weapons",
    "S5": "Controlled/Regulated Substances",
    "S6": "Suicide and Self Harm",
    "S7": "Sexual (minor)",
    "S8": "Hate/Identity Hate",
    "S9": "PII/Privacy",
    "S10": "Harassment",
    "S11": "Threat",
    "S12": "Profanity",
    "S13": "Needs Caution",
    "S14": "Other",
    "S15": "Manipulation",
    "S16": "Fraud/Deception",
    "S17": "Malware",
    "S18": "High Risk Gov Decision Making",
    "S19": "Political/Misinformation/Conspiracy",
    "S20": "Copyright/Trademark/Plagiarism",
    "S21": "Unauthorized Advice",
    "S22": "Illegal Activity",
    "S23": "Immoral/Unethical",
}


class Filter:
    """
    Nemotron-style safety classifier filter for OpenWebUI.
    
    Inlet:  checks user message before it reaches the main model.
    Outlet: checks assistant response before it's shown to the user.
    """

    class Valves(BaseModel):
        # --- Model ---
        safety_model_id: str = Field(
            default="safety-guard-qwen3-14b",
            description="OpenWebUI model ID for the safety guard LoRA"
        )

        # --- Behavior ---
        block_on_unsafe: bool = Field(
            default=True,
            description="Block unsafe content (True) or just log it (False)"
        )
        check_input: bool = Field(
            default=True,
            description="Run safety check on user input (inlet)"
        )
        check_output: bool = Field(
            default=True,
            description="Run safety check on assistant output (outlet)"
        )
        unsafe_message: str = Field(
            default="I'm unable to process this request as it may involve unsafe content.",
            description="Message shown to user when content is blocked"
        )

        # --- Category toggles (all enabled by default) ---
        # Set any to False to ignore that category during classification
        S1_Violence: bool = Field(default=True)
        S2_Sexual: bool = Field(default=True)
        S3_Criminal_Planning: bool = Field(default=True)
        S4_Guns_Illegal_Weapons: bool = Field(default=True)
        S5_Controlled_Substances: bool = Field(default=True)
        S6_Suicide_Self_Harm: bool = Field(default=True)
        S7_Sexual_Minor: bool = Field(default=True)
        S8_Hate_Identity: bool = Field(default=True)
        S9_PII_Privacy: bool = Field(default=True)
        S10_Harassment: bool = Field(default=True)
        S11_Threat: bool = Field(default=True)
        S12_Profanity: bool = Field(default=False, description="Disabled by default — too aggressive for casual chat")
        S13_Needs_Caution: bool = Field(default=False, description="Disabled by default — advisory only")
        S14_Other: bool = Field(default=False, description="Disabled by default — catch-all")
        S15_Manipulation: bool = Field(default=True)
        S16_Fraud_Deception: bool = Field(default=True)
        S17_Malware: bool = Field(default=True)
        S18_High_Risk_Gov: bool = Field(default=True)
        S19_Political_Misinfo: bool = Field(default=True)
        S20_Copyright: bool = Field(default=True)
        S21_Unauthorized_Advice: bool = Field(default=True)
        S22_Illegal_Activity: bool = Field(default=True)
        S23_Immoral_Unethical: bool = Field(default=True)

        # --- Logging ---
        violation_kb: str = Field(
            default="",
            description="Knowledge Base collection name to log violations to (leave empty to disable)"
        )
        enable_full_debug: bool = Field(
            default=False,
            description="Log full prompts and responses at DEBUG level"
        )
        enable_step_debug: bool = Field(
            default=False,
            description="Log step-by-step filter processing at INFO level"
        )

    def __init__(self):
        self.valves = self.Valves()
        # Map valve field names back to taxonomy codes
        self._valve_to_code = {
            "S1_Violence": "S1",
            "S2_Sexual": "S2",
            "S3_Criminal_Planning": "S3",
            "S4_Guns_Illegal_Weapons": "S4",
            "S5_Controlled_Substances": "S5",
            "S6_Suicide_Self_Harm": "S6",
            "S7_Sexual_Minor": "S7",
            "S8_Hate_Identity": "S8",
            "S9_PII_Privacy": "S9",
            "S10_Harassment": "S10",
            "S11_Threat": "S11",
            "S12_Profanity": "S12",
            "S13_Needs_Caution": "S13",
            "S14_Other": "S14",
            "S15_Manipulation": "S15",
            "S16_Fraud_Deception": "S16",
            "S17_Malware": "S17",
            "S18_High_Risk_Gov": "S18",
            "S19_Political_Misinfo": "S19",
            "S20_Copyright": "S20",
            "S21_Unauthorized_Advice": "S21",
            "S22_Illegal_Activity": "S22",
            "S23_Immoral_Unethical": "S23",
        }

    # ─── Taxonomy builder (respects valve toggles) ───────────────────────────

    def _get_enabled_categories(self) -> dict[str, str]:
        """Return only the taxonomy categories that are enabled via valves."""
        enabled = {}
        for valve_name, code in self._valve_to_code.items():
            if getattr(self.valves, valve_name, True):
                enabled[code] = SAFETY_TAXONOMY[code]
        return enabled

    def _build_taxonomy_text(self, enabled: dict[str, str]) -> str:
        """Build the taxonomy block for the classifier prompt."""
        lines = [f"{code}: {name}." for code, name in enabled.items()]
        return "\n".join(lines)

    # ─── Classifier prompt builder ───────────────────────────────────────────

    def _build_classifier_prompt(
        self,
        user_message: str,
        agent_response: Optional[str] = None,
    ) -> str:
        """
        Build the full Nemotron-style classifier prompt.
        This is sent as the USER message to the safety model.
        
        Format matches the training data exactly:
          - Task instruction
          - <BEGIN UNSAFE CONTENT CATEGORIES> ... <END UNSAFE CONTENT CATEGORIES>
          - <BEGIN CONVERSATION> ... <END CONVERSATION>
          - Output format instruction
        """
        enabled = self._get_enabled_categories()
        taxonomy_text = self._build_taxonomy_text(enabled)

        conversation = f"\nuser: {user_message}"
        if agent_response:
            conversation += f"\nresponse: agent: {agent_response}"

        prompt = f"""Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
{taxonomy_text}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
{conversation}
<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}}

Do not include anything other than the output JSON in your response.
Output JSON: """

        return prompt

    # ─── Response parser ─────────────────────────────────────────────────────

    def _parse_safety_response(self, raw: str) -> dict:
        """
        Parse the classifier's JSON output.
        
        Expected format:
            {"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "Violence,Threat"}
        
        Returns a normalized dict with keys:
            - user_safe: bool
            - response_safe: bool | None
            - categories: list[str]
            - raw: str
        """
        result = {
            "user_safe": True,
            "response_safe": None,
            "categories": [],
            "raw": raw.strip(),
        }

        # Clean the raw output — strip markdown fences, whitespace, thinking tags
        cleaned = raw.strip()

        # Remove <think>...</think> blocks (Qwen3 thinking mode)
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the output
            match = re.search(r"\{[^{}]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse safety response JSON: {cleaned[:200]}")
                    return result
            else:
                logger.warning(f"No JSON found in safety response: {cleaned[:200]}")
                return result

        # Parse User Safety
        user_safety = data.get("User Safety", "safe")
        result["user_safe"] = str(user_safety).strip().lower() == "safe"

        # Parse Response Safety (may be absent)
        if "Response Safety" in data:
            resp_safety = data["Response Safety"]
            result["response_safe"] = str(resp_safety).strip().lower() == "safe"

        # Parse Safety Categories
        if "Safety Categories" in data:
            raw_cats = str(data["Safety Categories"])
            cats = [c.strip() for c in raw_cats.split(",") if c.strip()]
            result["categories"] = cats

        return result

    # ─── Category filtering (apply valve toggles to model output) ────────────

    def _filter_categories(self, categories: list[str]) -> list[str]:
        """
        Remove any reported categories that are disabled in valves.
        This allows the model to still classify them, but we ignore them at runtime.
        """
        enabled = self._get_enabled_categories()
        enabled_names = set(enabled.values())

        filtered = []
        for cat in categories:
            # Match by full name
            if cat in enabled_names:
                filtered.append(cat)
            else:
                # Try matching by code prefix (e.g., "S7" in "S7: Sexual (minor)")
                for code, name in enabled.items():
                    if cat.startswith(code) or name.startswith(cat):
                        filtered.append(name)
                        break
        return filtered

    # ─── Safety check (shared by inlet/outlet) ──────────────────────────────

    async def _check_safety(
        self,
        user_message: str,
        agent_response: Optional[str],
        __user__: dict,
        __request__,
    ) -> dict:
        """
        Run the safety classifier on a user message (and optionally an agent response).
        Returns parsed result dict.
        """
        classifier_prompt = self._build_classifier_prompt(user_message, agent_response)

        if self.valves.enable_full_debug:
            logger.debug(f"[SafetyGuard] Classifier prompt:\n{classifier_prompt}")

        # Call the safety model via OpenWebUI's generate_chat_completions
        from open_webui.main import generate_chat_completions
        from open_webui.utils.auth import get_current_user

        # Build the request — classifier input goes as USER message, NO system prompt
        payload = {
            "model": self.valves.safety_model_id,
            "messages": [
                {"role": "user", "content": classifier_prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.0,
            "stream": False,
        }

        try:
            response = await generate_chat_completions(
                form_data=payload,
                user=__user__,
                bypass_filter=True,
            )

            if isinstance(response, dict):
                raw_output = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            else:
                # StreamingResponse — read body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk if isinstance(chunk, bytes) else chunk.encode()
                resp_data = json.loads(body.decode())
                raw_output = (
                    resp_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
        except Exception as e:
            logger.error(f"[SafetyGuard] Model call failed: {e}")
            # Fail open — don't block if the safety model is down
            return {"user_safe": True, "response_safe": None, "categories": [], "raw": f"ERROR: {e}"}

        if self.valves.enable_full_debug:
            logger.debug(f"[SafetyGuard] Raw model output: {raw_output}")

        result = self._parse_safety_response(raw_output)

        # Apply category toggles — ignore disabled categories
        if result["categories"]:
            result["categories"] = self._filter_categories(result["categories"])
            # If all flagged categories are disabled, treat as safe
            if not result["categories"]:
                result["user_safe"] = True
                result["response_safe"] = True if result["response_safe"] is not None else None

        if self.valves.enable_step_debug:
            logger.info(f"[SafetyGuard] Result: user_safe={result['user_safe']}, "
                       f"response_safe={result['response_safe']}, "
                       f"categories={result['categories']}")

        return result

    # ─── Violation logging ───────────────────────────────────────────────────

    async def _log_violation(
        self,
        direction: str,
        content: str,
        result: dict,
        __user__: dict,
    ):
        """Log a safety violation to the configured Knowledge Base."""
        if not self.valves.violation_kb:
            return

        try:
            from open_webui.apps.rag.main import store_text_in_collection

            violation_entry = json.dumps({
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                "direction": direction,
                "user_id": __user__.get("id", "unknown"),
                "user_safe": result["user_safe"],
                "response_safe": result["response_safe"],
                "categories": result["categories"],
                "content_preview": content[:500],
                "raw_verdict": result["raw"],
            }, indent=2)

            await store_text_in_collection(
                collection_name=self.valves.violation_kb,
                text=violation_entry,
            )
            logger.info(f"[SafetyGuard] Violation logged to KB: {self.valves.violation_kb}")
        except Exception as e:
            logger.warning(f"[SafetyGuard] Failed to log violation to KB: {e}")

    # ─── Inlet (user input check) ───────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        """Check user input for safety before passing to the main model."""
        if not self.valves.check_input:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Get the latest user message
        user_msg = messages[-1].get("content", "")
        if not user_msg or not user_msg.strip():
            return body

        if self.valves.enable_step_debug:
            logger.info(f"[SafetyGuard] Inlet: checking user message ({len(user_msg)} chars)")

        result = await self._check_safety(
            user_message=user_msg,
            agent_response=None,
            __user__=__user__,
            __request__=__request__,
        )

        if not result["user_safe"]:
            cats = ", ".join(result["categories"]) if result["categories"] else "unspecified"
            logger.warning(f"[SafetyGuard] BLOCKED inlet — categories: {cats}")

            await self._log_violation("inlet", user_msg, result, __user__)

            if self.valves.block_on_unsafe:
                raise Exception(
                    f"{self.valves.unsafe_message}\n\n"
                    f"[Safety categories: {cats}]"
                )

        return body

    # ─── Outlet (assistant response check) ───────────────────────────────────

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        """Check assistant response for safety before showing to user."""
        if not self.valves.check_output:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Find the last user message and last assistant message
        user_msg = ""
        assistant_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not assistant_msg:
                assistant_msg = msg.get("content", "")
            elif msg.get("role") == "user" and not user_msg:
                user_msg = msg.get("content", "")
            if user_msg and assistant_msg:
                break

        if not assistant_msg or not assistant_msg.strip():
            return body

        if self.valves.enable_step_debug:
            logger.info(f"[SafetyGuard] Outlet: checking response ({len(assistant_msg)} chars)")

        result = await self._check_safety(
            user_message=user_msg,
            agent_response=assistant_msg,
            __user__=__user__,
            __request__=__request__,
        )

        # For outlet, check response safety specifically
        is_unsafe = False
        if result["response_safe"] is not None and not result["response_safe"]:
            is_unsafe = True
        elif not result["user_safe"]:
            # If user was unsafe, the response might still be safe (refusal)
            # Only block if response is also flagged
            is_unsafe = result["response_safe"] is not None and not result["response_safe"]

        if is_unsafe:
            cats = ", ".join(result["categories"]) if result["categories"] else "unspecified"
            logger.warning(f"[SafetyGuard] BLOCKED outlet — categories: {cats}")

            await self._log_violation("outlet", assistant_msg, result, __user__)

            if self.valves.block_on_unsafe:
                # Replace the assistant's message with the safe message
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        msg["content"] = (
                            f"{self.valves.unsafe_message}\n\n"
                            f"[Safety categories: {cats}]"
                        )
                        break

        return body
