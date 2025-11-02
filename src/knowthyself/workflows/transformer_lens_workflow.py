# --- TRANSFORMERLENS AGENT (uses shared unified extractor) -------------------
import os
from typing import Dict, Any

import re
from PIL import Image
from langchain_core.messages import AIMessage

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState
from src.knowthyself.utils.model_manager import ModelManager
from src import agent_config
# your existing utilities for attention rendering
from src.knowthyself.tools.transformerlens.transformerlens_utils import (
    attention_patterns,
    convert_to_base64,
)

# NEW: shared unified extractor (exact sentence, LLM-constrained synthesis, numbers)
from src.knowthyself.utils.sentence_extractor import extract_sentence_unified


def transformerlens_agent(state: AgentState, model_manager: ModelManager) -> dict:
    """
    - Extracts a sentence + (layer/head) using the shared unified extractor:
        * exact user sentence if present (verbatim),
        * else LLM-synthesized sentence that MUST include required terms (e.g., 'meteor'),
        * else fallback LLM generator (provided by caller).
      Returns target_layer/head as 1-based (if present).
    - Loads HookedTransformer ('gpt2-small') via ModelManager for attention viz.
    - Renders attention heatmap PNG and asks orchestrator to explain it (vision input).
    """
    MAIN_LLM = model_manager.get_orchestrator()

    # Switch to HookedTransformer backend for TransformerLens
    try:
        # Get the currently configured user model name (already loaded earlier)
        current_user_model_name = model_manager.get_user_model_name()

        supported_transformerlens = sorted(list(getattr(agent_config, "TRANSFORMERLENS_SUPPORTED_MODELS", set())))

        if current_user_model_name not in supported_transformerlens:
            supported_str = "\n".join(f"- **{m}**" for m in supported_transformerlens) or "(none configured)"
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"The current user model **`{current_user_model_name}`** is not supported for **TransformerLens**.\n\n"
                            f"**Supported TransformerLens models:**\n"
                            f"{supported_str}\n\n"
                        ),
                        type="ai",
                    )
                ]
            }

        # If supported, use the already loaded model/tokenizer

        USER_MODEL = model_manager.get_user_model()
        if not hasattr(USER_MODEL, "run_with_cache"):
            raise RuntimeError(f"Loaded user model {USER_MODEL} is not a HookedTransformer required for TransformerLens.")

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to use the current user model for TransformerLens. Error: {e}",
                    type="ai",
                )
            ]
        }

    user_text = get_most_recent_human_message(state) or ""

    # Fallback LLM generator: if unified extractor can't find a sentence and has no required terms,
    # this function will be called; it MUST return {"user_sentence": "..."}.
    # Here we just ask the orchestrator to craft a plain sentence (8–22 words) with no task words.
    def _tlens_llm_fallback(u_text: str) -> Dict[str, Any]:
        prompt = (
            "Create one realistic, natural sentence (8–22 words) about an everyday situation.\n"
            "Do NOT mention task words like attention, layer, head, tokens, visualize, plot, compute.\n"
            "Return ONLY the sentence, no quotes."
        )
        raw = MAIN_LLM.invoke(prompt)
        content = getattr(raw, "content", raw)
        # Take first line, strip, ensure terminal punctuation
        sent = str(content or "").strip().splitlines()[0] if content else "The weather changed quickly over the afternoon."
        if sent and sent[-1] not in ".!?":
            sent += "."
        return {"user_sentence": sent}

    # Use the shared unified extractor (ALWAYS parses layer/head; uses LLM for constrained synthesis)
    uargs = extract_sentence_unified(
        user_message=user_text,
        llm_fallback_fn=_tlens_llm_fallback,
        llm=MAIN_LLM,  # enables constrained synthesis when required terms are present
    )

    text = uargs["user_sentence"]
    layer_1b = uargs.get("target_layer")  # 1-based or None
    head_1b  = uargs.get("target_head")   # 1-based or None

    # Convert to 0-based indices with safe defaults (layer=0, head=0)
    layer_idx = (layer_1b - 1) if isinstance(layer_1b, int) else 0
    head_idx  = (head_1b  - 1) if isinstance(head_1b, int) else 0

    # Render attention to file
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "src", "files", "documents", "results", "attention_output.png",
    )
    gpt2_str_tokens, gpt2_attn = attention_patterns(text, USER_MODEL, layer_idx, head_idx, filepath)

    # Explain the generated attention image with vision context
    pil_image = Image.open(filepath)
    image_b64 = convert_to_base64(pil_image)
    llm_with_image_context = MAIN_LLM.bind(images=[image_b64])

    explanation = llm_with_image_context.invoke(
        "The provided image shows a token–token attention map for the sentence above. "
        "Summarize the most salient attention patterns (e.g., punctuation, name/entity links, subject–verb ties), "
        "and relate them back to how the model might be using context."
    )
    explanation_text = getattr(explanation, "content", explanation)

    # Build response content (shows original request + actual sentence used)
    layer_head_str = []
    if isinstance(layer_1b, int):
        layer_head_str.append(f"Layer: {layer_1b}")
    if isinstance(head_1b, int):
        layer_head_str.append(f"Head: {head_1b}")
    layer_head_inline = (" (" + ", ".join(layer_head_str) + ")") if layer_head_str else ""

    response = (
        f"**User request**: {user_text}\n\n"
        f"**Sentence used for attention**: {text}{layer_head_inline}\n\n"
        f"{explanation_text}"
    )

    return {
        "messages": [
            AIMessage(
                content=response,
                additional_kwargs={
                    "token": gpt2_str_tokens,
                    "attention": gpt2_attn.tolist()[0],
                },
                type="ai",
            )
        ]
    }
# --- END TRANSFORMERLENS AGENT -----------------------------------------------
