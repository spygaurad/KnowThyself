# --- NEW / REPLACED CODE BELOW ------------------------------------------------
# Add these imports (safe if duplicated)
import re
from typing import Optional, Any
from pydantic import BaseModel, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from PIL import Image
from io import BytesIO
import json

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState, UserInput
from src.knowthyself.utils.model_manager import ModelManager
import os
from src.knowthyself.tools.transformerlens.transformerlens_utils import attention_patterns, get_attention_data_for_visualizer, convert_to_base64
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.knowthyself.utils.prompts import transformerlens_extraction_prompt


# Keep your original prompt text if you like (not required), but we’ll drive strict formatting
# via PydanticOutputParser + format_instructions.
extraction_prompt = (
    "You are an information extractor and sentence creator.\n"
    "Return ONLY the JSON described by {format_instructions} with fields:\n"
    "- user_question: one original, realistic, self-contained sentence (8–22 words).\n"
    "- layer_number: numerical layer index if the user stated one; else null.\n"
    "- head_number: numerical head index if the user stated one; else null.\n\n"
    "Strict rules for user_question:\n"
    "1) Do NOT paraphrase or repeat the user's instruction. Create a NEW sentence.\n"
    "2) Do NOT mention task words: attention, head, heatmap, layer, token, visualize, plot, compute, generate, analyze.\n"
    "3) If the request includes quoted token(s), include them in a natural way; minor inflections like pluralization "
    "   or casing are allowed (e.g., 'cat and dog' → 'Cats and dogs').\n"
    "4) Match the user's language unless another language is explicitly requested.\n"
    "5) Be creative but plausible and context-appropriate; use generic names if needed.\n"
    "6) If multiple tokens are provided, incorporate all of them naturally in the same sentence.\n\n"
    "Number extraction rules:\n"
    "- Only extract layer/head indices if explicitly provided; otherwise return null.\n\n"
    "Examples (for style; do not copy):\n"
    "User: \"attention head heatmap at layer 7 for sentence with token 'cat and dog'\"\n"
    "user_question → \"Cats and dogs often quarrel, yet they can become the best of friends.\"\n"
    "layer_number → \"7\"\n"
    "head_number  → null\n\n"
    "User: \"Study attention in layer 7 for sentence with token 'she'\"\n"
    "user_question → \"Maria is on leave today; she isn't feeling well.\"\n"
    "layer_number → \"7\"\n"
    "head_number  → null\n\n"
    "User request:\n{messages}"
)


class ExtractionSchema(BaseModel):
    user_question: str
    layer_number: Optional[str] = None   # keep as str | None to match your spec
    head_number: Optional[str] = None

    # Coerce ints/"None"/"null"/"" -> proper str or None
    @validator("layer_number", "head_number", pre=True)
    def _coerce_str_or_none(cls, v: Any):
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in {"", "none", "null"}:
            return None
        if s.isdigit():
            return str(int(s))
        return s

def _backfill_layer_head(args: dict, user_text: str) -> dict:
    """Optional: if LLM missed numbers, backfill from the raw text."""
    if args.get("layer_number") is None:
        m = re.search(r"\blayer\s+(\d+)\b", user_text, flags=re.IGNORECASE)
        if m:
            args["layer_number"] = str(int(m.group(1)))
    if args.get("head_number") is None:
        m = re.search(r"\bhead\s+(\d+)\b", user_text, flags=re.IGNORECASE)
        if m:
            args["head_number"] = str(int(m.group(1)))
    return args

def extract_required_args(user_message: str, llm) -> dict:
    """
    Returns a dict with keys:
      - user_question: str
      - layer_number: Optional[str]
      - head_number : Optional[str]
    """
    parser = PydanticOutputParser(pydantic_object=ExtractionSchema)
    prompt = PromptTemplate(
        template=extraction_prompt,
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    raw = (prompt | llm).invoke({"messages": user_message})
    text = getattr(raw, "content", raw)  # handle AIMessage or string
    try:
        obj: ExtractionSchema = parser.parse(text)
        data = obj.dict()
    except Exception:
        # Very defensive fallback
        data = {"user_question": user_message, "layer_number": None, "head_number": None}

    # Optional regex backfill
    data = _backfill_layer_head(data, user_message)
    return data


def transformerlens_agent(state: AgentState, model_manager: ModelManager) -> dict:
    """
    - Uses the orchestrator LLM to parse user args (sentence, layer/head)
    - Forces user backend to HookedTransformer ('gpt2-small') for attention viz
    - Generates attention image and asks the orchestrator to explain it
    """
    MAIN_LLM = model_manager.get_orchestrator()

    # Switch to HookedTransformer backend (lazy) for attention tooling
    try:
        model_manager.set_user_model(backend="hooked", model_name="gpt2-small")
        USER_MODEL = model_manager.get_user_model()
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to load the Hooked Transformer model ('gpt2-small'). "
                            f"Ensure it’s available for TransformerLens. Error: {e}",
                    type="ai"
                )
            ]
        }

    user_text = get_most_recent_human_message(state) or ""

    args = extract_required_args(user_text, MAIN_LLM)

    # Required inputs
    text = args["user_question"]
    layer_str = args.get("layer_number")
    head_str  = args.get("head_number")
    print("*" * 100)
    print(args)
    # Convert to 0-based indices with safe defaults (layer=0, head=0 if unspecified)
    layer_idx = int(layer_str) - 1 if layer_str is not None else 0
    head_idx  = int(head_str)  - 1 if head_str  is not None else 0

    # Render attention to file
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "src", "files", "documents", "results", "attention_output.png"
    )
    gpt2_str_tokens, gpt2_attn = attention_patterns(text, USER_MODEL, layer_idx, head_idx, filepath)

    # Explain the generated attention image
    pil_image = Image.open(filepath)
    image_b64 = convert_to_base64(pil_image)
    llm_with_image_context = MAIN_LLM.bind(images=[image_b64])
    response = llm_with_image_context.invoke("Explain the attention pattern in the image")

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
# --- NEW / REPLACED CODE ABOVE ------------------------------------------------
