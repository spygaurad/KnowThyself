# --- CLEANED CODE: BertViz Agent (HuggingFace backend) ------------------
import os
import re
from typing import Any, Dict

from pydantic import BaseModel, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState
from src.knowthyself.utils.model_manager import ModelManager
from src import agent_config

from bertviz import model_view
try:
    from bertviz.util import _html_from_ipy  # type: ignore
except Exception:
    _html_from_ipy = None

# ---------------------------
# Sentence extraction prompt
# ---------------------------
bertviz_extraction_prompt = (
    "You are a sentence selector/creator.\n"
    "Return ONLY the JSON described by {format_instructions} with fields:\n"
    "- user_sentence: one natural sentence (8–30 words) to analyze.\n\n"
    "Rules for user_sentence:\n"
    "1) If the user provides a sentence, use it as-is (light cleanup allowed).\n"
    "2) If not provided, create a plausible sentence matching the user's language.\n"
    "3) Avoid task words like: attention, visualize, head, layer, tokens.\n\n"
    "User request:\n{messages}"
)

class SentenceExtractionSchema(BaseModel):
    user_sentence: str

    @validator("user_sentence", pre=True)
    def _normalize_sentence(cls, v: Any) -> str:
        s = (str(v or "")).strip()
        s = re.sub(r"\s+", " ", s)
        if s and s[-1] not in ".?!":
            s += "."
        return s

def extract_sentence_arg(user_message: str, llm) -> Dict[str, str]:
    parser = PydanticOutputParser(pydantic_object=SentenceExtractionSchema)
    prompt = PromptTemplate(
        template=bertviz_extraction_prompt,
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    raw = (prompt | llm).invoke({"messages": user_message})
    text = getattr(raw, "content", raw)
    try:
        obj: SentenceExtractionSchema = parser.parse(text)
        return obj.dict()
    except Exception:
        cleaned = re.sub(r"\s+", " ", user_message).strip()
        if not cleaned:
            cleaned = "Please provide a valid input sentence."
        return {"user_sentence": cleaned}

def _save_model_view_html(attention, tokens, out_path: str) -> str:
    page = model_view(attention, tokens, html_action="return")
    if _html_from_ipy is not None:
        html_str = _html_from_ipy(page)
    else:
        html_str = str(getattr(page, "data", page))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return out_path

# ---------------------------
# Agent
# ---------------------------
def bertviz_agent(state: AgentState, model_manager: ModelManager) -> dict:
    """
    - Extracts a clean input sentence with orchestrator LLM
    - Ensures HuggingFace user model is loaded
    - Runs forward pass with attentions and renders BertViz HTML
    - Returns path to HTML inside AIMessage.additional_kwargs
    """
    MAIN_LLM = model_manager.get_orchestrator()
    user_model_name = agent_config.DEFAULT_BERTVIZ_MODEL

    try:
        model_manager.set_user_model(backend="huggingface", model_name=user_model_name)
        
        USER_MODEL = model_manager.get_user_model()
        if isinstance(USER_MODEL, (tuple, list)) and len(USER_MODEL) >= 2:
            hf_model, hf_tokenizer = USER_MODEL[0], USER_MODEL[1]
        # hf_model, hf_tokenizer = USER_MODEL["model"], USER_MODEL["tokenizer"]
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to load HuggingFace model '{user_model_name}'. Error: {e}. Loading Default model: {agent_config.DEFAULT_BERTVIZ_MODEL}",
                    type="ai",
                )
            ]
        }

    user_text = get_most_recent_human_message(state) or ""
    args = extract_sentence_arg(user_text, MAIN_LLM)
    sentence = args["user_sentence"]

    try:
        inputs = hf_tokenizer.encode(sentence, return_tensors="pt")
        outputs = hf_model(inputs, output_attentions=True)
        attention = getattr(outputs, "attentions", None) or getattr(outputs, "encoder_attentions", None)
        if attention is None:
            raise RuntimeError("Model outputs did not contain attention tensors.")
        tokens = hf_tokenizer.convert_ids_to_tokens(inputs[0])
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to compute attentions. Error: {e}",
                    type="ai",
                )
            ]
        }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "src", "files", "documents", "results", "bertviz_model_view.html"
    )
    try:
        saved_path = _save_model_view_html(attention, tokens, out_path)
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Computed attentions, but failed to render/save BertViz HTML. Error: {e}",
                    type="ai",
                )
            ]
        }

    try:
        guide = MAIN_LLM.invoke(
            "Provide a short (2–3 sentence) guide on interpreting the BertViz model view. "
            "Explain what the attention grids and heads represent, simply."
        )
        guide_text = getattr(guide, "content", guide)
    except Exception:
        guide_text = "BertViz model view generated. Each grid cell shows head-level attention across layers."

    return {
        "messages": [
            AIMessage(
                content=f"BertViz visualization generated for: “{sentence}”\n\nModel: {user_model_name}\n\n{guide_text}",
                additional_kwargs={"bert_viz_view": saved_path},
                type="ai",
            )
        ]
    }