# --- NEW / REPLACED CODE: Bias Evaluation Agent (Ollama backend) ---------------
import re
import json
import os
from typing import Optional, Any, Dict
from pydantic import BaseModel, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState
from src.knowthyself.utils.model_manager import ModelManager
from src import agent_config
from src.knowthyself.tools.bias_evaluation.honest_evaluation import evaluate_honest
from src.knowthyself.tools.bias_evaluation.regard_evaluation import evaluate_regard
from src.knowthyself.tools.bias_evaluation.toxicity_evaluation import evaluate_toxicity


bias_extraction_prompt = (
    "You are a precise information extractor. "
    "Return ONLY the JSON described by {format_instructions} with fields:\n"
    "- bias_detection_type: one of 'toxicity', 'regards', or 'honest'.\n\n"
    "Rules:\n"
    "1) Decide based on the user’s intent. Map common synonyms:\n"
    "   - toxicity: toxic, offensive, hate speech, abusive, toxicity score\n"
    "   - regards : regard, respect polarity, demographic regard, sentiment toward groups\n"
    "   - honest  : HONEST benchmark, stereotype bias, social bias measures, honesty bias\n"
    "2) If unclear, choose 'toxicity' as a safe default.\n\n"
    "User request:\n{messages}"
)

class BiasExtractionSchema(BaseModel):
    bias_detection_type: str

    @validator("bias_detection_type", pre=True)
    def _normalize_kind(cls, v: Any) -> str:
        if v is None:
            return "toxicity"
        s = str(v).strip().lower()

        # Soft matching for common synonyms and variants
        if re.search(r"honest|stereotype|social\s*bias|honesty", s):
            return "honest"
        if re.search(r"regard|demographic\s*regard|respect", s):
            return "regards"
        if re.search(r"toxic|toxicity|abusive|offensive|hate\s*speech|tox", s):
            return "toxicity"

        # If the model already gave a canonical value, keep it
        if s in {"toxicity", "regards", "honest"}:
            return s

        # Default fallback
        return "toxicity"

def extract_bias_args(user_message: str, llm) -> Dict[str, str]:
    """
    Returns: {"bias_detection_type": "toxicity"|"regards"|"honest"}
    """
    parser = PydanticOutputParser(pydantic_object=BiasExtractionSchema)
    prompt = PromptTemplate(
        template=bias_extraction_prompt,
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    raw = (prompt | llm).invoke({"messages": user_message})
    text = getattr(raw, "content", raw)  # handle AIMessage or plain string
    try:
        obj: BiasExtractionSchema = parser.parse(text)
        data = obj.dict()
    except Exception:
        data = {"bias_detection_type": "toxicity"}
    return data

# ---------------------------
# Agent
# ---------------------------
def bias_evaluation_agent(state: AgentState, model_manager: ModelManager) -> dict:
    """
    - Uses orchestrator LLM to classify the user's request into a bias type
    - Switches user backend to Ollama (default model via OLLAMA_MODEL or 'llama3:8b-instruct')
    - Runs the corresponding evaluator and asks the orchestrator to explain the scores
    """
    MAIN_LLM = model_manager.get_orchestrator()

    # Ensure user model is an Ollama-backed model (configurable)
    # ollama_model_name = model_manager.user_model_name

    # user_model_name = agent_config.LLAMA_USER_MODEL
    
    try:
        # Get the currently configured user model name (already loaded earlier)
        current_user_model_name = model_manager.get_user_model_name()

        supported_biaseval = sorted(list(getattr(agent_config, "OLLAMA_SUPPORTED_MODELS", set())))

        if current_user_model_name not in supported_biaseval:
            supported_str = "\n".join(f"- **{m}**" for m in supported_biaseval) or "(none configured)"
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"The current user model **`{current_user_model_name}`** is not supported for **BiasEval**.\n\n"
                            f"**Supported BiasEval models:**\n"
                            f"{supported_str}\n\n"
                        ),
                        type="ai",
                    )
                ]
            }

        # If supported, use the already loaded model/tokenizer

        USER_MODEL = model_manager.get_user_model()
        if not hasattr(USER_MODEL, "invoke"):
            raise RuntimeError("Loaded user model is not an Ollama LLM required for BiasEval.")
            
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to use the current user model for BiasEval. Error: {e}",
                    type="ai",
                )
            ]
        }

    user_text = get_most_recent_human_message(state) or ""
    args = extract_bias_args(user_text, MAIN_LLM)
    kind = args.get("bias_detection_type", "toxicity")

    # Map kind -> evaluator
    evaluator = {
        "toxicity": evaluate_toxicity,
        "regards":  evaluate_regard,
        "honest":   evaluate_honest,
    }.get(kind, evaluate_toxicity)

    # Run evaluation
    try:
        raw_scores = evaluator(USER_MODEL)  # expected to return a dict-like object
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Failed to run the {kind} evaluation. "
                        f"Please verify the evaluator and inputs. Error: {e}"
                    ),
                    type="ai",
                )
            ]
        }

    # Make sure it's JSON-serializable for prompting & tooling
    try:
        scores_json_str = json.dumps(raw_scores, ensure_ascii=False, indent=2)
    except Exception:
        # Fallback: coerce to string
        scores_json_str = json.dumps({"result": str(raw_scores)}, ensure_ascii=False, indent=2)

    # Ask the orchestrator to explain these results clearly for the user
    explanation_prompt = (
        "You are a helpful evaluator. The user asked for a bias analysis.\n"
        f"Bias detector used: {kind}\n"
        "Below are the raw evaluation outputs as JSON. Explain what each number means, "
        "any caveats/limitations, and how to interpret the overall score(s). "
        "Be concise, accurate, and avoid speculation beyond the data provided.\n\n"
        f"JSON:\n{scores_json_str}"
    )
    try:
        explanation_msg = MAIN_LLM.invoke(explanation_prompt)
        explanation_text = getattr(explanation_msg, "content", explanation_msg)
    except Exception as e:
        explanation_text = (
            "I computed the scores but failed to generate a natural-language explanation. "
            f"Here are the raw scores:\n{scores_json_str}\n\nError: {e}"
        )

    # Return a single AI message, including raw scores as metadata
    return {
        "messages": [
            AIMessage(
                content=explanation_text,
                additional_kwargs={
                    "bias_detection_type": kind,
                    "scores": raw_scores,
                },
                type="ai",
            )
        ]
    }
# --- NEW / REPLACED CODE: Bias Evaluation Agent (Ollama backend) ---------------
