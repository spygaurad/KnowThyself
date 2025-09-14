from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from langchain_core.messages import AIMessage
from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState
from src.knowthyself.utils.model_manager import ModelManager
from src.knowthyself.utils.sentence_extractor import extract_model_updates
from src import agent_config


@dataclass(frozen=True)
class ModelRoute:
    backend: str   # "hooked" | "huggingface" | "ollama"
    model_id: str  # exact user string


# ---------- helpers: membership checks ----------
def _user_backend_for(name: str) -> Optional[ModelRoute]:
    if not name:
        return None
    tl = set(getattr(agent_config, "TRANSFORMERLENS_SUPPORTED_MODELS", set()))
    hf = set(getattr(agent_config, "BERTVIZ_SUPPORTED_MODELS", set()))
    ol = set(getattr(agent_config, "OLLAMA_SUPPORTED_MODELS", set()))
    if name in tl:
        return ModelRoute("hooked", name)
    if name in hf:
        return ModelRoute("huggingface", name)
    if name in ol:
        return ModelRoute("ollama", name)
    return None

def _is_orchestrator_supported(name: str) -> bool:
    orch = set(getattr(agent_config, "SUPPORTED_ORCHESTRATOR_OLLAMA_MODELS", set()))
    return bool(name and name in orch)

def _mentions_orchestrator(text: str) -> bool:
    t = (text or "").lower()
    return ("orchestrator" in t) or ("supervisor" in t) or ("orchestrator_model" in t) or ("supervisor_model" in t)


# ---------- helpers: fallback scanning (only registries) ----------
def _scan_user_model(text: str) -> Optional[str]:
    if not text:
        return None
    tl = list(getattr(agent_config, "TRANSFORMERLENS_SUPPORTED_MODELS", set()))
    hf = list(getattr(agent_config, "BERTVIZ_SUPPORTED_MODELS", set()))
    ol = list(getattr(agent_config, "OLLAMA_SUPPORTED_MODELS", set()))
    cand = tl + hf + ol
    t = text.lower()
    best: Tuple[int, str] | None = None
    for name in cand:
        i = t.find(name.lower())
        if i != -1 and (best is None or i < best[0]):
            best = (i, name)
    return best[1] if best else None

def _scan_orchestrator_model(text: str) -> Optional[str]:
    if not text:
        return None
    cand = list(getattr(agent_config, "SUPPORTED_ORCHESTRATOR_OLLAMA_MODELS", set()))
    t = text.lower()
    best: Tuple[int, str] | None = None
    for name in cand:
        i = t.find(name.lower())
        if i != -1 and (best is None or i < best[0]):
            best = (i, name)
    return best[1] if best else None


# ---------- helpers: pretty formatting ----------
def _fmt_lines(items) -> str:
    s = "\n".join(f"**{m}**" for m in sorted(items)) if items else "(none configured)"
    return s


def load_model_agent(state: AgentState, model_manager: ModelManager) -> dict:
    """
    Robust model update agent:
      - Uses extract_model_updates with stricter regex (requires 'to' or '=').
      - Filters out bogus captures like the literal word 'model'.
      - If still nothing valid, scans for any known user model and defaults to user model
        (unless the text explicitly mentions orchestrator).
    """
    user_text = get_most_recent_human_message(state) or ""
    parsed = extract_model_updates(user_text)

    orch_req = parsed.get("orchestrator_model")
    user_req = parsed.get("user_model")

    print("**"*20, orch_req, user_req)
    # Filter out bad captures (e.g., literal 'model' or anything not in registries)
    if orch_req and not _is_orchestrator_supported(orch_req):
        orch_req = None
    if user_req and (_user_backend_for(user_req) is None):
        user_req = None

    # Fallbacks
    if not orch_req and not user_req:
        if _mentions_orchestrator(user_text):
            orch_req = _scan_orchestrator_model(user_text)
        else:
            user_req = _scan_user_model(user_text)

    changes, problems = [], []

    # --- orchestrator (ollama-only set) ---
    if orch_req:
        if not _mentions_orchestrator(user_text) and parsed.get("orchestrator_model") is None:
            # Default rule: bare names are user models, not orchestrator
            user_req = user_req or orch_req
        else:
            if not _is_orchestrator_supported(orch_req):
                problems.append(
                    "Unsupported orchestrator_model. Supported orchestrator (Ollama) models:\n"
                    f"{_fmt_lines(getattr(agent_config, 'SUPPORTED_ORCHESTRATOR_OLLAMA_MODELS', set()))}"
                )
            else:
                try:
                    model_manager.set_orchestrator(deployment_type="ollama", model_name=orch_req)
                    changes.append(f"orchestrator_model → `{orch_req}` (deployment: ollama)")
                except Exception as e:
                    problems.append(f"Failed to set orchestrator_model `{orch_req}`: {e}")

    # --- user model (route by backend) ---
    if user_req:
        route = _user_backend_for(user_req)
        if route is None:
            problems.append(
                "Unsupported user_model. Supported user models:\n"
                f"**TransformerLens (hooked):**\n{_fmt_lines(getattr(agent_config, 'TRANSFORMERLENS_SUPPORTED_MODELS', set()))}\n\n"
                f"**HuggingFace (BertViz):**\n{_fmt_lines(getattr(agent_config, 'BERTVIZ_SUPPORTED_MODELS', set()))}\n\n"
                f"**Ollama:**\n{_fmt_lines(getattr(agent_config, 'OLLAMA_SUPPORTED_MODELS', set()))}"
            )
        else:
            try:
                model_manager.set_user_model(backend=route.backend, model_name=route.model_id)
                changes.append(f"user_model → `{route.model_id}` (backend: {route.backend})")
            except Exception as e:
                problems.append(f"Failed to set user_model `{user_req}`: {e}")

    # Build response
    if not changes and not problems:
        msg = "No model updates detected."
    elif changes and not problems:
        msg = "Updated: " + "; ".join(changes)
    elif problems and not changes:
        msg = "Could not apply updates:\n" + "\n".join(problems)
    else:
        msg = "Updated: " + "; ".join(changes) + "\n\nIssues:\n" + "\n".join(problems)

    return {"messages": [AIMessage(content=msg, type="ai", additional_kwargs={"parse_debug": parsed})]}
