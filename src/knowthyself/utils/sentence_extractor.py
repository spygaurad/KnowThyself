# src/knowthyself/utils/sentence_extractor.py
import re
from typing import Optional, List, Dict, Callable, Any, Tuple

from pydantic import BaseModel, validator
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# ---------------------------
# Heuristics and regex
# ---------------------------
_TASK_WORDS = {
    "attention", "attention map", "visualization", "visualisation",
    "visualize", "visualise", "map", "heatmap",
    "head", "layer", "token", "tokens",
    "bertviz", "model view", "renderer", "html", "plot", "graph",
    "provide", "show", "render", "analyze", "analyse", "display",
    "generate", "explain", "create", "give", "make", "compute",
    "draw", "highlight", "compare", "investigate", "evaluate",
    "look", "see", "craft", "build", "construct"
}
_TASK_PHRASES = {
    "provide visualization", "visualization for this", "visualisation for this",
    "attention map", "show me", "give me", "make a", "generate a view",
    "create a view", "display the", "render the", "for this", "for me", "please",
}
_QUOTE_RE = re.compile(r"[\"“”](.+?)[\"“”]")

# required terms (quoted) and layer/head
_REQ_WORD_PAT = re.compile(
    r"(?:contains|contain|with)\s+(?:word\s+)?[\"'“”]([^\"”']+)[\"'“”]",
    re.IGNORECASE,
)
_LAYER_PAT = re.compile(r"(?:\bin\s+)?\blayer\s+(\d+)\b", re.IGNORECASE)
_HEAD_PAT  = re.compile(r"(?:\bin\s+)?\bhead\s+(\d+)\b", re.IGNORECASE)
_SINGLE_QUOTED_WORD = re.compile(r"[\"'“”]([^\"”']+)[\"'“”]")

# Directive-only fragments like: "in layer 10?", "layer 12.", "head 3"
_DIRECTIVE_ONLY_PATTERNS = [
    re.compile(r"^\s*(?:in\s+)?layer\s+\d+\s*[.?!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:at\s+)?head\s+\d+\s*[.?!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:layer|head)\s*[:#]?\s*\d+\s*[.?!]?\s*$", re.IGNORECASE),
]

def _is_directive_only(s: str) -> bool:
    return any(pat.match(s.strip()) for pat in _DIRECTIVE_ONLY_PATTERNS)

# ---------------------------
# Sentence helpers
# ---------------------------
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|[\n;:]+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]

def _is_instruction_like(s: str) -> bool:
    low = s.lower()
    if any(p in low for p in _TASK_PHRASES):
        return True
    first = (low.split() or [""])[0]
    return first in _TASK_WORDS

def _is_sentence_like(s: str) -> bool:
    if not re.search(r"[A-Za-z]", s):
        return False
    words = s.strip().split()
    if len(words) < 3:
        return False
    low = s.lower()
    # reject directive-only lines outright
    if _is_directive_only(s):
        return False
    # allow presence of task words only if there's real content evidence
    if any(w in low for w in _TASK_WORDS):
        hits = sum(1 for w in _TASK_WORDS if w in low)
        has_content_clues = bool(re.search(
            r"\b[A-Z][a-z]+\b|\b\d{4}\b|\b(next week|tomorrow|today|yesterday)\b",
            low
        ))
        return hits <= 1 and has_content_clues
    return True

def _choose_best(cands: List[str]) -> Optional[str]:
    # Filter out directive-only upfront so "in layer 10?" can never win
    cands = [c for c in cands if not _is_directive_only(c)]
    def score(s: str):
        n = len(s.split())
        window_penalty = abs(20 - min(max(n, 6), 35))
        return (-int(_is_instruction_like(s)), int(_is_sentence_like(s)), -window_penalty, -n)
    cands = [c for c in cands if re.search(r"[A-Za-z]", c)]
    if not cands:
        return None
    return sorted(cands, key=score, reverse=True)[0]

def extract_sentence_exact(user_message: str) -> Optional[str]:
    """Return EXACT user sentence from mixed input; None if not found (or instruction-like/directive-only)."""
    if not (user_message or "").strip():
        return None

    # 1) quoted takes priority
    qm = _QUOTE_RE.search(user_message)
    if qm:
        q = qm.group(1).strip()
        if _is_sentence_like(q) and not _is_instruction_like(q):
            return q

    # 2) split + rank
    sentences = _split_sentences(user_message)
    cleaned = [s.strip(" \"“”'—-").strip() for s in sentences if s.strip()]
    best = _choose_best(cleaned)
    if best and not _is_instruction_like(best) and _is_sentence_like(best):
        return best
    return None

# ---------------------------
# Parameter parsing
# ---------------------------
def parse_request_params(user_message: str) -> Tuple[List[str], Optional[int], Optional[int]]:
    """
    Returns (required_terms, target_layer, target_head).
    - required_terms: words/phrases to include if we must synthesize a sentence
    - target_layer/head: 1-based indices if present, else None
    """
    if not user_message:
        return [], None, None

    text = user_message.strip()
    req_terms: List[str] = []

    # explicit contains
    for m in _REQ_WORD_PAT.finditer(text):
        term = (m.group(1) or "").strip()
        if term:
            req_terms.append(term)

    # if none, consider generic quoted tokens (short)
    if not req_terms:
        quoted = [q.strip() for q in _SINGLE_QUOTED_WORD.findall(text)]
        req_terms.extend([q for q in quoted if 1 <= len(q.split()) <= 3])

    # 1-based layer/head
    layer_idx = None
    m = _LAYER_PAT.search(text)
    if m:
        try:
            layer_idx = int(m.group(1))
        except Exception:
            layer_idx = None

    head_idx = None
    m = _HEAD_PAT.search(text)
    if m:
        try:
            head_idx = int(m.group(1))
        except Exception:
            head_idx = None

    # dedupe
    uniq_terms, seen = [], set()
    for t in req_terms:
        tt = re.sub(r"\s+", " ", t).strip()
        if tt and tt.lower() not in seen:
            uniq_terms.append(tt)
            seen.add(tt.lower())
    return uniq_terms, layer_idx, head_idx

# ---------------------------
# LLM-based synthetic sentence (hard-constrained to include required terms)
# ---------------------------
class _LLMGeneratedSentence(BaseModel):
    user_sentence: str
    @validator("user_sentence", pre=True)
    def _clean(cls, v):
        s = str(v or "").strip()
        if s and s[-1] not in ".!?":
            s += "."
        return s

def _synthesize_sentence_with_llm(required_terms: List[str], llm) -> str:
    term_list = ", ".join(f"'{t}'" for t in required_terms)
    parser = PydanticOutputParser(pydantic_object=_LLMGeneratedSentence)
    prompt = PromptTemplate(
        template=(
            "Create one realistic, natural sentence (8–22 words) that includes ALL of these words: "
            f"{term_list}.\n"
            "Rules:\n"
            "- Do NOT mention task words like attention, layer, head, tokens, visualize, plot, compute.\n"
            "- Match the user's language if obvious.\n"
            "- Return ONLY the JSON described by {format_instructions}."
        ),
        input_variables=[],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    raw = (prompt | llm).invoke({})
    text = getattr(raw, "content", raw)
    try:
        obj: _LLMGeneratedSentence = parser.parse(text)
        return obj.user_sentence
    except Exception:
        # Very safe fallback if LLM fails
        return " ".join(required_terms) + " appeared in the sky last night."

# ---------------------------
# Unified extractor
# ---------------------------
def extract_sentence_unified(
    user_message: str,
    llm_fallback_fn: Callable[[str], Dict[str, Any]],
    llm=None
) -> Dict[str, Any]:
    """
    Always parse (required_terms, layer, head). Then:
      1) Try exact sentence (verbatim).
      2) Else if required_terms present → ask LLM to synthesize sentence with them.
      3) Else → use llm_fallback_fn(user_message).
    Returns dict with:
      - user_sentence: str
      - target_layer: Optional[int]  (1-based)
      - target_head : Optional[int]  (1-based)
      - required_terms: List[str]
      - source: "exact" | "synthetic_llm" | "llm"
    """
    required_terms, target_layer, target_head = parse_request_params(user_message)

    # 1) exact sentence if present and not directive/instruction
    exact = extract_sentence_exact(user_message)
    if exact is not None:
        return {
            "user_sentence": exact,
            "target_layer": target_layer,
            "target_head": target_head,
            "required_terms": required_terms,
            "source": "exact",
        }

    # 2) LLM-based synthesis when constraints exist
    if required_terms and llm is not None:
        sentence = _synthesize_sentence_with_llm(required_terms, llm)
        return {
            "user_sentence": sentence,
            "target_layer": target_layer,
            "target_head": target_head,
            "required_terms": required_terms,
            "source": "synthetic_llm",
        }

    # 3) final fallback to caller-provided LLM generator
    out = llm_fallback_fn(user_message) or {}
    sentence = out.get("user_sentence") or user_message
    return {
        "user_sentence": sentence,
        "target_layer": target_layer,
        "target_head": target_head,
        "required_terms": required_terms,
        "source": "llm",
    }


# ---------------------------
# Model update extraction (orchestrator_model / user_model)
# ---------------------------
# Accept common model-name characters: letters, digits, / - _ . :

# Role keywords (lowercased) that map to orchestrator vs user
_ORCH_WORDS = {"orchestrator", "orchestrator_model", "supervisor", "supervisor_model"}
_USER_WORDS = {"user", "user_model"}

_MODEL_TOKEN = r"[A-Za-z0-9][A-Za-z0-9_\-./:]*[A-Za-z0-9]"
_ROLE_TO_MODEL = re.compile(
    rf"\b(?P<role>orchestrator(?:_model)?|supervisor(?:_model)?|user(?:_model)?)"
    rf"(?:\s+model)?\s*(?:to|=|:)\s*(?P<model>{_MODEL_TOKEN})\b",
    flags=re.IGNORECASE,
)
_TRAIL_PUNCT = re.compile(r"[.,;:!?]+$")

def _clean_model_token(m: str) -> str:
    m = m.strip()
    m = _TRAIL_PUNCT.sub("", m)  # drop trailing punctuation if any
    return m

def _role_key(role_word: str) -> Optional[str]:
    lw = role_word.lower()
    if lw in _ORCH_WORDS:
        return "orchestrator_model"
    if lw in _USER_WORDS:
        return "user_model"
    return None

def _normalize_lines(text: str) -> str:
    # Remove leading '#' per line and collapse whitespace
    cleaned_lines = [re.sub(r"^\s*#\s*", "", ln) for ln in (text or "").splitlines()]
    norm = " ".join(cleaned_lines)
    return re.sub(r"\s+", " ", norm).strip()

def extract_model_updates(user_message: str) -> Dict[str, Optional[str]]:
    """
    Extract model names for orchestrator_model and user_model from a free-form instruction.
    Returns:
      {
        "orchestrator_model": Optional[str],
        "user_model": Optional[str],
        "matches": List[Tuple[str, str]],   # (canonical_role, model) in order seen
        "has_updates": bool
      }
    Examples handled:
      - "Change user model to llama2:7b."
      - "Update orchestrator_model gemma3:27b and user_model llama2:7b"
      - "update orchestrator or supervisor model to gemma3."
    """
    norm_text = _normalize_lines(user_message)
    if not norm_text:
        return {"orchestrator_model": None, "user_model": None, "matches": [], "has_updates": False}

    orchestrator_model: Optional[str] = None
    user_model: Optional[str] = None
    matches: List[Tuple[str, str]] = []

    for m in _ROLE_TO_MODEL.finditer(norm_text):
        role_raw = m.group("role")
        model_raw = m.group("model")
        role = _role_key(role_raw)
        if not role:
            continue
        model = _clean_model_token(model_raw)

        if role == "orchestrator_model":
            orchestrator_model = model
        elif role == "user_model":
            user_model = model

        matches.append((role, model))

    return {
        "orchestrator_model": orchestrator_model,
        "user_model": user_model,
        "matches": matches,
        "has_updates": bool(matches),
    }

# ---------------------------
# Optional: one-call convenience that returns sentence + model updates
# ---------------------------
def extract_all(
    user_message: str,
    llm_fallback_fn: Callable[[str], Dict[str, Any]],
    llm=None,
) -> Dict[str, Any]:
    """
    Runs both:
      - extract_sentence_unified(...)  -> sentence + layer/head/required_terms
      - extract_model_updates(...)     -> orchestrator_model/user_model

    Returns a merged dict:
      {
        "user_sentence": ...,
        "target_layer": Optional[int],
        "target_head": Optional[int],
        "required_terms": List[str],
        "source": "exact" | "synthetic_llm" | "llm",
        "orchestrator_model": Optional[str],
        "user_model": Optional[str],
        "model_matches": List[Tuple[str,str]],
        "has_model_updates": bool
      }
    """
    sent = extract_sentence_unified(user_message, llm_fallback_fn=llm_fallback_fn, llm=llm)
    models = extract_model_updates(user_message)
    # merge
    out = dict(sent)
    out.update({
        "orchestrator_model": models.get("orchestrator_model"),
        "user_model": models.get("user_model"),
        "model_matches": models.get("matches", []),
        "has_model_updates": models.get("has_updates", False),
    })
    return out
