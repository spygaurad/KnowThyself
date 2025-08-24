from __future__ import annotations
from typing import List, Optional, Tuple
import random

from datasets import load_dataset
import evaluate

from src import agent_config


# ---- Config helpers ----
def _cfg(name: str, default):
    return getattr(agent_config, name, default)

_REGARD_DATASET_NAME: str = _cfg("REGARD_DATASET_NAME", "AlexaAI/bold")
_REGARD_DATASET_SPLIT: str = _cfg("REGARD_DATASET_SPLIT", "train")
_REGARD_CATEGORY_A: str = _cfg("REGARD_CATEGORY_A", "American_actors")
_REGARD_CATEGORY_B: str = _cfg("REGARD_CATEGORY_B", "American_actresses")
_REGARD_PROMPTS_FIELD: str = _cfg("REGARD_PROMPTS_FIELD", "prompts")  # list field
_REGARD_PROMPT_INDEX: int = _cfg("REGARD_PROMPT_INDEX", 0)
_REGARD_SAMPLES_PER_GROUP: int = _cfg("REGARD_SAMPLES_PER_GROUP", 10)

_DEFAULT_INSTRUCTION: str = ""  # Pass prompts as-is by default (matches your example)


# ---- Small utils ----
def _filter_and_sample_rows(ds, category: str, k: int, seed: int) -> List[dict]:
    """Filter dataset by category and sample up to k rows."""
    rows = [row for row in ds if row.get("category") == category]
    if not rows:
        raise ValueError(f"No rows found for category '{category}'.")
    k = min(k, len(rows))
    rng = random.Random(seed)
    return rng.sample(rows, k)


def _extract_prompts(rows: List[dict], field: str, prompt_index: int) -> List[str]:
    """Extract a specific prompt (by index) from a list field on each row."""
    out: List[str] = []
    for r in rows:
        prompts = r[field]
        if not isinstance(prompts, list) or len(prompts) == 0:
            continue
        idx = min(max(prompt_index, 0), len(prompts) - 1)
        out.append(prompts[idx])
    if not out:
        raise ValueError("No prompts extracted; check REGARD_PROMPTS_FIELD and REGARD_PROMPT_INDEX.")
    return out


def _load_group_prompts_from_config(
    *,
    num_per_group: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Loads two prompt groups from config categories A and B.
    Returns (group_a_prompts, group_b_prompts).
    """
    ds = load_dataset(_REGARD_DATASET_NAME, split=_REGARD_DATASET_SPLIT)

    n = num_per_group if num_per_group is not None else _REGARD_SAMPLES_PER_GROUP

    rows_a = _filter_and_sample_rows(ds, _REGARD_CATEGORY_A, n, seed)
    rows_b = _filter_and_sample_rows(ds, _REGARD_CATEGORY_B, n, seed + 1)  # different seed for variety

    prompts_a = _extract_prompts(rows_a, _REGARD_PROMPTS_FIELD, _REGARD_PROMPT_INDEX)
    prompts_b = _extract_prompts(rows_b, _REGARD_PROMPTS_FIELD, _REGARD_PROMPT_INDEX)
    return prompts_a, prompts_b


def _generate_texts(
    llm,
    prompts: List[str],
    instruction: Optional[str] = None,
) -> List[str]:
    """
    Generates outputs using a LangChain LLM (string-in/string-out).
    Compatible with langchain_community.llms.ollama.Ollama and similar.
    """
    prefix = instruction if instruction is not None else _DEFAULT_INSTRUCTION
    outs: List[str] = []

    for p in prompts:
        inp = f"{prefix}{p}"
        try:
            out = llm.invoke(inp)  # modern interface
        except Exception:
            out = llm.predict(inp)  # fallback

        if hasattr(out, "content"):
            out = out.content
        outs.append(str(out))
    return outs


# ---- Public API ----
def evaluate_regard(
    llm,
    *,
    num_per_group: Optional[int] = None,
    seed: int = 42,
    instruction: Optional[str] = None,
    aggregation: Optional[str] = "average",  # returns average dict if set
):
    """
    Compare REGARD between two groups (A vs B) using the 'compare' config.

    Returns:
        (average_regard, pairwise_regard)
        - average_regard: dict from evaluate(..., aggregation=aggregation) if aggregation is not None, else None
        - pairwise_regard: dict from evaluate(..., without aggregation)
    """
    prompts_a, prompts_b = _load_group_prompts_from_config(
        num_per_group=num_per_group, seed=seed
    )
    texts_a = _generate_texts(llm, prompts_a, instruction=instruction)
    texts_b = _generate_texts(llm, prompts_b, instruction=instruction)

    metric = evaluate.load("regard", "compare")
    pairwise = metric.compute(data=texts_a, references=texts_b)

    average = None
    if aggregation is not None:
        average = metric.compute(data=texts_a, references=texts_b, aggregation=aggregation)

    return average, pairwise


# ---- Local test (mirrors toxicity test style) ----
def test():
    from langchain_community.llms import ollama

    model_name = getattr(agent_config, "LLAMA_MODEL_NAME",
                  getattr(agent_config, "LLAMA_USER_MODEL", "llama3:8b"))

    llm = ollama.Ollama(model=model_name)

    avg, pairwise = evaluate_regard(llm)  # uses REGARD_SAMPLES_PER_GROUP from config
    print("average_regard:", avg)
    print("pairwise_regard:", pairwise)


if __name__ == "__main__":
    test()
