from __future__ import annotations
from typing import List, Optional, Tuple
import random

from datasets import load_dataset
import evaluate

from src import agent_config


# ---- Config helpers ----
def _cfg(name: str, default):
    return getattr(agent_config, name, default)

_HONEST_DATASET_NAME: str = _cfg("HONEST_DATASET_NAME", "MilaNLProc/honest")
_HONEST_DATASET_CONFIG: Optional[str] = _cfg("HONEST_DATASET_CONFIG", "en_queer_nonqueer")
_HONEST_DATASET_SPLIT: str = _cfg("HONEST_DATASET_SPLIT", "honest")

_HONEST_GROUP_A_PREFIX: str = _cfg("HONEST_GROUP_A_PREFIX", "queer")
_HONEST_GROUP_B_PREFIX: str = _cfg("HONEST_GROUP_B_PREFIX", "nonqueer")

_HONEST_TEMPLATE_FIELD: str = _cfg("HONEST_TEMPLATE_FIELD", "template_masked")
_HONEST_SUFFIXES_TO_STRIP: List[str] = _cfg("HONEST_SUFFIXES_TO_STRIP", [" [M].", " [F]."])

_HONEST_SAMPLES_PER_GROUP: int = _cfg("HONEST_SAMPLES_PER_GROUP", 10)
_HONEST_METRIC_CONFIG: str = _cfg("HONEST_METRIC_CONFIG", "en")

_DEFAULT_INSTRUCTION: str = ""  # prompts as-is by default


# ---- Small utils ----
def _filter_and_sample_by_prefix(ds, category_prefix: str, k: int, seed: int) -> List[dict]:
    """Filter dataset rows whose 'category' startswith(category_prefix), sample up to k."""
    rows = [row for row in ds if str(row.get("category", "")).startswith(category_prefix)]
    if not rows:
        raise ValueError(f"No rows found for category prefix '{category_prefix}'.")
    k = min(k, len(rows))
    rng = random.Random(seed)
    return rng.sample(rows, k)


def _extract_prompts(rows: List[dict], field: str, suffixes_to_strip: List[str]) -> List[str]:
    """Extract prompts from field, stripping any configured suffixes."""
    out: List[str] = []
    for r in rows:
        text = r[field]
        if not isinstance(text, str):
            continue
        for sfx in suffixes_to_strip:
            if text.endswith(sfx):
                text = text[: -len(sfx)]
        out.append(text)
    if not out:
        raise ValueError(f"No prompts extracted; check field '{field}'.")
    return out


def _load_group_prompts_from_config(
    *,
    num_per_group: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Loads two prompt groups using configured dataset + category prefixes.
    Returns (group_a_prompts, group_b_prompts).
    """
    ds = load_dataset(
        _HONEST_DATASET_NAME,
        _HONEST_DATASET_CONFIG,
        split=_HONEST_DATASET_SPLIT,
    )

    n = num_per_group if num_per_group is not None else _HONEST_SAMPLES_PER_GROUP

    rows_a = _filter_and_sample_by_prefix(ds, _HONEST_GROUP_A_PREFIX, n, seed)
    rows_b = _filter_and_sample_by_prefix(ds, _HONEST_GROUP_B_PREFIX, n, seed + 1)

    prompts_a = _extract_prompts(rows_a, _HONEST_TEMPLATE_FIELD, _HONEST_SUFFIXES_TO_STRIP)
    prompts_b = _extract_prompts(rows_b, _HONEST_TEMPLATE_FIELD, _HONEST_SUFFIXES_TO_STRIP)
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
            out = llm.invoke(inp)  # modern LC interface
        except Exception:
            out = llm.predict(inp)  # fallback for older LC

        if hasattr(out, "content"):
            out = out.content
        outs.append(str(out))
    return outs


# ---- Public API ----
def evaluate_honest(
    llm,
    *,
    num_per_group: Optional[int] = None,
    seed: int = 42,
    instruction: Optional[str] = None,
):
    """
    Computes HONEST score comparing two groups (A vs B), returning the full result dict.

    Returns (example):
        {
          'honest_score_per_group': {'queer': 0.20, 'nonqueer': 0.22}
        }
    """
    prompts_a, prompts_b = _load_group_prompts_from_config(
        num_per_group=num_per_group, seed=seed
    )

    texts_a = _generate_texts(llm, prompts_a, instruction=instruction)
    texts_b = _generate_texts(llm, prompts_b, instruction=instruction)

    # HONEST expects tokenized predictions (list of tokens per output)
    preds_tokenized = [t.split() for t in texts_a] + [t.split() for t in texts_b]
    groups = ([_HONEST_GROUP_A_PREFIX] * len(texts_a)) + ([_HONEST_GROUP_B_PREFIX] * len(texts_b))

    metric = evaluate.load("honest", _HONEST_METRIC_CONFIG)
    result = metric.compute(predictions=preds_tokenized, groups=groups)
    return result


# ---- Local test (like toxicity/regard) ----
def test():
    from langchain_community.llms import ollama

    model_name = getattr(agent_config, "LLAMA_MODEL_NAME",
                  getattr(agent_config, "LLAMA_USER_MODEL", "llama3:8b"))

    llm = ollama.Ollama(model=model_name)

    result = evaluate_honest(llm)  # uses HONEST_SAMPLES_PER_GROUP from config
    print("HONEST result:", result)


if __name__ == "__main__":
    test()
