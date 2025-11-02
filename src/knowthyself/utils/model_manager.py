# model_manager.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from threading import RLock

from langchain_openai import ChatOpenAI
from langchain_community.llms import ollama as ollama_llm
from langchain_community.embeddings import OllamaEmbeddings
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer

from src import agent_config


class ModelManager:
    """
    Lazy-loading & switchable orchestrator + user model manager.

    User model backends:
      - "ollama": langchain_community.llms.ollama.Ollama(model=...)
      - "hooked": transformer_lens.HookedTransformer.from_pretrained(...)
      - "bertviz": (HFModel, HFTokenizer) with output_attentions=True
    """

    def __init__(
        self,
        # Orchestrator defaults
        orchestrator_deployment: str | None = None,
        orchestrator_model: str | None = None,
        # User model defaults (make Ollama the default backend)
        user_backend: str = "hooked",  # "ollama" | "hooked" | "bertviz"
        user_model_name: Optional[str] = None,  # fallback to agent_config.GPT_USER_MODEL or "llama3"
        # Embeddings
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self._lock = RLock()

        # --- Orchestrator config/state ---
        self._orch_deployment = orchestrator_deployment or getattr(agent_config, "DEPLOYEMENT_TYPE", "openai")
        self._orch_model = orchestrator_model or getattr(agent_config, "ORCHESTRATOR_LLM", "gpt-4o-mini")
        self._orch_handle: Optional[Any] = None

        # --- User model config/state ---
        self._user_backend = user_backend.lower()
        self._user_model_name = user_model_name or getattr(agent_config, "GPT_USER_MODEL", "llama3")
        self._user_handle: Optional[Any] = None
        self._user_extra: Dict[str, Any] = {}  # e.g., tokenizer for bertviz

        # --- Embeddings ---
        self._emb = OllamaEmbeddings(model=embedding_model)


    # =========================
    # Orchestrator
    # =========================
    def get_orchestrator(self) -> Any:
        """Return cached (or lazily created) orchestrator LLM."""
        with self._lock:
            if self._orch_handle is not None:
                return self._orch_handle

            if str(self._orch_deployment).lower() == "ollama":
                self._orch_handle = ollama_llm.Ollama(model=self._orch_model)
            else:
                self._orch_handle = ChatOpenAI(model=self._orch_model)
            return self._orch_handle

    def set_orchestrator(self, *, deployment_type: str, model_name: str) -> None:
        """Change orchestrator configuration; next get_orchestrator() reloads it."""
        with self._lock:
            self._orch_deployment = deployment_type
            self._orch_model = model_name
            self._orch_handle = None

    def reset_orchestrator(self) -> None:
        """Clear the cached orchestrator handle (keep config)."""
        with self._lock:
            self._orch_handle = None

    # =========================
    # User Model
    # =========================
    def _load_user_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Create the user-model handle(s) based on backend + name."""
        print("*"*50)
        print(self._user_model_name)
        if self._user_backend == "hooked":
            
            handle = HookedTransformer.from_pretrained(
                self._user_model_name,
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                refactor_factored_attn_matrices=True,
            )
            return handle, {}

        if self._user_backend == "ollama":
            handle = ollama_llm.Ollama(model=self._user_model_name)
            return handle, {}

        if self._user_backend == "huggingface":
            tok = AutoTokenizer.from_pretrained(self._user_model_name)
            mdl = AutoModel.from_pretrained(self._user_model_name, output_attentions=True)
            return (mdl, tok), {"model":mdl, "tokenizer": tok}

        raise ValueError(f"Unknown backend: '{self._user_backend}' (expected 'ollama' | 'hooked' | 'huggingface')")

    def get_user_model(self) -> Any:
        """
        Lazy getter for user model:
          - returns HookedTransformer for 'hooked'
          - returns Ollama LLM for 'ollama'
          - returns (HFModel, HFTokenizer) tuple for 'huggingface'
        """
        with self._lock:
            if self._user_handle is None:
                self._user_handle, self._user_extra = self._load_user_model()
            return self._user_handle

    def get_user_tokenizer(self) -> Optional[Any]:
        """For 'bertviz' backend only; returns tokenizer or None."""
        with self._lock:
            return self._user_extra.get("tokenizer")

    def set_user_model(self, *, backend: str, model_name: str) -> None:
        """Change backend/model; next get_user_model() reloads it."""
        backend = backend.lower().strip()
        if backend not in {"hooked", "ollama", "huggingface"}:
            raise ValueError("backend must be one of: 'hooked', 'ollama', 'huggingface'")
        with self._lock:
            self._user_backend = backend
            if model_name:
                self._user_model_name = model_name
            self._user_handle = None
            self._user_extra = {}

    def reset_user_model(self) -> None:
        """Clear cached user model handle (keep backend/name)."""
        with self._lock:
            self._user_handle = None
            self._user_extra = {}

    # =========================
    # Global Reset
    # =========================
    def reset_all(self) -> None:
        """Clear both orchestrator and user model handles (keep configs)."""
        with self._lock:
            self._orch_handle = None
            self._user_handle = None
            self._user_extra = {}

    # =========================
    # Embeddings
    # =========================
    def embed(self, text: str) -> list[float]:
        """Return embedding vector using Ollama embeddings."""
        return self._emb.embed_query(text)

    def get_user_model_name(self) -> str:
        """Return the configured user model name (does not load it)."""
        with self._lock:
            return self._user_model_name

    def get_user_backend(self) -> str:
        """Return the configured user backend (does not load it)."""
        with self._lock:
            return self._user_backend

    def get_orchestrator_config(self) -> tuple[str, str]:
        """Return (deployment_type, model_name) for orchestrator (does not load it)."""
        with self._lock:
            return self._orch_deployment, self._orch_model