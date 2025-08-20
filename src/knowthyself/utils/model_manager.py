# src/utils/load_models.py

from langchain_community.llms import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from transformer_lens import HookedTransformer

# Correctly import config from the parent directory ('src')
from src import agent_config


class ModelManager:
    """
    Manages loading and provides access to all project models.

    Models are loaded lazily upon their first use to save memory and
    reduce initial startup time.
    """
    def __init__(self, config):
        """Initializes the manager with application configuration."""
        self._config = config
        # Internal storage for lazily loaded models
        self._orchestrator_llm = None
        self._user_model = None
        self._llama_model = None
        self._embedding_model = None

    @property
    def orchestrator_llm(self):
        """
        Loads and returns the main orchestrator LLM (Ollama or OpenAI).
        """
        if self._orchestrator_llm is None:
            print(f"INFO: Lazily loading Orchestrator LLM...")
            if self._config.DEPLOYEMENT_TYPE == "ollama":
                self._orchestrator_llm = ollama.Ollama(
                    model=self._config.ORCHESTRATOR_LLM
                )
            else:
                # Assuming the config also specifies the model for OpenAI
                self._orchestrator_llm = ChatOpenAI(
                )
        return self._orchestrator_llm

    @property
    def vision_llm(self):
        """
        Returns the vision model. In the current setup, it's an alias
        for the orchestrator LLM.
        """
        return self.orchestrator_llm

    @property
    def user_model(self):
        """
        Loads and returns the HookedTransformer model for interpretability.
        """
        if self._user_model is None:
            print(f"INFO: Lazily loading User Model (HookedTransformer)...")
            self._user_model = HookedTransformer.from_pretrained(
                self._config.GPT_USER_MODEL,
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                refactor_factored_attn_matrices=True,
            )
        return self._user_model

    @property
    def llama_model(self):
        """
        Loads and returns the Llama user model via Ollama.
        This now correctly uses the model name from your config file.
        """
        if self._llama_model is None:
            print(f"INFO: Lazily loading Llama Model...")
            self._llama_model = ollama.Ollama(
                model=self._config.LLAMA_USER_MODEL # CORRECTED: Was hardcoded
            )
        return self._llama_model

    @property
    def embedding_model(self):
        """
        Loads and returns the embedding model via Ollama.
        """
        if self._embedding_model is None:
            print(f"INFO: Lazily loading Embedding Model...")
            self._embedding_model = OllamaEmbeddings(
                model=self._config.EMBEDDING_MODEL
            )
        return self._embedding_model

# Create a single, accessible instance of the ModelManager.
# Other parts of your application can import and use this 'models' object.
models = ModelManager(agent_config)