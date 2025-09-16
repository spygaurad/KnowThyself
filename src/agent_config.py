#Model Deploy Type
DEPLOYEMENT_TYPE = "ollama" # Can use ChatGPT API's later on.

#Embedding Models
EMBEDDING_MODEL = 'nomic-embed-text'
DEFAULT_USER_MODEL = 'gpt2-small' #Default user model for non-ollama backends
DEFAULT_BERTVIZ_MODEL = "google-bert/bert-base-uncased"  # HuggingFace model name for BertViz workflow

# --- Toxicity eval config ---
TOXICITY_DATASET_NAME = "allenai/real-toxicity-prompts"
TOXICITY_DATASET_SPLIT = "train"
TOXICITY_PROMPT_PATH = "prompt.text"  # column path in dataset
TOXICITY_SAMPLE_SIZE = 10  

#Default Orchestrator Models
ORCHESTRATOR_LLM = 'gemma3:27b'

#Default User Models
GPT_USER_MODEL = 'gpt2-small'

# --- Supported orchestrator models ---
SUPPORTED_ORCHESTRATOR_OLLAMA_MODELS = {
    "gemma3:27b",
    "llava:34b",
}

# --- Supported user models by backend ---
# Note: these are the EXACT model names/ids as used by each backend.
TRANSFORMERLENS_SUPPORTED_MODELS = {
  "gpt2-small",
  "gpt2-medium",
  "distilbert/distilgpt2",
  "mistralai/Mistral-7B-Instruct-v0.1",
}


BERTVIZ_SUPPORTED_MODELS = {
    "microsoft/xtremedistil-l12-h384-uncased",
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/roberta-large",
    "distilbert/distilbert-base-uncased",
    "distilbert/distilroberta-base",
}

OLLAMA_SUPPORTED_MODELS = {
    "llama2:7b-chat",
    "llama2:13b-chat",
    "mistral:7b-instruct",
    "falcon3:7b",
}

#Default ollama model for bias eval
LLAMA_USER_MODEL = 'llama2:13b-chat'

# --- Regard eval config ---
REGARD_DATASET_NAME = "AlexaAI/bold"
REGARD_DATASET_SPLIT = "train"
REGARD_CATEGORY_A = "American_actors"
REGARD_CATEGORY_B = "American_actresses"
REGARD_PROMPTS_FIELD = "prompts"   # list on each row
REGARD_PROMPT_INDEX = 0            # which prompt to use from the list
REGARD_SAMPLES_PER_GROUP = 10      # ← as requested


# --- HONEST eval config ---
HONEST_DATASET_NAME = "MilaNLProc/honest"
HONEST_DATASET_CONFIG = "en_queer_nonqueer"   # e.g., "en_binary", "en_queer_nonqueer"
HONEST_DATASET_SPLIT = "honest"
HONEST_GROUP_A_PREFIX = "queer"
HONEST_GROUP_B_PREFIX = "nonqueer"
HONEST_TEMPLATE_FIELD = "template_masked"
HONEST_SUFFIXES_TO_STRIP = [" [M].", " [F]."]  # remove any of these suffixes if present
HONEST_SAMPLES_PER_GROUP = 10                  # sample size per group for tests
HONEST_METRIC_CONFIG = "en"                    # evaluate.load("honest", HONEST_METRIC_CONFIG)
