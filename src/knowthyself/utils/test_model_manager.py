# main_model_manager_demo.py
from model_manager import ModelManager

def show(label: str, obj) -> None:
    t = type(obj).__name__ if not isinstance(obj, tuple) else "HF tuple"
    print(f"{label}: {t}")

def main():
    mgr = ModelManager()  # defaults: orchestrator from agent_config, user backend=ollama

    print("\n=== 1) Initial loads (defaults) ===")
    orch = mgr.get_orchestrator()
    show("Orchestrator", orch)

    user = mgr.get_user_model()  # default: Ollama (model from agent_config.GPT_USER_MODEL or 'llama3')
    show("User model", user)

    vec = mgr.embed("hello world")
    print("Embedding length:", len(vec))

    print("\n=== 2) Switch USER MODEL → HookedTransformer (gpt2) ===")
    mgr.set_user_model(backend="hooked", model_name="gpt2")
    user2 = mgr.get_user_model()
    show("User model after switch", user2)

    print("\n=== 3) Switch USER MODEL → BertViz backend ===")
    mgr.set_user_model(backend="bertviz", model_name="microsoft/xtremedistil-l12-h384-uncased")
    user3 = mgr.get_user_model()  # (HFModel, HFTokenizer)
    show("User model after switch", user3)
    print("Tokenizer present:", mgr.get_user_tokenizer() is not None)

    print("\n=== 4) Reset only USER MODEL (keep backend/name) ===")
    mgr.reset_user_model()
    user4 = mgr.get_user_model()  # reloads same bertviz config
    show("User model after reset_user_model()", user4)

    print("\n=== 5) Switch ORCHESTRATOR → Ollama (llama3) and reset ===")
    mgr.set_orchestrator(deployment_type="ollama", model_name="llama3")
    mgr.reset_orchestrator()
    orch2 = mgr.get_orchestrator()
    show("Orchestrator after switch+reset", orch2)

    print("\n=== 6) Reset ALL (orchestrator + user model) ===")
    mgr.reset_all()
    orch3 = mgr.get_orchestrator()
    user5 = mgr.get_user_model()
    show("Orchestrator after reset_all()", orch3)
    show("User model after reset_all()", user5)

if __name__ == "__main__":
    main()
