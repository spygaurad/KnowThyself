# --- NEW / REPLACED CODE: Helpdesk Agent ---------------------------------------
from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState
from src.knowthyself.utils.model_manager import ModelManager
from src.knowthyself.utils.generate_embed import retrieve_document
from src import agent_config

# If your retriever lives elsewhere, adjust the import accordingly.
# Example:
# from src.knowthyself.utils.retrieve import retrieve_document
# For now, we assume retrieve_document(question: str) -> str is available in scope.

HELPDESK_PROMPT = PromptTemplate.from_template(
    """KnowYourLLM is an interactive learning platform designed to help naive users understand 
Large Language Models (LLMs) through an intuitive chatbot and visualizations. 
It breaks down LLM architecture into layers, explaining their roles and interactions 
in an easy-to-grasp manner. Users can explore model behaviors, biases, 
and response patterns through guided experiments. 
Real-time visual insights help users see how input changes affect outputs, 
improving their comprehension of LLM decision-making. 
The platform bridges the gap between technical complexity and user-friendly learning, 
making AI more accessible.

Relevant Document Content:
{context}

Question:
{question}
"""
)

def rag_agent(state: AgentState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Simple helpdesk agent:
      1) Read user's latest question from AgentState
      2) Retrieve relevant document content
      3) Use orchestrator LLM to produce an answer grounded in the retrieved context
      4) Return an AIMessage
    """

    # 1) Get orchestrator and user question
    MAIN_LLM = model_manager.get_orchestrator()
    question = (get_most_recent_human_message(state) or "").strip()
    ollama_emb = model_manager._emb
    # 2) Retrieve context (be defensive if retriever fails)
    try:
        context = retrieve_document(question, ollama_emb=ollama_emb)  # must exist in your codebase
        if context is None:
            context = ""
    except Exception as e:
        context = ""
        # You can log `e` internally if you have logging.
    print("Retrieved context:", context)

    # 3) Build and run the chain
    chain = HELPDESK_PROMPT | MAIN_LLM
    try:
        result = chain.invoke({"question": question, "context": context})
        
    except Exception as e:
        result = (
            "I couldn't generate an answer right now. "
            "Please try rephrasing your question or attempting again."
        )

    return {
        "messages": [
            AIMessage(
                content=result,
                type="ai",
            )
        ]
    }