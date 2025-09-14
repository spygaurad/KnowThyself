from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from langchain_community.llms import ollama
import os
from typing import TypedDict, Annotated, Sequence
from transformer_lens import ActivationCache, HookedTransformer
from typing import Optional
import json
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
from functools import partial


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
from IPython.display import HTML
from fastapi.responses import HTMLResponse

from langgraph.graph import StateGraph, END
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings


from src.knowthyself.utils.generate_embed import find_nearest_function, retrieve_document, load_tool_description
from src.knowthyself.tools.transformerlens.transformerlens_utils import check_next_word_prob, ablate_layers, attention_patterns, get_attention_data_for_visualizer, convert_to_base64
from src import agent_config

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState, UserInput, analyze_question
from src.knowthyself.utils.model_manager import ModelManager

from src.knowthyself.workflows.transformer_lens_workflow import transformerlens_agent
from src.knowthyself.workflows.bias_detection_workflow import bias_evaluation_agent
from src.knowthyself.workflows.bertviz_workflow import bertviz_agent
from src.knowthyself.workflows.rag_workflow import rag_agent
from src.knowthyself.workflows.load_model_workflow import load_model_agent

utils.logging.set_verbosity_error()  # Suppress standard warnings
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



def load_model_manager():
    MAIN_LLM_NAME = agent_config.ORCHESTRATOR_LLM
    # LLAMA_MODEL_NAME = agent_config.LLAMA_USER_MODEL
    USER_MODEL_NAME = agent_config.GPT_USER_MODEL
    DEPLOYEMENT_TYPE = agent_config.DEPLOYEMENT_TYPE

    model_manager = ModelManager(orchestrator_model=MAIN_LLM_NAME, orchestrator_deployment=DEPLOYEMENT_TYPE, user_model_name=USER_MODEL_NAME, embedding_model="nomic-embed-text")
    return model_manager


def create_conversation_graph(model_manager: ModelManager, embeddings):
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", partial(analyze_question, model_manager=model_manager, embeddings=embeddings))
    workflow.add_node("transformerlens_agent", partial(transformerlens_agent, model_manager=model_manager))
    workflow.add_node("bertviz_agent", partial(bertviz_agent, model_manager=model_manager))
    workflow.add_node("rag_agent", partial(rag_agent, model_manager=model_manager))
    workflow.add_node("bias_detection_agent", partial(bias_evaluation_agent, model_manager=model_manager))
    workflow.add_node("load_model", partial(load_model_agent, model_manager=model_manager))

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "bias_detection": "bias_detection_agent",
            "transformerlens": "transformerlens_agent",
            "bertviz": "bertviz_agent",
            "rag": "rag_agent",
            "load_model": "load_model",

        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("bias_detection_agent", END)
    workflow.add_edge("transformerlens_agent", END)
    workflow.add_edge("bertviz_agent", END)
    workflow.add_edge("rag_agent", END)

    return workflow.compile()

model_manager = load_model_manager()
model_manager.get_orchestrator_config()
print("Default Backend : " , model_manager.get_user_backend())
model_manager.get_user_model_name()


bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb, load_model_description_emb = load_tool_description(model_manager)
embeds = [bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb, load_model_description_emb]
graph = create_conversation_graph(model_manager, embeds)

# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)
# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"foo": ""}, config)
# # graph.invoke({"messages": "", "continue_conversation": True})