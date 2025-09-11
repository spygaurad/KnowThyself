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
    # MAIN_LLM = model_manager.get_orchestrator()
    # USER_MODEL = model_manager.get_user_model()
    # embedding_model = model_manager._emb
    # vision_llm = MAIN_LLM


# def preprocess(state):
#     user_input = state["messages"].lower()
#     # print("Preprocessing: ", type(user_input), user_input)
#     answer = user_input.split("/evaluate_model")[-1].split("answer:")[-1]
#     question = user_input.split("/evaluate_model")[-1].split("answer:")[0].split("question:")[-1]
#     return question, answer



# def helo_desk_answerer(question):
#     relevant_document = retrieve_document(question)
#     llm = get_llm()

#     prompt = PromptTemplate.from_template(
#         """KnowYourLLM is an interactive learning platform designed to help naive users understand 
#         Large Language Models (LLMs) through an intuitive chatbot and visualizations. 
#         It breaks down LLM architecture into layers, explaining their roles and interactions 
#         in an easy-to-grasp manner. Users can explore model behaviors, biases, 
#         and response patterns through guided experiments. 
#         Real-time visual insights help users see how input changes affect outputs, 
#         improving their comprehension of LLM decision-making. 
#         The platform bridges the gap between technical complexity and user-friendly learning, 
#         making AI more accessible. Relevant Document Content: {context} Question:{question}"""
#     )
#     chain = prompt | llm
#     response = chain.invoke({"question":question, "context":relevant_document})
#     return response

# # Creating the generic agent
# def answer_generic_question(state):

#     llm = get_llm()
#     prompt = PromptTemplate.from_template(
#         "Give a general and concise answer to the question: {input}"
#     )
#     chain = prompt | llm

#     logging.info("Generic Question Detected: Generating Answer ")
#     response = chain.invoke({"messages": state["messages"]})

#     logging.info(f"Generated Answer: {response}")

#     return {"output": response}

# def load_model(state):
#     global USER_MODEL
#     global MAIN_LLM

#     user_message = get_most_recent_human_message(state)
#     model_request = json.loads(user_message)
#     requested_orchestrator = model_request['orchestrator_model']
#     requested_user_model = model_request['user_model']
#     if requested_orchestrator!=MAIN_LLM_NAME:
#         MAIN_LLM = load_ollama(requested_orchestrator)
    
#     if requested_user_model!=USER_MODEL_NAME:
#         USER_MODEL - load_hooked_model(requested_user_model)

#     return {
#         "messages": [
#             AIMessage(
#                 # id=count,
#                 content="Succesfully Loaded Orchestrator: " +requested_orchestrator + " and User Model: " + requested_user_model,        
#                 type="ai"        

#                 )
#             ]
#         }


# def test_transformerlens():
#     # Validate model
#     print("Running TransformerLens")
#     response = check_next_word_prob(USER_MODEL, "", "")
#     # print(response)
#     # return {"model_name":state["messages"], "output": response}
#     return "\n".join(response)

# def _html_from_ipy(obj: HTML) -> str:
#     if getattr(obj, "data", None) is not None:
#         return obj.data
#     if hasattr(obj, "_repr_html_"):
#         return obj._repr_html_() or ""
#     return str(obj)


# def write_model_view_html(html: str, filename: str = "model_view.html") -> str:
#     """Atomically write HTML to viz_static/model_view.html and return its public URL."""

#     BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "..", 'src', 'files', 'documents')


#     # A secure mapping of front-end tab names to their actual server-side directory paths.
#     # This prevents malicious users from requesting arbitrary file paths.
#     ALLOWED_FOLDERS = {
#         "documentation": os.path.join(BASE_DIR, "documentation"),
#         "workflows": os.path.join(BASE_DIR, "workflows"),
#         "tools": os.path.join(BASE_DIR, "tools"),
#         "results": os.path.join(BASE_DIR, "results"),

#         # Add any other folders you want to expose if they exist under src/files/documents/
#     }

#     VIZ_DIR = Path(os.path.join(BASE_DIR, "results"))
#     # VIZ_DIR.mkdir(exist_ok=True)

#     target = VIZ_DIR / filename
#     tmp = VIZ_DIR / (filename + ".tmp")
#     tmp.write_text(html, encoding="utf-8")
#     tmp.replace(target)  # atomic on same filesystem
#     # add a cache-buster query so the browser doesn’t reuse stale content
#     return True

# def bertviz_agent(state):
#     model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
#     input_text = "The cat sat on the mat"  
#     model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
#     outputs = model(inputs)  # Run model
#     attention = outputs[-1]  # Retrieve attention from model outputs
#     tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
#     model_view_page = model_view(attention, tokens, html_action="return")  # Display model view
#     html_str = _html_from_ipy(model_view_page)
#     file_saved_status = write_model_view_html(html_str, filename="model_view.html")
    
#     if file_saved_status:
#         content = "Sorry"
        

#     return {
#             "messages": [
#                 AIMessage(
#                     # id=count,
#                     # content="I am temporarily unable to serve Image based responses. \n However if you have access to logs, you can view the pattern there.",
#                     content = "Bertviz Visualizations",
#                     # timestamp= "2025-08-19T17:25:00.000Z",
#                     additional_kwargs={
#                             "bert_viz_view": "model_view.html",
#                         },
#                     type="ai"                 
#                 )
#             ]
#     }


# def understand_attention(state):
#     response = "I see the relationship between words and help you understand it."
#     return {"output": response}

# def get_layer_importance(state):
#     response = "I can help you answer indepth model related questions like the value of attention in 16th head of layer 12"
#     return {"output": response}

# #You can precise the format here which could be helpfull for multimodal graphs


# def process_question(state: UserInput):
#     graph = create_graph()
#     result = graph.invoke({"messages": state["messages"]})
#     # logging.info("\n--- Final answer ---")
#     logging.info(f"{result['output']}")
#     return state


# def next_word_prob(state:AgentState):
#     # global count
#     user_message = get_most_recent_human_message(state)
#     # response = check_next_word_prob()
#     response = check_next_word_prob(USER_MODEL, "", "")

#     return {
#             "messages": [
#                 AIMessage(
#                     # id=count,
#                     content=response,
#                 )
#             ]
#     }


# def global_interpretation(state:AgentState):
#     user_message = get_most_recent_human_message(state)
#     # response = ablate_layers(USER_MODEL, "","","")
#     from datasets import load_dataset

#     toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

#     return {
#             "messages": [
#                 AIMessage(
#                     # id=count,
#                     content=response,
#                     type="ai"
#                 )
#             ]
#     }


# def help_desk(state:AgentState):
#     # global count
#     user_message = get_most_recent_human_message(state)
#     response = helo_desk_answerer(user_message)
#     print("You are reaching out to help desk agent", " ", response)

#     return {
#             "messages": [
#                 AIMessage(
#                     # id=count,
#                     content=response,
#                     type="ai"  
#                 )
#             ]
#     }

def load_model(state):
    pass

def create_conversation_graph(model_manager: ModelManager, embeddings):
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", partial(analyze_question, model_manager=model_manager, embeddings=embeddings))
    workflow.add_node("transformerlens_agent", partial(transformerlens_agent, model_manager=model_manager))
    workflow.add_node("bertviz_agent", partial(bertviz_agent, model_manager=model_manager))
    workflow.add_node("rag_agent", partial(rag_agent, model_manager=model_manager))
    workflow.add_node("bias_detection_agent", partial(bias_evaluation_agent, model_manager=model_manager))
    workflow.add_node("load_model", load_model)

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

bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb, load_model_description_emb = load_tool_description(model_manager)
embeds = [bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb, load_model_description_emb]
graph = create_conversation_graph(model_manager, embeds)

# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)
# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"foo": ""}, config)
# # graph.invoke({"messages": "", "continue_conversation": True})