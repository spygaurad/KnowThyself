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


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, END
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

from src.demo.generate_embed import find_nearest_function, retrieve_document
from src.transformer_lens_utils.indirect_obj_identification import check_next_word_prob, ablate_layers, attention_patterns, get_attention_data_for_visualizer

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Creating the first analysis agent to check the prompt structure
# This print part helps you to trace the graph decisions

model = "ollama" # chatgpt
if model == "ollama":
    MAIN_LLM = ollama.Ollama(
                            # base_url=ollama_base_url, 
                            # model='mapler/gpt2',
                            # model = 'llama3.1:70b'
                            model = 'gemma3:27b'

                            )
else:
    MAIN_LLM = ChatOpenAI()

# vision_llm = ollama.Ollama(
#                         # base_url=ollama_base_url, 
#                         # model='mapler/gpt2',
#                         model = 'gemma3:27b'
#                         )
vision_llm = MAIN_LLM

# USER_MODEL = None
# global USER_MODEL  # Access the global variable
USER_MODEL = HookedTransformer.from_pretrained(
                "gpt2-small",
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                refactor_factored_attn_matrices=True,
            )

Llama_model = ollama.Ollama(
                            # base_url=ollama_base_url, 
                            # model='mapler/gpt2',
                            # model = 'llama3.1:70b'
                            model = 'llama2:13b-chat'

                            )

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)

def get_llm():
    return MAIN_LLM


count  = 0
def analyze_question(state):
    global count
    count += 1
    user_message = get_most_recent_human_message(state)
    try:
        tool_request = json.loads(user_message)
        print("Tool Request: ", tool_request)
        return {"decision": "visualization_tool_router"}
    except:
        print("User Message: ##### ", user_message)
        decision = find_nearest_function(user_message)

        print("Analysing... Made Decision: ", decision)
        return {"decision": decision}
    # return {"decision": decision, "messages": state["messages"]}

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Creating the code agent that could be way more technical
def answer_code_question(state):

    prompt = PromptTemplate.from_template(
        "Provide short answer to given question : {input}"
    )
    chain = prompt | USER_MODEL
    response = chain.invoke({"messages": state["messages"]})
    # logging.info(f"User Model Answered: {response}")
    logging.info(f"User Model Answered: \n")

    return {"output": response}

def preprocess(state):
    user_input = state["messages"].lower()
    # print("Preprocessing: ", type(user_input), user_input)
    answer = user_input.split("/evaluate_model")[-1].split("answer:")[-1]
    question = user_input.split("/evaluate_model")[-1].split("answer:")[0].split("question:")[-1]
    return question, answer

def evaluate_answer(state):
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        "This is the question asked: \n QUESTION: \n {question} \n This is the answer provided by user: \n USER ANSWER: \n{input}. \n Your task is to evaluate the answer and provide a score between 0 and 10 with a short and simple explanation. "
    )
    chain = prompt | llm

    question,answer = preprocess(state)
    response = chain.invoke({"question":question,"messages": answer})
    # logging.info(f"Evaluation: \n {response}")
    logging.info(f"Evaluation: \n")

    return {"output": response}

def extract_in_json(question, template):
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        "From the give question,  {question} \n extract information into the given template {template}"
    )
    chain = prompt | llm
    response = chain.invoke({"question":question,"template": template})
    return response

def helo_desk_answerer(question):
    relevant_document = retrieve_document(question)
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """KnowYourLLM is an interactive learning platform designed to help naive users understand 
        Large Language Models (LLMs) through an intuitive chatbot and visualizations. 
        It breaks down LLM architecture into layers, explaining their roles and interactions 
        in an easy-to-grasp manner. Users can explore model behaviors, biases, 
        and response patterns through guided experiments. 
        Real-time visual insights help users see how input changes affect outputs, 
        improving their comprehension of LLM decision-making. 
        The platform bridges the gap between technical complexity and user-friendly learning, 
        making AI more accessible. Relevant Document Content: {context} Question:{question}"""
    )
    chain = prompt | llm
    response = chain.invoke({"question":question, "context":relevant_document})
    return response
# Creating the generic agent
def answer_generic_question(state):

    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain = prompt | llm

    logging.info("Generic Question Detected: Generating Answer ")
    response = chain.invoke({"messages": state["messages"]})

    logging.info(f"Generated Answer: {response}")

    return {"output": response}

def load_model(state):
    response = "I support following models: \n 1. LLAMA \n 2. Mistral \n 3. GPT \n\n Provide me with the huggingface repo and I will help you understand your model."
    print(response)
    return {"output": response}

def validate_and_load_model(state):
    global USER_MODEL  # Access the global variable
    model_name = state["messages"]

    USER_MODEL = ollama.Ollama(model=model_name) #'llama2:7b'
    
    try:
        answer_code_question(state)
        response = "Succesfully loaded "+ model_name +". You can ask questions to your model now."

    except:
        model_name = ""
        USER_MODEL = None
        response = "Invalid Model. Please provide me with Huggingface Model"
    # Validate model

    return {"model_name":state["messages"], "output": response}

def test_transformerlens():

    
    # Validate model
    print("Running TransformerLens")
    response = check_next_word_prob(USER_MODEL, "", "")
    # print(response)
    # return {"model_name":state["messages"], "output": response}
    return "\n".join(response)



def understand_attention(state):
    response = "I see the relationship between words and help you understand it."
    return {"output": response}

def get_layer_importance(state):
    response = "I can help you answer indepth model related questions like the value of attention in 16th head of layer 12"
    return {"output": response}

#You can precise the format here which could be helpfull for multimodal graphs
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # messages: Annotated[Sequence[BaseMessage]]
    input: str
    output: str
    model_name : str
    decision: str
    # interaction_id: str
    

class UserInput(TypedDict):
    input: str
    continue_conversation: bool

# Assuming messages contain a sender attribute to distinguish human and system messages
def get_most_recent_human_message(state: AgentState) -> Optional[str]:
    # Iterate over the messages in reverse order and return the first HumanMessage
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):  # Check if the message is a HumanMessage
            return msg.content
    return None  # Return None if no HumanMessage is found

def generate_interaction_id() -> str:
    return str(uuid4())

def get_user_input(state: UserInput) -> UserInput:
    user_input = input("\nUser (ou 'q' to quit) : ")
    return {
        "messages": user_input,
        "continue_conversation": user_input.lower() != 'q'
    }

def process_question(state: UserInput):
    graph = create_graph()
    result = graph.invoke({"messages": state["messages"]})
    # logging.info("\n--- Final answer ---")
    logging.info(f"{result['output']}")
    return state


def next_word_prob(state:AgentState):
    global count
    user_message = get_most_recent_human_message(state)
    # response = check_next_word_prob()
    response = check_next_word_prob(USER_MODEL, "", "")

    return {
            "messages": [
                AIMessage(
                    id=count,
                    content=response,
                )
            ]
    }

def tool_use_response(state:AgentState):
    global count

    user_message = get_most_recent_human_message(state)
    tool_request = json.loads(user_message)

    model, layer, head, text = tool_request["model"], int(tool_request["layer"]), int(tool_request["head"]), tool_request["text"]
    layer -=1
    head -= 1

    print("user uploaded model: ", model)
    if "model"=="gpt2":
        model = USER_MODEL
    gpt2_str_tokens, gpt2_attn = attention_patterns(text, USER_MODEL, layer, head)
    gpt_2_attn_all = get_attention_data_for_visualizer(text=text, model=USER_MODEL)

    file_path = "/home/prasais/projects/KnowYourLLM/attention_output.png"
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)
    llm_with_image_context = vision_llm.bind(images=[image_b64])
    response = llm_with_image_context.invoke("Explain the attention pattern in the image")
    # response = "Please play with the below visualization: \n"
    print("*"*200)
    print(gpt2_str_tokens)
    print(gpt2_attn.tolist()[0])

    return {
            "messages": [
                AIMessage(
                    id=count,
                    # content="I am temporarily unable to serve Image based responses. \n However if you have access to logs, you can view the pattern there.",
                    content = response,
                    timestamp= "2025-08-19T17:25:00.000Z",
                    additional_kwargs={
                            "token": gpt2_str_tokens,
                            "attention": gpt2_attn.tolist()[0],
                            "bert_attention":gpt_2_attn_all,
                            "is_type_attention": True  # Flag to indicate this message has an attention matrix
                        },
                    type="ai"                 
                )
            ]
    }

def global_interpretation(state:AgentState):
    global count
    user_message = get_most_recent_human_message(state)
    # response = ablate_layers(USER_MODEL, "","","")
    from datasets import load_dataset

    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")

    return {
            "messages": [
                AIMessage(
                    id=count,
                    content=response,
                )
            ]
    }

# def global_interpretation(state:AgentState):
#     global count
#     user_message = get_most_recent_human_message(state)
#     response = ablate_layers(USER_MODEL, "","","")
#     return {
#             "messages": [
#                 AIMessage(
#                     id=count,
#                     content=response,
#                 )
#             ]
#     }

def help_desk(state:AgentState):
    global count
    user_message = get_most_recent_human_message(state)

    print("You are reaching out to help desk agent")
    # response = test_transformerlens()
    response = helo_desk_answerer(user_message)
    # response = "Hello"
    # print(response)

    return {
            "messages": [
                AIMessage(
                    id=count,
                    content=response,
                )
            ]
    }

def is_model_biased():
    pass

def local_interpretation(state:AgentState):
    global count
    user_message = get_most_recent_human_message(state)

    print("Tool Call Router")
    # response = test_transformerlens()
    response = "Please use the tool block on the left hand side to study attention patterns. :)  \n Feel free to explore other available tools as well!"
    # response = "Hello"
    # print(response)

    return {
            "messages": [
                AIMessage(
                    id=count,
                    content=response,        
                    additional_kwargs={
                            "type": "llm_visualization",
                            },
                    type="ai"        

                )
            ]
    }


#Here is a simple 3 steps graph that is going to be working in the bellow "decision" condition
def create_conversation_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_question)

    workflow.add_node("local_interpretation_agent", local_interpretation)
    workflow.add_node("global_interpretation_agent", global_interpretation)
    workflow.add_node("visualization_tool_router", tool_use_response)

    
    workflow.add_node("help_desk_agent", help_desk)

    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("evaluate_agent", evaluate_answer)

    # workflow.add_node("is_model_biased", is_model_biased)
    # workflow.add_node("is_model_robust_to_halucinnation", is_model_robust_to_halucinnation)


    workflow.add_node("generic_agent", answer_generic_question)
    workflow.add_node("next_word_probs_agent", next_word_prob)

    
    workflow.add_node("load_model", load_model)
    # workflow.add_node("get_head_importance", get_head_importance)
    workflow.add_node("get_layer_importance", get_layer_importance)

    workflow.add_node("validate_and_load_model", validate_and_load_model)
    workflow.add_node("get_user_input", get_user_input)

    workflow.add_edge("load_model", "get_user_input")
    workflow.add_edge("get_user_input", "validate_and_load_model")

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "help_desk": "help_desk_agent",
            "next_word_probs": "next_word_probs_agent",
            "attention_pattern": "local_interpretation_agent",
            "ablate_layers": "global_interpretation_agent",
            "visualization_tool_router" : "visualization_tool_router",

        }
    )

    workflow.set_entry_point("analyze")
    # workflow.add_edge("code_agent", END)
    # workflow.add_edge("evaluate_agent", END)
    workflow.add_edge("local_interpretation_agent", "get_layer_importance")

    workflow.add_edge("get_layer_importance", END)

    workflow.add_edge("visualization_tool_router", END)
    workflow.add_edge("global_interpretation_agent", END)
    # workflow.add_edge("is_model_biased", END)
    # workflow.add_edge("is_model_robust_to_halucinnation", END)

    workflow.add_edge("help_desk_agent", END)
    # workflow.add_edge("study_attention", END)
    # workflow.add_edge("model_info", END)
    return workflow.compile()

graph = create_conversation_graph()

# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)
# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"foo": ""}, config)
# # graph.invoke({"messages": "", "continue_conversation": True})