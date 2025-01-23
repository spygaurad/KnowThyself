from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from langchain_community.llms import ollama
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

from generate_embed import find_nearest_function
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
                            model = 'llama3.1:70b'
                            )
else:
    MAIN_LLM = ChatOpenAI()


USER_MODEL = None

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)

def get_llm():
    return MAIN_LLM


'''
def analyze_question(state):
    llm = get_llm()
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to classify a user question into different classes.
                                          
    Question : {input}

    The class name and the description of questions that falls under the class is defined below:

    1) code : If the question is related to programming language like java, C, python.
    2) general : Question falls under general open ended question.
    3) model : Question is about loading an AI model.
    4) chart : Question is about generating charts or visualing data.
    5) attention : Question is about attention weight of a model.
    6) info : Question is related to a specific layer or a head of a language model.

    Analyse the question. Only answer with the class names.
    
    Your answer (code/general/model/chart/attention/info) :
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})

    logging.info(f"Got response from LLM : {response}")
    # try:
    if model=="ollama":
        decision = response
    else:
        decision = response.content.strip().lower()
    # except:
    # decision = response
    return {"decision": decision, "input": state["input"]}

def generate_code(state):
    llm = get_llm()
    
    prompt = PromptTemplate.from_template(
        "Generate a questions whose answer will help to answer the given question. Donot give any explanation : {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    logging.info(f"Generating questions: \n {response}")
    logging.info("Asking the question to User Model")

    prompt = PromptTemplate.from_template(
        "Provide short answer to given question : {response}"
    )
    chain = prompt | USER_MODEL
    response = chain.invoke({"response": response})
    logging.info(f"User Model Answered: {response}")

    logging.info(f"Analysing Model Answer: \n")

    prompt = PromptTemplate.from_template(
        "Given question: {input} \n Answer: {response}. \n If the answer is correct, reply back positively else reply back with what is lacking."
    
    )
    # 
    chain = prompt | llm
    response = chain.invoke({"input": state["input"], "response":response})

    return {"output": response}

'''
def analyze_question(state):
    decision = find_nearest_function(state['input'])
    return {"decision": decision, "input": state["input"]}

# Creating the code agent that could be way more technical
def answer_code_question(state):

    prompt = PromptTemplate.from_template(
        "Provide short answer to given question : {input}"
    )
    chain = prompt | USER_MODEL
    response = chain.invoke({"input": state["input"]})
    # logging.info(f"User Model Answered: {response}")
    logging.info(f"User Model Answered: \n")

    return {"output": response}

def preprocess(state):
    user_input = state['input'].lower()
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
    response = chain.invoke({"question":question,"input": answer})
    # logging.info(f"Evaluation: \n {response}")
    logging.info(f"Evaluation: \n")

    return {"output": response}

# Creating the generic agent
def answer_generic_question(state):

    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain = prompt | llm

    logging.info("Generic Question Detected: Generating Answer ")
    response = chain.invoke({"input": state["input"]})

    logging.info(f"Generated Answer: {response}")

    return {"output": response}

def load_model(state):
    response = "I support following models: \n 1. LLAMA \n 2. Mistral \n 3. GPT \n\n Provide me with the huggingface repo and I will help you understand your model."
    print(response)
    return {"output": response}

def validate_and_load_model(state):
    global USER_MODEL  # Access the global variable
    model_name = state['input']

    USER_MODEL = ollama.Ollama(model=model_name) #'llama2:7b'
    try:
        answer_code_question(state)
        response = "Succesfully loaded "+ model_name +". You can ask questions to your model now."

    except:
        model_name = ""
        USER_MODEL = None
        response = "Invalid Model. Please provide me with Huggingface Model"
    # Validate model
    print(response)
    return {"model_name":state['input'], "output": response}

def chart_generator(state):
    response = "I help you understand your LLM visually."
    return {"output": response}

def understand_attention(state):
    response = "I see the relationship between words and help you understand it."
    return {"output": response}

def model_inquiry(state):
    response = "I can help you answer indepth model related questions like the value of attention in 16th head of layer 12"
    return {"output": response}

#You can precise the format here which could be helpfull for multimodal graphs
class AgentState(TypedDict):
    input: str
    output: str
    model_name : str
    decision: str

class UserInput(TypedDict):
    input: str
    continue_conversation: bool

def get_user_input(state: UserInput) -> UserInput:
    user_input = input("\nUser (ou 'q' to quit) : ")
    return {
        "input": user_input,
        "continue_conversation": user_input.lower() != 'q'
    }

def process_question(state: UserInput):
    graph = create_graph()
    result = graph.invoke({"input": state["input"]})
    # logging.info("\n--- Final answer ---")
    logging.info(f"{result['output']}")
    return state

#Here is a simple 3 steps graph that is going to be working in the bellow "decision" condition
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("evaluate_agent", evaluate_answer)

    workflow.add_node("generic_agent", answer_generic_question)

    workflow.add_node("load_model", load_model)
    workflow.add_node("chart_generator", chart_generator)
    workflow.add_node("study_attention", understand_attention)
    workflow.add_node("model_info", model_inquiry)


    workflow.add_node("validate_and_load_model", validate_and_load_model)
    workflow.add_node("get_user_input", get_user_input)
    workflow.add_edge("load_model", "get_user_input")
    workflow.add_edge("get_user_input", "validate_and_load_model")

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "code": "code_agent",
            "evaluate": "evaluate_agent",
            "general": "generic_agent",
            "model": "load_model",
            "chart": "chart_generator",
            "attention": "study_attention",
            "info": "model_info",

        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("code_agent", END)
    workflow.add_edge("evaluate_agent", END)
    workflow.add_edge("generic_agent", END)
    workflow.add_edge("validate_and_load_model", END)
    workflow.add_edge("chart_generator", END)
    workflow.add_edge("study_attention", END)
    workflow.add_edge("model_info", END)

    return workflow.compile()





def create_conversation_graph():
    workflow = StateGraph(UserInput)

    workflow.add_node("get_input", get_user_input)
    workflow.add_node("process_question", process_question)

    workflow.set_entry_point("get_input")

    workflow.add_conditional_edges(
        "get_input",
        lambda x: "continue" if x["continue_conversation"] else "end",
        {
            "continue": "process_question",
            "end": END
        }
    )

    workflow.add_edge("process_question", "get_input")

    return workflow.compile()

def main():
    conversation_graph = create_conversation_graph()
    conversation_graph.invoke({"input": "", "continue_conversation": True})

if __name__ == "__main__":
    main()


'''
The user wants to understand if their model has knowledge of C programming


Question -> MODEL -> Question related to C Programming
Question related to C Programming -> user_model -> answers ( option : Do RAG Here with precomputed tables )
answers -> MODEL -> Validate and Answer User Question

'''

'''
ollama 1 llm serving at a time

i need to switch to VLLM ()


'''

