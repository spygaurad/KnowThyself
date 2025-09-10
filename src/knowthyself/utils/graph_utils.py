
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, BaseMessage
import os
from uuid import uuid4
from typing import Any, Optional, Sequence, TypedDict, Annotated
import logging
from langgraph.graph import add_messages
from .generate_embed import find_nearest_function, retrieve_document, load_tool_description


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



def analyze_question(state, model_manager, embeddings):
    user_message = get_most_recent_human_message(state)
    decision = find_nearest_function(user_message,model_manager, embeddings)
    categories = ["bias_detection", "transformerlens", "bertviz", "rag", "load_model"]
    decision = categories[decision]
    print("Analysing... Made Decision: ", decision)
    return {"decision": decision}


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

