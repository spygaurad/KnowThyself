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
