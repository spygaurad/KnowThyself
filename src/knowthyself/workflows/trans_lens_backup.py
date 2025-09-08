
from PIL import Image
from io import BytesIO
import json
from langchain.prompts import PromptTemplate

from src.knowthyself.utils.graph_utils import get_most_recent_human_message, AgentState, UserInput
from src.knowthyself.utils.model_manager import ModelManager
import os
from src.knowthyself.tools.transformerlens.transformerlens_utils import attention_patterns, get_attention_data_for_visualizer, convert_to_base64
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.knowthyself.utils.prompts import transformerlens_extraction_prompt


def extract_required_args(user_message: str, llm) -> dict:
    prompt = PromptTemplate.from_template(transformerlens_extraction_prompt)
    chain = prompt | llm
    response = chain.invoke({"messages": user_message})
    print("*"*100)
    print("Response from LLM for extraction: ", response)
    try:
        response_dict = json.loads(response)
        print("*"*100)
        print(response_dict)
    except json.JSONDecodeError:
        response_dict = {"user_question": None, "layer_number": None, "head_number": None}
    return response_dict


def transformerlens_agent(state:AgentState, model_manager: ModelManager) -> dict:
    MAIN_LLM = model_manager.get_orchestrator()


    # USER_MODEL = model_manager.get_user_model()

    #Change model manager to hooked transformer
    try:
        model_manager.set_user_model(backend="hooked", model_name="gpt2-small")
    except:
            return {
            "messages": [
                AIMessage(
                    # id=count,
                    # content="I am temporarily unable to serve Image based responses. \n However if you have access to logs, you can view the pattern there.",
                    content = model_manager._user_extra.get("error_message", "Failed to load the Hooked Transformer model. Please ensure that the model is supported by Hooked Transformer."),
                    # timestamp= "2025-08-19T17:25:00.000Z",
                    type="ai"                 
                )
            ]
    }

    USER_MODEL = model_manager.get_user_model()
        # embedding_model = model_manager._emb
    user_message = get_most_recent_human_message(state)
    user_message = extract_required_args(user_message, MAIN_LLM)
    print("*"*100)
    print("User message to transformer lens agent: ", user_message)
    tool_request = json.loads(user_message)

    model, layer, head, text = tool_request["model"], int(tool_request["layer"]), int(tool_request["head"]), tool_request["text"]
    layer -=1
    head -= 1

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..', "..", 'src', 'files', 'documents', "results", "attention_output.png")
    gpt2_str_tokens, gpt2_attn = attention_patterns(text, USER_MODEL, layer, head, filepath)
    # gpt_2_attn_all = get_attention_data_for_visualizer(text=text, model=USER_MODEL)

    # file_path = "/home/prasais/projects/KnowYourLLM/attention_output.png"
    pil_image = Image.open(filepath)
    image_b64 = convert_to_base64(pil_image)
    llm_with_image_context = MAIN_LLM.bind(images=[image_b64])
    response = llm_with_image_context.invoke("Explain the attention pattern in the image")
    # response = "Please play with the below visualization: \n"
    print("*"*200)
    # print(gpt2_str_tokens)
    # print(gpt2_attn.tolist()[0])
    print(response)

    return {
            "messages": [
                AIMessage(
                    # id=count,
                    # content="I am temporarily unable to serve Image based responses. \n However if you have access to logs, you can view the pattern there.",
                    content = response,
                    # timestamp= "2025-08-19T17:25:00.000Z",
                    additional_kwargs={
                            "token": gpt2_str_tokens,
                            "attention": gpt2_attn.tolist()[0],
                            # "bert_attention":gpt_2_attn_all,
                            # "is_type_attention": True  # Flag to indicate this message has an attention matrix
                        },
                    type="ai"                 
                )
            ]
    }