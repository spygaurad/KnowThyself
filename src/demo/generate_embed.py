from langchain_community.embeddings import OllamaEmbeddings
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)

def get_embedding(text):
    return ollama_emb.embed_query(text)
    # return ollama.embeddings(model='nomic-embed-text', prompt=text)

code_description = """Ask question to user model.
Example Questions:
/ask_model
"""

evaluate_description = """Evaluate user model response.
Example Questions:
/evaluate_model
"""

general_description = """Question falls under general open ended question.
Example Questions:
1) What is the height of Mt. everest?
2) Who is the first president of America?
"""

model_description = """Question is about loading an AI model.
Example Questions:
1) Load Model
2) I want to load my model.
"""

chart_description = """Question is about generating charts or visualing data.
Example Questions:
1) Can you draw chart of the attention?
2) Can you visualize this information?
"""

attention_description = """Question is about attention weight of a model.
1) Give me the attention map of below question.
2) Generate attention map.
"""

info_description = """Question is related to a specific layer or a head of a language model.
1) What is the 30th layer of the model learning?
2) What concepts are stored in 10th layer and 16th head of the model?
"""


def write_to_pickle(filename, list):
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(list,f)

def pickle_to_list(filename):
    with open(filename+'.pkl', 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist

def write_all_options():

    code_embed  = get_embedding(code_description)
    evaluate_embed = get_embedding(evaluate_description)
    general_embed  = get_embedding(general_description)
    model_embed  = get_embedding(model_description)
    chart_embed  = get_embedding(chart_description)
    attention_embed  = get_embedding(attention_description)
    info_embed  = get_embedding(info_description)

    write_to_pickle("code",code_embed)
    write_to_pickle("evaluate",evaluate_embed)
    write_to_pickle("general", general_embed)
    write_to_pickle("model", model_embed)
    write_to_pickle("chart", chart_embed)
    write_to_pickle("attention", attention_embed)
    write_to_pickle("info", info_embed)

def read_all_options():
    code_embed  = pickle_to_list("code")
    evaluate_embed = pickle_to_list("evaluate")
    general_embed  = pickle_to_list("general")
    model_embed  = pickle_to_list("model")
    chart_embed  = pickle_to_list("chart")
    attention_embed  = pickle_to_list("attention")
    info_embed  = pickle_to_list("info")
    return code_embed, evaluate_embed, general_embed, model_embed, chart_embed, attention_embed, info_embed

def get_embedding(text):
    return ollama_emb.embed_query(text)
    # return ollama.embeddings(model='nomic-embed-text', prompt=text)

# write_all_options()
code_embed, evaluate_embed, general_embed, model_embed, chart_embed, attention_embed, info_embed = read_all_options()

def find_nearest_function(question):
    question_embed = np.array(get_embedding(question))
    # print(len(question_embed), len(code_embed))
    all_embeddings = [code_embed, evaluate_embed, general_embed, model_embed, chart_embed, attention_embed, info_embed]
    all_embeddings = [np.array(embed) for embed in all_embeddings]
    similarities = [cosine_similarity(question_embed.reshape(1, -1), embed.reshape(1, -1))[0][0] for embed in all_embeddings]
    # Find the best match
    best_match_index = np.argmax(similarities)
    categories = ["code", "evaluate", "general", "model", "chart", "attention", "info"]
    return categories[best_match_index]

# print(find_nearest_function("/evaluate_model Question:Hi Answer:Hello"))