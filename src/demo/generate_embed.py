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



general_description = """Question falls under general open ended question.
Example Questions:
1) What is the height of Mt. everest?
2) Who is the first president of America?
"""


help_desk = """Question is about what this chat agent does.
Example Questions:
1) How can you help me?
2) What are features of this application?
"""

local_interpretation = """Question is about local interpretation techniques. This includes studying layers and heads info of a model.
1) Analyse the patterns present in the attention heads.
2) Which layer is responsible for pronouns resolution?
"""

global_interpretation = """Question is about global interpretation. Questions related to the models benaviour.
1) What is the behaviour of the model towards religious queries?
2) In which scenarios is the model hallucinating?
"""


def load_all_options():

    help_desk_embed = get_embedding(help_desk)
    general_embed  = get_embedding(general_description)
    local_interpretation_embed  = get_embedding(local_interpretation)
    global_interpretation_embed  = get_embedding(global_interpretation)
    return help_desk_embed, general_embed, local_interpretation_embed, global_interpretation_embed

def get_embedding(text):
    return ollama_emb.embed_query(text)
    # return ollama.embeddings(model='nomic-embed-text', prompt=text)

# write_all_options()
help_desk_embed, general_embed, local_interpretation_embed, global_interpretation_embed = load_all_options()

def find_nearest_function(question):
    question_embed = np.array(get_embedding(question))
    # print(len(question_embed), len(code_embed))
    all_embeddings = [help_desk_embed, general_embed, local_interpretation_embed, global_interpretation_embed]
    all_embeddings = [np.array(embed) for embed in all_embeddings]
    similarities = [cosine_similarity(question_embed.reshape(1, -1), embed.reshape(1, -1))[0][0] for embed in all_embeddings]
    # Find the best match
    best_match_index = np.argmax(similarities)
    categories = ["help_desk", "general", "local", "global"]
    return categories[best_match_index]

# print(find_nearest_function("/evaluate_model Question:Hi Answer:Hello"))