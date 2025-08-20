from langchain_community.embeddings import OllamaEmbeddings
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from markitdown import MarkItDown
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)

def get_embedding(text):
    return ollama_emb.embed_query(text)
    # return ollama.embeddings(model='nomic-embed-text', prompt=text)

# def convert_files_to_markdown(file_path):
#     md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
#     result = md.convert(file_path)
#     return result.text_content

def save_index():
    loader = DirectoryLoader(
    "/home/prasais/projects/KnowYourLLM/src/demo/documents",
    # glob=".txt",
    loader_cls=TextLoader, # Assumes plain text files
    loader_kwargs={'encoding': 'utf-8'}, # Specify encoding if needed
    show_progress=True,
    use_multithreading=True # Can speed up loading for many files
    )

    documents = loader.load()
    # print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, ollama_emb)
    db.save_local("src/demo/faiss_index")
    # pass 

def load_index():
    new_db = FAISS.load_local("src/demo/faiss_index", ollama_emb, allow_dangerous_deserialization=True)
    return new_db

def retrieve_document(query):
    db = load_index()
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    content = ["".join(item.page_content) for item in docs][0]
    # print(content)
    return content

    
general_description = """next word probability calculate
1) induction heads?
"""


help_desk = """Question is about what this chat agent does.
Example Questions:
1) How can you help me?
2) Hi!
3) What is
4) What type
5) Can you explain
6) Can you explain about interpretable AI?
"""

local_interpretation = """view the attention pattern
Example Questions:
1) attention values in a model.
2) visualize GPT attention pattern?
"""

global_interpretation = """Toxicity score calculate.
Example Questions:
1) What is the toxicity of the model?
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
    categories = ["help_desk", "next_word_probs", "attention_pattern", "ablate_layers"]
    return categories[best_match_index]
# print(find_nearest_function("/evaluate_model Question:Hi Answer:Hello"))


# retrieve_document("What are the different categories of explainable AI?")