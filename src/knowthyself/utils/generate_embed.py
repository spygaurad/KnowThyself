from langchain_community.embeddings import OllamaEmbeddings
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from markitdown import MarkItDown
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
# from model_manager import get_embedding
from .prompts import bias_detection_description, attention_heatmap_transformerlens, model_view_bertzview, general_rag_description, load_model_description
from .model_manager import ModelManager

def save_index():
    document_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "..", 'src', 'files', 'documents')
    # index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "..", 'src', 'files', 'indexes',"")
    loader = DirectoryLoader(
            document_path,
            glob="**/*.txt",
            loader_cls=TextLoader, # Assumes plain text files
            loader_kwargs={'encoding': 'utf-8'}, # Specify encoding if needed
            show_progress=True,
            use_multithreading=True # Can speed up loading for many files
        )

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, ollama_emb)
    db.save_local("src/files/indexes/document_faiss_index")
    return True
    # pass 

def load_index():
    file_path = "src/files/indexes/document_faiss_index/index.faiss"
    if not os.path.exists(file_path):
        save_index()
    new_db = FAISS.load_local("src/files/indexes/document_faiss_index", ollama_emb, allow_dangerous_deserialization=True)
    return new_db

def retrieve_document(query):
    db = load_index()
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    content = ["".join(item.page_content) for item in docs][0]
    # print(content)
    return content


def load_tool_description(model_manager:ModelManager):

    bias_detection_emb = model_manager.embed(bias_detection_description)
    transformer_lens_emb  = model_manager.embed(attention_heatmap_transformerlens)
    bertviz_emb  = model_manager.embed(model_view_bertzview)
    rag_emb  = model_manager.embed(general_rag_description)
    load_model_description_emb = model_manager.embed(load_model_description)
    return bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb, load_model_description_emb



def find_nearest_function(question,model_manager, embeddings):
    question_embed = np.array(model_manager.embed(question))

    all_embeddings = embeddings
    all_embeddings = [np.array(embed) for embed in all_embeddings]
    similarities = [cosine_similarity(question_embed.reshape(1, -1), embed.reshape(1, -1))[0][0] for embed in all_embeddings]

    # Find the best match
    best_match_index = np.argmax(similarities)
    categories = ["bias_detection", "transformerlens", "bertviz", "rag", "load_model"]
    return categories[best_match_index]
