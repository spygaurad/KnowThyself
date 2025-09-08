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

bias_detection_description = """
Sentences related to male vs female pronouns, names, or roles unequally. Helps quantify and visualize gender-related disparities in completions, associations, and predictions.
What is the toxicity of the model?
Does the model prefer male pronouns over female ones when predicting occupations?
If I ask the model about “doctor” vs “nurse”, does it associate one with a gender?
How can I measure if the model is biased toward male names in text completion?
Does the model autocomplete “She is a ___” differently than “He is a ___”?
Are there biases in how it generates job titles for men vs women?
How does the model rank “Mr.” vs “Ms.” in likelihood?
Does the model assume different personality traits when given male/female subjects?
Can I check if the model overuses male pronouns in summaries?
Is there gender bias in story character role assignments?
How can I quantify disparities between male/female predictions?
Does the model prefer safe/flattering responses over factual ones?
How do I measure if the model shows “sycophancy” bias?
Does the model agree with obviously wrong user statements just to be polite?
Can I quantify how often the model adds unnecessary compliments?
Does the model treat “please” vs no “please” differently?
Is there bias toward positive sentiment over honesty in answers?
"""

attention_heatmap_transformerlens = """
Visualizes how transformer heads distribute attention across tokens. Provides interpretable heatmaps of attention matrices for each layer and head.

How much attention does each head give to specific tokens?
Which attention heads are responsible for long-range dependencies?
Can I visualize how the model attends to punctuation vs words?
Do early layers attend differently than later layers?
How do attention weights shift when I change one word in the input?
Which heads specialize in copying vs syntax?
Can I see token-by-token heatmaps of attention strength?
How can I compare attention across layers visually?
Do certain heads always focus on the start/end tokens?
How can I diagnose spurious attention connections?
Can i see the attention for 4th head? and 12 layer?
I want to see the attention values for 12 head when i ask question 'What is the temperature of the surface of sun?'
"""

model_view_bertzview = """
Provides an interactive, graph-based visualization of token-to-token attention patterns. Lets users click and explore word relationships across layers.
How do specific tokens attend to each other in BERT?
Can I interactively explore which words are linked by attention?
How can I compare self-attention across multiple layers?
Which heads link semantically related words?
How do I visualize cross-attention patterns for a sentence?
Can I explore attention flow from subject to object in a sentence?
How do I view attention connections between question and answer tokens?
Is there a way to click on tokens and see attention arcs?
How do attention maps differ between fine-tuned and base BERT?
Can I explore hierarchical relationships between tokens interactively?
"""

general_rag_description="""
Retrieves relevant documents and answers knowledge-based queries by grounding LLM output in external sources. Useful for factual, up-to-date, or document-specific questions.
What is the capital of Nepal?
Can you summarize the latest paper on retrieval-augmented generation?
How does contrastive learning improve embeddings?
What are the applications of interpretable AI in healthcare?
Can you find me the differences between BERT and GPT?
How does RAG help in knowledge-intensive QA?
Summarize a document on fairness in AI models.
What are key challenges in explainability research?
Can you retrieve guidelines for AI transparency?
What are the limitations of interpretability methods in NLP?
"""



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

def load_tool_description():

    bias_detection_emb = get_embedding(bias_detection_description)
    transformer_lens_emb  = get_embedding(attention_heatmap_transformerlens)
    bertviz_emb  = get_embedding(model_view_bertzview)
    rag_emb  = get_embedding(general_rag_description)
    return bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb


def get_embedding(text):
    return ollama_emb.embed_query(text)
    # return ollama.embeddings(model='nomic-embed-text', prompt=text)

# write_all_options()
bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb = load_tool_description()

def find_nearest_function(question):
    question_embed = np.array(get_embedding(question))

    all_embeddings = [bias_detection_emb, transformer_lens_emb, bertviz_emb, rag_emb]
    all_embeddings = [np.array(embed) for embed in all_embeddings]
    similarities = [cosine_similarity(question_embed.reshape(1, -1), embed.reshape(1, -1))[0][0] for embed in all_embeddings]

    # Find the best match
    best_match_index = np.argmax(similarities)
    categories = ["bias_detection", "transformerlens", "bertviz", "rag"]
    return categories[best_match_index]
