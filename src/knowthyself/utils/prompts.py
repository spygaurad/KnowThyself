bias_detection_description = """
calculate bias, calculate toxicity
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
Transformerlens is a library for visualizing and interpreting transformer models, particularly focusing on attention mechanisms. It allows users to inspect how different attention heads in various layers of a transformer model attend to different parts of the input sequence.
Use TransformerLens to generate attention heatmaps for specific layers and heads.
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
bertviz visualization 
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
Do retrieval-augmented generation, retrieval-augmented generation (RAG)
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

load_model_description="""
{
"user_model":"gpt2",
"load_model",
"orchestrator_model":"gemma3"
}
"""



transformerlens_extraction_prompt = """
You are an information extractor.
From the user’s natural language request, extract and return the following in JSON format:

user_question: a valid example sentence (if the user provides only a token or vague context, generate a short realistic sentence containing it).

layer_number: the numerical layer index if mentioned, otherwise None.

head_number: the numerical head index if mentioned, otherwise None.

Always respond only in JSON.

Example Input:
"I want to study attention pattern in layer 7 for a sentence with token 'she'"

Example Output:

  "user_question": "Maria is on leave today, she is not feeling good.",
  "layer_number": "7",
  "head_number": null


Provided User Input:
{messages}

Output:
"user_question": ..., "layer_number": ..., "head_number": ...

Give final json only, no other text.
"""