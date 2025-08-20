from functools import partial
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from circuitsvis.attention import attention_heads

from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
import circuitsvis as cv
from transformer_lens.hook_points import (
    HookPoint,
) 


# Implementation of https://transformerlensorg.github.io/TransformerLens/generated/demos/Exploratory_Analysis_Demo.html


def plot_attention_to_file(attention_matrix, tokens, filename="attention_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 10))

    n = len(tokens)
    # Mask upper triangle
    masked = np.tril(attention_matrix)

    # Plot lower triangle with purple color map
    cax = ax.imshow(masked, cmap='Purples', vmin=0, vmax=1)

    # Add gray patches to upper triangle (above diagonal)
    for i in range(n):
        for j in range(n):
            if j > i:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgray'))

    # Token labels with indices
    xtick_labels = [f"{t} ({i})" for i, t in enumerate(tokens)]
    ytick_labels = [f"{t} ({i})" for i, t in enumerate(tokens)]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_yticklabels(ytick_labels)

    # Grid styling
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_title("Token Attention Visualization", fontsize=14, weight='bold', pad=20)

    # Colorbar
    fig.colorbar(cax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()

def check_next_word_prob(model, example_prompt, example_answer):
    #Added return answer_ranks in line 846 of venv/lib/site.../trans_lens/utils.py

    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"

    sent = "I am testing with exmaple prompt: '"+example_prompt+"'\n and example answer: '"+example_answer+"'\n The observed probability is: \n"

    return sent + "\n".join(utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True))


def get_attention_data_for_visualizer(model, text) -> dict:
    """
    Generates the data structure required by BertHeadVisualizer's additionalKwargs.

    Args:
        model_name: The name of the TransformerLens model (e.g., "gpt2-small").
        text: The input text to analyze.

    Returns:
        A dictionary formatted to be used as additional_kwargs for the frontend.
    """
    # model = HookedTransformer.from_pretrained(model_name)


    tokens_tensor = model.to_tokens(text)
    str_tokens = model.to_str_tokens(text)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens_tensor)

    num_layers = model.cfg.n_layers
    
    # Initialize the attn structure: [layer][head][query_idx][key_idx]
    all_layers_attention = []

    for layer_idx in range(num_layers):
        # Get attention for the current layer: [batch, head, query_pos, key_pos]
        layer_attention_tensor = cache[f"blocks.{layer_idx}.attn.hook_pattern"]

        # Select the batch dimension (usually 0 for a single input)
        # Move to CPU and convert to a nested Python list (number[][][])
        layer_attention_list = layer_attention_tensor[0].cpu().tolist()
        all_layers_attention.append(layer_attention_list)

    # For self-attention, right_text is the same as left_text
    right_text = str_tokens

    # Construct the AttentionDataObject as a Python dictionary
    bert_attention_data_object = {
        "name": "GPT2",
        "attn": all_layers_attention,
        "left_text": str_tokens,
        "right_text": right_text,
    }

    # The 'token' key in your frontend component refers to the string tokens
    # (what you've called 'left_text' or 'str_tokens')
    return {
        "token": str_tokens,
        "bert_attention": bert_attention_data_object,
        # You might also want to add 'is_type_attention: True' if that flag is still used
        # "is_type_attention": True
    }


def attention_patterns(text,model,layer, head):
    # gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_text = text
    gpt2_tokens = model.to_tokens(gpt2_text)
    gpt2_str_tokens = model.to_str_tokens(gpt2_text)

    # gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
    # attention_pattern = gpt2_cache["pattern", 0, "attn"]
    # print(cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern))
    attn_layer = layer
    attn_hook_name = "blocks."+str(layer)+".attn.hook_pattern"
    _, gpt2_attn_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
    
    # get 0th layer, attention
    gpt2_attn = gpt2_attn_cache[attn_hook_name]

    # The 0 index below is for 0th head
    plot_attention_to_file(gpt2_attn.cpu()[head], gpt2_str_tokens, filename="attention_output.png")
    return gpt2_str_tokens, gpt2_attn
    # return cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=gpt2_attn)


def ablate_layers(model,text, layer, head):
    # Define the head ablation hook to deactivate head 10 in layer 5
    """
    If the ablated loss is higher than the original loss, it indicates that the deactivated head 
    played a significant role in the model's ability to make accurate predictions.
    """
    
    layer_to_ablate = 5
    head_index_to_ablate = 10

    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = model.to_tokens(gpt2_text)


    def head_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Print the shape of the tensor to check the activation size
        print(f"Shape of the value tensor: {value.shape}")
        # Set the values for head 10 to zero
        value[:, :, head_index_to_ablate, :] = 0.
        return value

    # Get the original loss
    original_loss = model(gpt2_tokens, return_type="loss")

    # Run the model with the ablation hook for head 10 in layer 5
    ablated_loss = model.run_with_hooks(
        gpt2_tokens, 
        return_type="loss", 
        fwd_hooks=[(
            utils.get_act_name("v", layer_to_ablate), 
            head_ablation_hook
        )]
    )
    output = "The original loss of the model on question: '" + gpt2_text + "' is :  "+ str(original_loss.item())
    out_2 = "\n After ablating layer: " + str(layer_to_ablate) + "head: " + str(head_index_to_ablate) +"values to 0, the loss is: " + str(ablated_loss.item())
    return output + out_2




model = HookedTransformer.from_pretrained(
                    "gpt2-small",
                    center_unembed=True,
                    center_writing_weights=True,
                    fold_ln=True,
                    refactor_factored_attn_matrices=True,
                )

# test_obs(model,"","")
# attention_patterns(model,"","")
# ablate_layers(model,"","","")