"""Useful functions for getting distance information and embedding information."""

import functools
import torch
from torch.nn.functional import softmax
import numpy as np
from skdim.id import TwoNN, MLE, KNN




def estimate_id(embs, method="twonn"):
    if method == "twonn":
        return TwoNN().fit(embs).dimension_
    elif method == "mle":
        return MLE().fit(embs).dimension_
    elif method == "knn":
        return KNN().fit(embs).dimension_



def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None

@functools.lru_cache(maxsize=None)  # This will cache results, handy later...



def run_model(model, tokenizer, sentence, device):
    """Run model on a sentence and return hidden states, attentions, and tokens."""

    # Tokenize and move to the correct device
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)

    hidden_states = output.hidden_states
    attentions = output.attentions

    # Get token strings (move input_ids back to CPU first if needed)
    token_ids = inputs["input_ids"].detach().cpu()[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        'hidden_states': hidden_states,
        'attentions': attentions,
        'tokens': inputs
    }



### ... grab the embeddings for your target tokens
def get_embedding(hidden_states, inputs, tokenizer, target, layer, device):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding



def calculate_attention_entropy(attention_distribution):
    """
    Calculate entropy over an attention distribution.
    
    Args:
        attention_distribution (torch.Tensor): Attention weights for a single token 
                                                (1D tensor of size seq_len).
    
    Returns:
        float: Entropy value.
    """
    # Normalize the attention distribution using softmax
    attention_probs = softmax(attention_distribution, dim=-1)
    
    # Avoid log(0) by masking zeros
    attention_probs = attention_probs + 1e-12  # Small epsilon to prevent NaN
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attention_probs * torch.log(attention_probs)).item()
    
    return entropy

def get_attention_and_entropy_for_head(
    attentions, inputs, tokenizer, target, disambiguating, layer, head, device
):
    """
    Get entropy over attention from a target token to all tokens,
    and the attention from a target token to a specific disambiguating token
    for a specified head in a given layer.
    
    Args:
        model: Pretrained Transformer model.
        tokenizer: Corresponding tokenizer.
        sentence (str): Input sentence.
        target (str): Target word.
        disambiguating (str): Disambiguating word.
        layer (int): Layer index for attention extraction.
        head (int): Head index for attention extraction.
        device (str): Device to run computations on (e.g., 'cpu', 'cuda').
    
    Returns:
        dict: Contains entropy of attention distribution, attention to disambiguating token, 
              and attention distribution.
    """
    # Tokenize target and disambiguating words
    target_enc = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False).to(device)
    disambiguating_enc = tokenizer.encode(disambiguating, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Find indices of target and disambiguating words in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )
    disambiguating_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        disambiguating_enc[0].tolist()
    )
    
    if target_inds is None:
        raise ValueError(f"Target word '{target}' not found in the tokenized input.")
    if disambiguating_inds is None:
        raise ValueError(f"Disambiguating word '{disambiguating}' not found in the tokenized input.")
    
    # Extract attention from the specified layer
    attention_layer = attentions[layer][0]  # Shape: (num_heads, seq_len, seq_len)
    
    # Select the specified head
    attention_head = attention_layer[head]  # Shape: (seq_len, seq_len)
    
    # Get attention distribution for the target token(s)
    target_attention = attention_head[target_inds[0]:target_inds[1]]  # Shape: (target_len, seq_len)
    
    # Average over multiple tokens if target spans multiple subwords
    attention_distribution = torch.mean(target_attention, dim=0)  # Shape: (seq_len)
    
    # Calculate entropy over the attention distribution
    attention_probs = softmax(attention_distribution, dim=-1)
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-12)).item()
    
    # Calculate attention to the disambiguating token(s)
    disambiguating_attention = attention_distribution[
        disambiguating_inds[0]:disambiguating_inds[1]
    ]
    attention_to_disambiguating = torch.mean(disambiguating_attention).item()
    
    return {
        "entropy": entropy,
        "attention_to_disambiguating": attention_to_disambiguating,
        "attention_distribution": attention_distribution
    }


def get_attention_and_entropy_for_head_modified(
    attentions, inputs, tokenizer, target, disambiguating, layer, head, device
):
    """
    Get entropy over attention from a target token to all tokens,
    attention from a target token to a specific disambiguating token,
    number of tokens in the target and disambiguating words,
    and attention to the token just before the target.
    
    Args:
        attentions: Attention weights from the model.
        inputs: Tokenized input data.
        tokenizer: Corresponding tokenizer.
        target (str): Target word.
        disambiguating (str): Disambiguating word.
        layer (int): Layer index for attention extraction.
        head (int): Head index for attention extraction.
        device (str): Device to run computations on (e.g., 'cpu', 'cuda').
    
    Returns:
        dict: Contains entropy of attention distribution, attention to disambiguating token,
              attention to the token just before the target, number of tokens in target,
              and number of tokens in disambiguating word.
    """
    # Tokenize target and disambiguating words
    target_enc = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False).to(device)
    disambiguating_enc = tokenizer.encode(disambiguating, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Find indices of target and disambiguating words in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )
    disambiguating_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        disambiguating_enc[0].tolist()
    )
    
    if target_inds is None:
        raise ValueError(f"Target word '{target}' not found in the tokenized input.")
    if disambiguating_inds is None:
        raise ValueError(f"Disambiguating word '{disambiguating}' not found in the tokenized input.")
    
    # Get number of tokens in target and disambiguating words
    num_target_tokens = len(target_enc[0])
    num_disambiguating_tokens = len(disambiguating_enc[0])
    
    # Extract attention from the specified layer
    attention_layer = attentions[layer][0]  # Shape: (num_heads, seq_len, seq_len)
    
    # Select the specified head
    attention_head = attention_layer[head]  # Shape: (seq_len, seq_len)
    
    # Get attention distribution for the target token(s)
    target_attention = attention_head[target_inds[0]:target_inds[1]]  # Shape: (target_len, seq_len)
    
    # Average over multiple tokens if target spans multiple subwords
    attention_distribution = torch.mean(target_attention, dim=0)  # Shape: (seq_len)
    
    # Calculate entropy over the attention distribution
    attention_probs = softmax(attention_distribution, dim=-1)
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-12)).item()
    
    # Calculate attention to the disambiguating token(s)
    disambiguating_attention = attention_distribution[
        disambiguating_inds[0]:disambiguating_inds[1]
    ]
    attention_to_disambiguating = torch.mean(disambiguating_attention).item()
    
    # Get attention to the token just before the target
    target_prev_index = max(0, target_inds[0] - 1)  # Ensure index is not negative
    attention_to_previous = attention_distribution[target_prev_index].item()
    
    return {
        "entropy": entropy,
        "attention_to_disambiguating": attention_to_disambiguating,
        "attention_to_previous": attention_to_previous,
        "num_target_tokens": num_target_tokens,
        "num_disambiguating_tokens": num_disambiguating_tokens,
        "attention_distribution": attention_distribution
    }

### ... grab the number of trainable parameters in the model

def count_parameters(model):
    """credit: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    
    total_params = 0
    for name, parameter in model.named_parameters():
        
        # if the param is not trainable, skip it
        if not parameter.requires_grad:
            continue
        
        # otherwise, count it towards your number of params
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    
    return total_params




def modify_bert_attention_weights(model, layer_idx, head_idx, device):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_size = hidden_size // num_heads

    q_module = model.encoder.layer[layer_idx].attention.self.query
    k_module = model.encoder.layer[layer_idx].attention.self.key

    q_weight = q_module.weight.data
    k_weight = k_module.weight.data

    q_bias = q_module.bias.data
    k_bias = k_module.bias.data

    q_start_idx = head_idx * head_size
    k_start_idx = head_idx * head_size

    q_weight[q_start_idx:q_start_idx + head_size, :] = 0
    k_weight[k_start_idx:k_start_idx + head_size, :] = 0
    q_bias[q_start_idx:q_start_idx + head_size] = 0
    k_bias[k_start_idx:k_start_idx + head_size] = 0

    return model, q_start_idx, k_start_idx, head_size, hidden_size


def mask_all_but_one_head(model, layer_idx, head_idx, device):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_size = hidden_size // num_heads

    q_module = model.encoder.layer[layer_idx].attention.self.query
    k_module = model.encoder.layer[layer_idx].attention.self.key

    q_weight = q_module.weight.data
    k_weight = k_module.weight.data

    q_bias = q_module.bias.data
    k_bias = k_module.bias.data

    # Compute mask: indices to preserve
    qk_keep_idx = slice(head_idx * head_size, (head_idx + 1) * head_size)

    # Zero out all Q weights except for the selected head
    q_weight[:qk_keep_idx.start, :] = 0
    q_weight[qk_keep_idx.stop:, :] = 0

    # Zero out all K weights except for the selected head
    k_weight[:qk_keep_idx.start, :] = 0
    k_weight[qk_keep_idx.stop:, :] = 0

    # Zero out all Q bias except selected head
    q_bias[:qk_keep_idx.start] = 0
    q_bias[qk_keep_idx.stop:] = 0

    # Zero out all K bias except selected head
    k_bias[:qk_keep_idx.start] = 0
    k_bias[qk_keep_idx.stop:] = 0

    return model, qk_keep_idx.start, qk_keep_idx.start, head_size, hidden_size



def mask_all_but_one_head_globally_with_v(model, preserve_layer_idx, preserve_head_idx, device):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_size = hidden_size // num_heads

    for layer_idx in range(len(model.encoder.layer)):
        attn = model.encoder.layer[layer_idx].attention.self

        q_weight = attn.query.weight.data
        k_weight = attn.key.weight.data
        v_weight = attn.value.weight.data

        q_bias = attn.query.bias.data
        k_bias = attn.key.bias.data
        v_bias = attn.value.bias.data

        for head_idx in range(num_heads):
            if layer_idx == preserve_layer_idx and head_idx == preserve_head_idx:
                continue  # keep this head intact

            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size

            # Zero Q, K, V weights and biases for all other heads
            q_weight[start_idx:end_idx, :] = 0
            k_weight[start_idx:end_idx, :] = 0
            v_weight[start_idx:end_idx, :] = 0

            q_bias[start_idx:end_idx] = 0
            k_bias[start_idx:end_idx] = 0
            v_bias[start_idx:end_idx] = 0

    preserved_start_idx = preserve_head_idx * head_size
    return model, preserved_start_idx, preserved_start_idx, head_size, hidden_size