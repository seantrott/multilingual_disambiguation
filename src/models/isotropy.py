"""Calculate isotropy for each sentence/embedding/layer across models."""

import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


import utils



### Models to test
SPANISH_MODELS = ["dccuchile/bert-base-spanish-wwm-cased",
          "dccuchile/albert-tiny-spanish",
          "dccuchile/albert-base-spanish",
          "dccuchile/albert-large-spanish",
          "dccuchile/albert-xlarge-spanish",
          "dccuchile/albert-xxlarge-spanish",
          "PlanTL-GOB-ES/roberta-base-bne",
          "PlanTL-GOB-ES/roberta-large-bne",
          "dccuchile/bert-base-spanish-wwm-uncased", 
          "dccuchile/distilbert-base-spanish-uncased"]

ENGLISH_MODELS = ["bert-base-uncased",
          "bert-base-cased",
          "albert/albert-base-v1",
          "albert/albert-base-v2",
          "albert/albert-large-v2",
          "albert/albert-xlarge-v2",
          "albert/albert-xxlarge-v2",
          "FacebookAI/roberta-base",
          "FacebookAI/roberta-large",
          "distilbert/distilbert-base-uncased"]

MULTILINGUAL_MODELS = [# "FacebookAI/xlm-roberta-base",
          # "google-bert/bert-base-multilingual-cased",
          "FacebookAI/xlm-roberta-large",
          "distilbert/distilbert-base-multilingual-cased"
          ]


### Stim paths
ENGLISH_STIMULI = "data/raw/raw-c_sentences.csv" # "data/raw/raw-c_with_dominance.csv"
SPANISH_STIMULI = 'data/raw/saw-c_sentences.csv' ## get individual sentences ### "data/raw/sawc_avg_relatedness.csv"


### Mappings
lang_to_models_and_stimuli = {
    'english': ("raw-c", ENGLISH_STIMULI, ENGLISH_MODELS),
    'spanish': ("saw-c", SPANISH_STIMULI, SPANISH_MODELS)
}


def run_model(df, mpath, savepath, lang, multilingual, max_batch_size=10):
    """Run model on df using batching (safe for short sentences)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading model:", mpath)

    model = AutoModel.from_pretrained(mpath, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mpath)

    n_layers = model.config.num_hidden_layers
    print("number of layers:", n_layers)
    n_heads = model.config.num_attention_heads
    print("number of heads:", n_heads)

    n_params = utils.count_parameters(model)

    results = []

    # Split into batches
    for start in tqdm(range(0, len(df), max_batch_size)):
        batch_df = df.iloc[start:start + max_batch_size]
        sentences = batch_df["sentence"].tolist()

        # Tokenize all at once
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states_all = outputs.hidden_states  # list of [batch, seq, dim]

        for i_in_batch, row in enumerate(batch_df.itertuples()):
            attention_mask = inputs["attention_mask"][i_in_batch]
            valid_tokens = attention_mask.bool()

            for layer, h in enumerate(hidden_states_all):
                token_reps = h[i_in_batch][valid_tokens].detach().cpu().numpy()

                if token_reps.shape[0] < 3:
                    continue

                embs_unique = np.unique(token_reps, axis=0)
                if embs_unique.shape[0] < 3:
                    continue

                id_mle = utils.estimate_id(embs_unique, method="mle")
                id_twonn = utils.estimate_id(embs_unique, method="twonn")
                cosine_dists = pdist(embs_unique, metric="cosine")
                mean_cosine_dist = cosine_dists.mean()

                centered = embs_unique - embs_unique.mean(axis=0, keepdims=True)
                centered_normed = centered / np.linalg.norm(centered, axis=1, keepdims=True)
                cosine_sim_matrix = cosine_similarity(centered_normed)
                n = cosine_sim_matrix.shape[0]
                off_diag = cosine_sim_matrix[~np.eye(n, dtype=bool)]
                centered_isotropy = 1 - off_diag.mean()

                results.append({
                    'sentence': row.sentence,
                    'word': row.word,
                    'string': row.string,
                    'Layer': layer,
                    'id_mle': id_mle,
                    'id_twonn': id_twonn,
                    'num_tokens_original': token_reps.shape[0],
                    'num_tokens_unique': embs_unique.shape[0],
                    'prop_original_tokens': token_reps.shape[0] / embs_unique.shape[0],
                    'mean_dist': mean_cosine_dist,
                    'centered_isotropy': centered_isotropy
                })

    df_results = pd.DataFrame(results)
    df_results['n_params'] = n_params
    df_results['mpath'] = mpath
    df_results['language'] = lang
    df_results['multilingual'] = multilingual

    if savepath:
        df_results.to_csv(savepath, index=False)


### Handle logic for a dataset/model
def main(lang):

    ### Get right stimuli and models for language
    dataset, stim_path, models = lang_to_models_and_stimuli[lang]
    df = pd.read_csv(stim_path)

    ### For each language-specific model...
    
    for mpath in models:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/isotropy/{dataset}_{model}_isotropy.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="No")
    

    ### Also do for multilingual models
    
    for mpath in MULTILINGUAL_MODELS:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/isotropy/{dataset}_{model}_isotropy.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="Yes")
    



if __name__ == "__main__":

    ## Read stimuli
    main("spanish")
