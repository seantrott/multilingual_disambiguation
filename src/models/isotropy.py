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


def run_model(df, mpath, savepath, lang, multilingual):
    """Run model on df."""
    ### Running on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    print(mpath)
 
    model = AutoModel.from_pretrained(
        mpath,
        output_hidden_states = True
    )
    model.to(device) # allocate model to desired device

    tokenizer = AutoTokenizer.from_pretrained(mpath)


    n_layers = model.config.num_hidden_layers
    print("number of layers:", n_layers)
    n_heads = model.config.num_attention_heads
    print("number of heads:", n_heads)

    n_params = utils.count_parameters(model)

    results = []

    for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

        ### Get word
        target = " {w}".format(w = row['string'])
        disambiguating_word = " {w}".format(w = row['disambiguating_word']) # row['string']
        sentence = row['sentence']

        ### Run model for each sentence
        model_outputs = utils.run_model(model, tokenizer, sentence, device)
        hidden_states = model_outputs['hidden_states']

        ### Now, for each layer...
        for layer in range(len(hidden_states)):

            ### Get intrinsic dimensionality
            e_matrix = hidden_states[layer][0]
            ### Need to get unique embeddings first
            embs_unique = np.unique(e_matrix, axis=0)

            ### Final #tokens should be at least 3 or more
            if embs_unique.shape[0] < 3:
                ### 
                continue
            ### MLE estimator
            ### MLE and TwoNN
            id_mle = utils.estimate_id(embs_unique, method="mle")
            id_twonn = utils.estimate_id(embs_unique, method="twonn")

            ### Mean pairwise cosine distance
            cosine_dists = pdist(embs_unique, metric="cosine")
            mean_cosine_dist = cosine_dists.mean()

            ### Centered Isotropy (Ethayarajh)
            centered = embs_unique - embs_unique.mean(axis=0, keepdims=True)
            centered_normed = centered / np.linalg.norm(centered, axis=1, keepdims=True)
            cosine_sim_matrix = cosine_similarity(centered_normed)
            n = cosine_sim_matrix.shape[0]
            off_diag = cosine_sim_matrix[~np.eye(n, dtype=bool)]
            centered_isotropy = 1 - off_diag.mean()

            ### Add to results dictionary
            results.append({
                'sentence': row['sentence'],
                'word': row['word'],
                'string': row['string'],
                'Layer': layer,
                'id_mle': id_mle,
                'id_twonn': id_twonn,
                'num_tokens_original': len(e_matrix),
                'num_tokens_unique': len(embs_unique),
                'prop_original_tokens': len(e_matrix) / len(embs_unique),
                'mean_dist': mean_cosine_dist,
                'centered_isotropy': centered_isotropy

            })

    df_results = pd.DataFrame(results)
    df_results['n_params'] = np.repeat(n_params,df_results.shape[0])
    df_results['mpath'] = mpath
    df_results['language'] = lang 
    df_results['multilingual'] = multilingual
    

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
