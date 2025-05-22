"""Get cosine distances for each sentence pair."""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import gc

import utils
from tqdm import tqdm
from scipy.spatial.distance import cosine



MULTILINGUAL_MODELS = ["FacebookAI/xlm-roberta-base",
          # "google-bert/bert-base-multilingual-cased"
          # "FacebookAI/xlm-roberta-large",
          # "distilbert/distilbert-base-multilingual-cased"
          ]


### Stim paths
ENGLISH_STIMULI = "data/raw/raw-c_with_dominance.csv" # "data/raw/raw-c_with_dominance.csv"
SPANISH_STIMULI = 'data/raw/sawc_avg_relatedness.csv' ## get individual sentences ### "data/raw/sawc_avg_relatedness.csv"


### Mappings
lang_to_stimuli = {
    'english': ("raw-c", ENGLISH_STIMULI),
    'spanish': ("saw-c", SPANISH_STIMULI)
}



def run_model(df, mpath, dataset, lang, multilingual = "Yes"):

    ### Running on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    ### Do this for each layer/head index
    for layer_idx in range(1, n_layers):
        print("Layer: " + str(layer_idx))
        for head_idx in range(n_heads):

            ### Reload model in memory to deal with overwriting
            model = AutoModel.from_pretrained(
                mpath,
                output_hidden_states = True
            )
            model.to(device) # allocate model to desired device

            # Save pre-modification Q and K weights (BERT-style)
            q_weight_pre = model.encoder.layer[layer_idx].attention.self.query.weight.data.cpu().numpy()
            k_weight_pre = model.encoder.layer[layer_idx].attention.self.key.weight.data.cpu().numpy()

            # Apply modification
            modified_model, q_start_idx, k_start_idx, head_size, hidden_size = utils.mask_all_but_one_head_globally_with_v(
                model, layer_idx, head_idx, device
            )


            n_layers = modified_model.config.num_hidden_layers
            n_params = utils.count_parameters(model)

            results = []


            for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

                ### Get word
                target = " {w}".format(w = row['string']) if lang == "english" else " {w}".format(w = row["Word"])
                word = row["word"] if lang == "english" else row["Word"]

                ### Run model for each sentence
                s1_outputs = utils.run_model(modified_model, tokenizer, row['sentence1'], device)
                s2_outputs = utils.run_model(modified_model, tokenizer, row['sentence2'], device)

                ### Now, for each layer...
                for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval

                    ### Get embeddings for word
                    s1 = utils.get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, target, layer, device)
                    s2 = utils.get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, target, layer, device)

                    ### Now calculate cosine distance 
                    model_cosine = cosine(s1.detach().cpu(), s2.detach().cpu())

                    if row['same'] == True:
                        same_sense = "Same Sense"
                    else:
                        same_sense = "Different Sense"


                    ### Figure out how many tokens you're
                    ### comparing across sentences
                    n_tokens_s1 = len(tokenizer.encode(row['sentence1']))
                    n_tokens_s2 = len(tokenizer.encode(row['sentence2']))

                    ### Add to results dictionary
                    results.append({
                        'sentence1': row['sentence1'],
                        'sentence2': row['sentence2'],
                        'word': word,
                        'string': target,
                        'Same_sense': same_sense,
                        'Distance': model_cosine,
                        'Layer': layer,
                        'mean_relatedness': row['mean_relatedness'],
                        'S1_ntokens': n_tokens_s1,
                        'S2_ntokens': n_tokens_s2,
                        'Ablation': 'Zero-Ablation',
                        'Layer_ablated': layer_idx,
                        'Head_ablated': head_idx,
                        'Ablation_idx': (layer_idx, head_idx)
                    })

            df_results = pd.DataFrame(results)
            df_results['n_params'] = n_params
            df_results['mpath'] = mpath
            df_results['language'] = lang 
            df_results['multilingual'] = multilingual


            ### Save it
            just_model_name = mpath.split("/")[1] if "/" in mpath else mpath
            savepath = "data/processed/distances_ablated/{dataset}_{model}_distances_L{l}H{h}.csv".format(dataset = dataset, model = just_model_name,
                                                                                                          l = str(layer_idx), h = str(head_idx))

            df_results.to_csv(savepath, index=False)

            ### Delete to save memory
            del model
            del modified_model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()


### Handle logic for a dataset/model
def main(lang):


    ### Get right stimuli and models for language
    dataset, stim_path = lang_to_stimuli[lang]
    df = pd.read_csv(stim_path)

    for mpath in MULTILINGUAL_MODELS:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        ### TODO: Modify
        
        run_model(df, mpath, dataset, lang=lang, multilingual="Yes")



if __name__ == "__main__":

    ## Run main
    main("english")

