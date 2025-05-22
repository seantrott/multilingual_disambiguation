"""Get cosine distances for each sentence pair."""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

import utils
from tqdm import tqdm
from scipy.spatial.distance import cosine



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
          #"google-bert/bert-base-multilingual-cased"
          "FacebookAI/xlm-roberta-large",
          "distilbert/distilbert-base-multilingual-cased"
          ]


### Stim paths
ENGLISH_STIMULI = "data/raw/raw-c_with_dominance.csv" # "data/raw/raw-c_with_dominance.csv"
SPANISH_STIMULI = 'data/raw/sawc_avg_relatedness.csv' ## get individual sentences ### "data/raw/sawc_avg_relatedness.csv"


### Mappings
lang_to_models_and_stimuli = {
    'english': ("raw-c", ENGLISH_STIMULI, ENGLISH_MODELS),
    'spanish': ("saw-c", SPANISH_STIMULI, SPANISH_MODELS)
}



def run_model(df, mpath, savepath, lang, multilingual = "Yes"):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        mpath,
        output_hidden_states = True
    )
    model.to(device) # allocate model to desired device

    tokenizer = AutoTokenizer.from_pretrained(mpath)
    n_layers = model.config.num_hidden_layers

    n_params = utils.count_parameters(model)

    results = []

    ### TODO: Why tqdm not working here?
    for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

        ### Get word
        target = " {w}".format(w = row['string']) if lang == "english" else " {w}".format(w = row["Word"])
        word = row["word"] if lang == "english" else row["Word"]

        ### Run model for each sentence
        s1_outputs = utils.run_model(model, tokenizer, row['sentence1'], device)
        s2_outputs = utils.run_model(model, tokenizer, row['sentence2'], device)

        ### Now, for each layer...
        for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval

            ### Get embeddings for word
            s1 = utils.get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, target, layer, device)
            s2 = utils.get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, target, layer, device)

            ### Now calculate cosine distance 
            #.  note, tensors need to be copied to cpu to make this run;
            #.  still faster to do this copy than to just have everything
            #.  running on the cpu
            if device.type == "mps":  
                model_cosine = cosine(s1.cpu(), s2.cpu())

            else: 
                model_cosine = cosine(s1, s2)


            if row['same'] == True or row['same'] == "Same Sense":
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
                'Ablation': 'Original'
            })

    df_results = pd.DataFrame(results)
    df_results['n_params'] = n_params
    df_results['mpath'] = mpath
    df_results['language'] = lang 
    df_results['multilingual'] = multilingual

    df_results.to_csv(savepath, index=False)


### Handle logic for a dataset/model
def main(lang):


    ### Get right stimuli and models for language
    dataset, stim_path, models = lang_to_models_and_stimuli[lang]
    df = pd.read_csv(stim_path)

    """
    for mpath in models:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/distances/{dataset}_{model}.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="No")
    """

    for mpath in MULTILINGUAL_MODELS:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/distances/{dataset}_{model}_distances.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="Yes")



if __name__ == "__main__":

    ## Run main
    main("spanish")