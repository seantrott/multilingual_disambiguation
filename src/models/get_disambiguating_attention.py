"""Get attention for a given model from target ambiguous word to disambiguating word"""


import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import statsmodels.formula.api as smf


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
          "distilbert/distilbert-base-multilingual-cased"]


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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

        ### clue
        if lang == 'spanish':
            cue = row['context_cue']
            pos = 'N'
        elif lang == 'english':
            pos = row['Class']
            cue = 'adjective' if pos == 'N' else 'noun'


        ### Now, for each layer...
        for layer in range(n_layers): 

            for head in range(n_heads): 

                ### Get heads
                ### TODO: Store attention to each token index maybe, and track which is disambiguating word, which is target, etc.?

                attention_info = utils.get_attention_and_entropy_for_head(model_outputs['attentions'], model_outputs['tokens'], tokenizer, 
                                                                         target, disambiguating_word, layer, head, device)
                

                ### Get attention weights for the given head
                attn_weights = model_outputs['attentions'][layer][0, head]  # Shape: (seq_len, seq_len)

                ### Extract attention to previous token (diagonal just below main diagonal)
                seq_len = attn_weights.shape[0]
                if seq_len > 1:  # Ensure sequence is long enough
                    prev_token_attention = torch.diagonal(attn_weights, offset=-1).mean().item()
                    next_token_attention = torch.diagonal(attn_weights, offset=1).mean().item()
                else:
                    prev_token_attention = None  # Skip if not applicable
                    next_token_attention = None

                ### Add to results dictionary
                results.append({
                    'sentence': row['sentence'],
                    'word': row['word'],
                    'string': row['string'],
                    'disambiguating_word': disambiguating_word,
                    'Attention': attention_info['attention_to_disambiguating'],
                    'Entropy': attention_info['entropy'],
                    'Head': head,
                    'Layer': layer,
                    'POS': pos,
                    'cue_pos': cue,
                    'prev_token_attention': prev_token_attention,
                    'next_token_attention': next_token_attention

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
    """
    for mpath in models:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/{dataset}_{model}.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="No")
    """

    ### Also do for multilingual models
    for mpath in MULTILINGUAL_MODELS:
        just_model_name = mpath.split("/")[1] if "/" in mpath else mpath

        savepath = "data/processed/{dataset}_{model}.csv".format(dataset = dataset, model = just_model_name)
        run_model(df, mpath, savepath, lang=lang, multilingual="Yes")



if __name__ == "__main__":

    ## Read stimuli
    main("english")
