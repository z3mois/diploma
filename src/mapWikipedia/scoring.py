from sklearn.metrics import accuracy_score
import pandas as pd
from config.const import GOLD_DATA
from src.mapWikipedia import Mapping
from typing import Dict

def scoring_wikipedia_bindings(second_dict:Dict[str, Mapping], final_dict:Dict[str, Mapping], path_to_dataset:str = GOLD_DATA) -> Dict[str, float]:
    """
        Calculate binding metrics based on Wikipedia data.

        Parameters:
        - second_dict (Dict[str, Mapping]): A dictionary obtained from earlier stages of binding (unambiguous concepts).
        - final_dict (Dict[str, Mapping]): The final dictionary.
        - path_to_dataset (str, optional): Path to the dataset. Defaults to GOLD_DATA.

        Returns:
        Dict[str, float]: A dictionary containing calculated binding metrics across the dataset and separately for multi and single bindings.

    """
    set_single = []
    for _, value in second_dict.items():
        set_single.append(value.wordId)
    set_single = set(set_single)

    df = pd.read_csv(path_to_dataset)
    synset_id = set(df["synset_id"].values)

    dict_for_check = {}
    for _, value in final_dict.items():
        if value.wordId in synset_id:
            dict_for_check[value.wordId] = value.title
    dataset = df[["synset_id", "wiki_title_gold"]]

    dataset["wiki_title"] = df["synset_id"].apply(lambda x: dict_for_check[x] if x in dict_for_check else "не связан")

    y_pred = dataset["wiki_title"].values
    y_true = dataset["wiki_title_gold"].values
    all_dataset_score = accuracy_score(y_true, y_pred)

    dataset["multi"] = df["synset_id"].apply(lambda x: 0 if x in set_single else 1)


    y_true = dataset.loc[dataset['multi'] == 0, 'wiki_title_gold'].values
    y_pred = dataset.loc[dataset['multi'] == 0, 'wiki_title'].values
    single_score = accuracy_score(y_true, y_pred)

    y_true = dataset.loc[dataset['multi'] == 1, 'wiki_title_gold'].values
    y_pred = dataset.loc[dataset['multi'] == 1, 'wiki_title'].values
    multi_score = accuracy_score(y_true, y_pred)
    without_bindings = dataset[dataset["wiki_title"]=="не связан"].shape[0]
    return {'all_dataset_score':all_dataset_score, 'single_dataset_score':single_score,'multi_dataset_score':multi_score, 'no_bindings': without_bindings}
