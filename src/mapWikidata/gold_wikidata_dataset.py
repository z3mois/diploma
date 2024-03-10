import pandas as pd
from config.const import (
    DAMP_OF_WIKIDATA_PATH,
    GOLD_DATA,
    GOLD_WIKIDATA_DATASET,
    PATH_TO_TMP_FILE
)
from os import listdir
from os.path import isfile, join
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from .classes import DisplaySynset2Wikidata
from sklearn.metrics import accuracy_score


def create_dataset_for_wikidata(path_to_wikipedia_dataset:str=GOLD_DATA, path_to_wikidata_dataset:str=GOLD_WIKIDATA_DATASET) -> None:
    '''
        Creating a dataset to evaluate linking Wikidata to RuWrodNet based on the dataset for Wikipedia
    '''
    data = pd.read_csv(path_to_wikipedia_dataset)
    data_wiki_title = set([elem.lower() for elem in data['wiki_title_gold'].values])
    onlyfiles = [f for f in listdir(DAMP_OF_WIKIDATA_PATH) if isfile(join(DAMP_OF_WIKIDATA_PATH, f))]

    dict_wiki_to_wikidata = {}
    for file in tqdm(onlyfiles):
        with open(f'{DAMP_OF_WIKIDATA_PATH}\\{file}', 'r', encoding='utf-8') as f:
            for line in f:
                info = json.loads(line)
                if info['label'] and info['sitelinks'] and  'ruwiki' in info['sitelinks']:
                    if isinstance(info['sitelinks']['ruwiki'], list):
                        for elem in info['sitelinks']['ruwiki']:
                            if elem.lower() in data_wiki_title:
                                dict_wiki_to_wikidata[elem] = (info['label'], info['id'])
                    else:
                        if info['sitelinks']['ruwiki'].lower() in data_wiki_title:
                            dict_wiki_to_wikidata[info['sitelinks']['ruwiki']] = (info['label'], info['id'])
    data['WikiDataGoldTitle'] = data['wiki_title_gold'].apply(lambda x: dict_wiki_to_wikidata[x][0]['ru'] if x in dict_wiki_to_wikidata else np.nan)
    data['WikiDataGoldId'] = data['wiki_title_gold'].apply(lambda x: dict_wiki_to_wikidata[x][1] if x in dict_wiki_to_wikidata else np.nan)
    data.to_csv(path_to_wikidata_dataset, index=False)


def scoring(mapping: Dict[str, DisplaySynset2Wikidata]=None, path_to_gold:str=GOLD_WIKIDATA_DATASET) -> Tuple[float, float, int, int]:
    '''
         This function calculates accuracy scores for predicted mappings compared to gold standard data.
        Parameters:
            mapping (Dict[str, DisplaySynset2Wikidata], optional): A dictionary where keys are synset IDs and values are DisplaySynset2Wikidata objects representing predicted mappings. Defaults to None.
            path_to_gold (str, optional): The path to the gold standard dataset. Defaults to GOLD_WIKIDATA_DATASET.

        Returns:
            Tuple[float, float, int, int]: A tuple containing the accuracy score for predicted mappings against non-'не связан' entries,
            the accuracy score for all entries, the count of non-'не связан' entries, and the total count of entries.
    '''
    data = pd.read_csv(path_to_gold)
    data['predict_id'] = data['synset_id'].apply(lambda x: mapping[x].id if x in mapping else 'не связан')
    data_part = data[data['predict_id'] !='не связан']
    score = accuracy_score(data_part.WikiDataGoldId.astype(str), data_part.predict_id.astype(str))
    score_all = accuracy_score(data.WikiDataGoldId.astype(str), data.predict_id.astype(str))
    return score, score_all, len(data_part), len(data)
