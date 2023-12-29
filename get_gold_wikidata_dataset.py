import pandas as pd
from config.const import DAMP_OF_WIKIDATA_PATH, GOLD_DATA, GOLD_WIKIDATA_DATASET
from os import listdir
from os.path import isfile, join
import json
import numpy as np


def main():
    data = pd.read_csv(GOLD_DATA)
    data_wiki_title = set([elem.lower() for elem in data['wiki_title_gold'].values])
    onlyfiles = [f for f in listdir(DAMP_OF_WIKIDATA_PATH) if isfile(join(DAMP_OF_WIKIDATA_PATH, f))]

    dict_wiki_to_wikidata = {}
    for file in onlyfiles:
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
    data.to_csv(GOLD_WIKIDATA_DATASET)

if __name__ == "__main__":
    main()