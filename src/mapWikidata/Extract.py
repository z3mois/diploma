
import json
from os import listdir
from os.path import isfile, join
from config.const import DAMP_OF_WIKIDATA_PATH, PATH_TO_TMP_FILE
from tqdm import tqdm
from ..mapWikipedia import (
    get_lemma_by_title,
    write_pkl,
    read_pkl,
    create_info_about_sense,
    wn
)
from .utils_local import (
    is_lat
)
from typing import Tuple, List


def extract_title_with_title_in_RuWordNet(path_to_data:str=DAMP_OF_WIKIDATA_PATH, mode:str='read') -> Tuple[set[str], List[dict]]:
    '''
        There is a path to the files with wikidata data. Each file contains one json line with information about the page.
          The files were obtained using https://github.com/SuLab/WikidataIntegrato.
    '''
    if mode != 'read':
        dictWn = create_info_about_sense()
        tweets = []
        to_add = set()
        onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))]
        print('Start create data')
        for file in tqdm(onlyfiles):
            with open(f'{path_to_data}\\{file}', 'r', encoding='utf-8') as f:
                for line in f:
                    info = json.loads(line)
                    if info['id'] == 'Q190':
                            print(info)
                    if 'ru' in info['label'] and int(info['id'][1:]) <= 10000000 and (is_lat(info["label"]['ru'])) and 'фильм' not in info["label"]['ru']:
                        try:
                            label = info["label"]['ru']
                            sense = wn.get_senses(label)
                            if info['id'] == 'Q190':
                                    print(sense)        
                            if sense:
                                tweets.append((info, sense[0].lemma))
                                for elem in info['rels']:
                                    to_add.add(elem['rel_id'])
                            else:
                                lemma = get_lemma_by_title(label, dictWn)
                                if info['id'] == 'Q190':
                                    print(lemma, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                                if lemma:
                                    tweets.append((info, lemma))
                                    for elem in info['rels']:
                                        to_add.add(elem['rel_id'])     
                        except:
                            pass
        print(f'Was found {len(tweets)} articles and need to add {len(to_add)}')
        write_pkl(to_add, PATH_TO_TMP_FILE+'titleWn_to_add.pkl')
        write_pkl(tweets, PATH_TO_TMP_FILE+'titleWn_tweets.pkl')
        print('First stage reading end')
    else:
        print("Start reading from file")
        to_add = read_pkl(PATH_TO_TMP_FILE+'titleWn_to_add.pkl')
        tweets = read_pkl(PATH_TO_TMP_FILE+'titleWn_tweets.pkl')
        print("Successful recording")
    return to_add,   tweets



def extract_referring_pages(path_to_data:str=DAMP_OF_WIKIDATA_PATH, to_add:set[str]=None, articles:List[dict]=None, depth:int=10, mode:str='read') -> List[dict]:
    '''
        We add all the pages leading to the data, and so recursively, due to problems related to limited memory.
        We will go through all the files several times and add these pages. We have depth of passages in total.
        param:
            path_to_data: path to dirictory with parse Wikidata
            to_add: set of the ID of the pages for which we will search for referring pages
            articles: List pages
            depth: The number of passes through all files
            mode: read or overwrite
        return:
            update list of Wikidata page
    '''
    if mode != 'read':
        print('Start adding referiing pages')
        onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))]
        dictWn = create_info_about_sense()
        for i in tqdm(range(depth)):
            if i != 0:
                to_add = new_to_add
            print(len(to_add))
            new_to_add = set()
            for file in onlyfiles:
                with open(f'{path_to_data}\\{file}', 'r', encoding='utf-8') as f:
                    for line in f:
                        info = json.loads(line)
                        if info['id'] in to_add and int(info['id'][1:]) <= 10000000 and 'ru' in info['label']  and 'фильм' not in info["label"]['ru']:
                            try:
                                label = info["label"]['ru']
                                sense = wn.get_senses(label)
                                if sense:
                                    articles.append((info, sense[0].lemma))
                                else:
                                    lemma = get_lemma_by_title(label, dictWn)
                                    if lemma is not None:
                                        articles.append((info, lemma))
                                for elem in info['rels']:
                                    new_to_add.add(elem['rel_id'])
                            except:
                                pass
                        elif info['label'] and info['id'] in to_add:
                            for elem in info['rels']:
                                new_to_add.add(elem['rel_id'])
                            articles.append((info, info['label']['en']))
        print(f'Was found {len(articles)} articles')
        write_pkl(articles, PATH_TO_TMP_FILE+'articles_all.pkl')
        print('Second stage reading end')
    else:
        print("Start reading from file")
        articles = read_pkl(PATH_TO_TMP_FILE+'articles_all.pkl')
        print("Successful recording")
    return articles
