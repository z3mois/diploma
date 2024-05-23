
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
    wn,
    remove_non_ascii_cyrillic,
)
from .utils_local import (
    is_lat
)
from collections import defaultdict
from typing import Tuple, List
from .classes import WikidataPage

def extract_title_with_title_in_RuWordNet(path_to_data:str=DAMP_OF_WIKIDATA_PATH, mode:str='read') -> Tuple[set[str], List[WikidataPage]]:
    '''
        There is a path to the files with wikidata data. Each file contains one json line with information about the page.
          The files were obtained using https://github.com/SuLab/WikidataIntegrato.
    '''
    if mode != 'read':
        dictWn = create_info_about_sense()
        tweets:List[WikidataPage] = []
        to_add:set[str] = set()
        onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))]
        print('Start create data')
        for file in tqdm(onlyfiles):
            with open(f'{path_to_data}\\{file}', 'r', encoding='utf-8') as f:
                for line in f:
                    info = json.loads(line)
                    if info['id'] == 'Q190':
                            print(info)
                    if 'ru' in info['label'] and int(info['id'][1:]) <= 10000000 and 'фильм' not in info["label"]['ru'].lower():
                        label = remove_non_ascii_cyrillic(info["label"]['ru'])
                        sense = wn.get_senses(label)
                        if info['id'] == 'Q190':
                                print(sense)        
                        if sense:
                            tweets.append(WikidataPage(info, sense[0].lemma))
                            for elem in info['rels']:
                                to_add.add(elem['rel_id'])
                        else:
                            lemma = get_lemma_by_title(label, dictWn)
                            if info['id'] == 'Q190':
                                print(lemma, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                            if lemma:
                                tweets.append(WikidataPage(info, lemma))
                                for elem in info['rels']:
                                    to_add.add(elem['rel_id'])     
        print(f'Was found {len(tweets)} articles and need to add {len(to_add)}')
        write_pkl(to_add, path=PATH_TO_TMP_FILE+'titleWn_to_add.pkl')
        write_pkl(tweets, path=PATH_TO_TMP_FILE+'titleWn_tweets.pkl')
    else:
        to_add = read_pkl(path=PATH_TO_TMP_FILE+'titleWn_to_add.pkl')
        tweets = read_pkl(path=PATH_TO_TMP_FILE+'titleWn_tweets.pkl')
    return to_add, tweets



def extract_referring_pages(path_to_data:str=DAMP_OF_WIKIDATA_PATH, to_add:set[str]=None, articles:List[WikidataPage]=None,
                             depth:int=10, mode:str='read') -> List[WikidataPage]:
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
                to_add = to_add.union(new_to_add)
            print(len(to_add))
            new_to_add:set[str] = set()
            for file in onlyfiles:
                with open(f'{path_to_data}\\{file}', 'r', encoding='utf-8') as f:
                    for line in f:
                        info = json.loads(line)
                        if info['id'] in to_add and int(info['id'][1:]) <= 10000000 and 'ru' in info['label']  and 'фильм' not in info["label"]['ru'].lower():
                            try:
                                label = remove_non_ascii_cyrillic(info["label"]['ru'])
                                sense = wn.get_senses(label)
                                if sense:
                                    articles.append(WikidataPage(info, sense[0].lemma))
                                else:
                                    lemma = get_lemma_by_title(label, dictWn)
                                    if lemma is not None:
                                        articles.append(WikidataPage(info, lemma))
                                for elem in info['rels']:
                                    new_to_add.add(elem['rel_id'])
                            except:
                                pass
                        elif info['label'] and info['id'] in to_add and 'en' in info['label']:
                            for elem in info['rels']:
                                new_to_add.add(elem['rel_id'])
                            articles.append(WikidataPage(info, info['label']['en']))
        print(f'Was found {len(articles)} articles')
        write_pkl(articles, path=PATH_TO_TMP_FILE+'articles_all.pkl')
    else:
        articles = read_pkl(path=PATH_TO_TMP_FILE+'articles_all.pkl')
    return articles

def create_graph(articles, mode='read'):
    if mode != 'read':

        graph_path_straight = defaultdict(set)
        graph_path_inverse = defaultdict(set)
        id_artircle2idx_article = {}
        idx_article2id_artircle = {}
        for idx, wikidatapahe in tqdm(enumerate(articles)):
            article = wikidatapahe.page
            id_artircle2idx_article[article['id']] = idx
            idx_article2id_artircle[idx] = article['id']
            for link in article['rels']:
                graph_path_straight[link['rel_id']].add(article['id'])
                graph_path_inverse[article['id']].add(link['rel_id'])
        write_pkl(graph_path_straight, path=PATH_TO_TMP_FILE + 'graph_path_straight.pkl')
        write_pkl(graph_path_inverse, path=PATH_TO_TMP_FILE  + 'graph_path_inverse.pkl')
        write_pkl(id_artircle2idx_article, path=PATH_TO_TMP_FILE + 'id_artircle2idx_article.pkl')
        write_pkl(idx_article2id_artircle, path=PATH_TO_TMP_FILE  + 'idx_article2id_artircle.pkl')
    else:
        graph_path_straight = read_pkl(path=PATH_TO_TMP_FILE + 'graph_path_straight.pkl')
        graph_path_inverse = read_pkl(path=PATH_TO_TMP_FILE + 'graph_path_inverse.pkl')
        id_artircle2idx_article = read_pkl(path=PATH_TO_TMP_FILE + 'id_artircle2idx_article.pkl')
        idx_article2id_artircle = read_pkl(path=PATH_TO_TMP_FILE + 'idx_article2id_artircle.pkl')
    return graph_path_straight, graph_path_inverse, id_artircle2idx_article, idx_article2id_artircle