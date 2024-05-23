from .classes import WikidataPage, DisplaySynset2Wikidata
from typing import (
    List,
    Dict,
    Tuple
)
from collections import defaultdict
from tqdm import tqdm
from ..mapWikipedia import (
    get_lemma_by_title,
    write_pkl,
    read_pkl,
    create_info_about_sense,
    wn,
    remove_non_ascii_cyrillic,
    SentenceBertTransformer,
    cosine_similarity
)
from config.const import PATH_TO_TMP_FILE, PATH_TO_FASTTEXT
from .utils_local import (
    get_score,
    extract_ctx_wikidata
)
import fasttext
import numpy as np
import pandas as pd


def create_dict_candidates(articles:List[WikidataPage]=None,
                            mode:str='read') -> Dict[str, List[WikidataPage]]:
    """
        This function takes a list of WikidataPage objects and constructs a dictionary of candidates in the format '3131-N-313' -> list[WikidataPage].

        Parameters:
            articles (List[WikidataPage], optional): A list of WikidataPage objects. Defaults to None.
            mode (str, optional): The mode of operation (read or overwrite). Defaults to 'read'.
        Returns:
            Dict[str, List[WikidataPage]]: A dictionary where keys are synset IDs and values are lists of WikidataPage objects.
    """
    if mode!='read':
        dictWn = create_info_about_sense()
        canidates = defaultdict(list)
        for wikipage in tqdm(articles):
            article, lemma = wikipage.page, wikipage.lemma
            if 'ru' in article['label']:
                if lemma:
                    synset_base_on_lemma = wn.get_synsets(lemma)
                    if synset_base_on_lemma:
                        for synset in wn.get_synsets(lemma):
                            canidates[synset.id].append(WikidataPage(article, lemma))
                else:
                    lemma_new = get_lemma_by_title(article['label']['ru'], dictWn)
                    if lemma_new:
                        for synset in wn.get_synsets(lemma_new):
                            canidates[synset.id].append(WikidataPage(article, lemma_new))
        write_pkl(canidates, path=PATH_TO_TMP_FILE+'candidates_wikidata.pkl')
    else:
        canidates = read_pkl(path=PATH_TO_TMP_FILE+'candidates_wikidata.pkl')
    return canidates


def bindings(candidates:Dict[str, List[WikidataPage]]=None, type_bindings:str='base',
                           model_name:str='setu4993/LaBSE', log_len:int=1000, mode:str='read')-> Dict[str, List[float]]:
    '''
        This is the function of linking a wikidata page to a specific synset (similar to wikipedia, here we immediately assume that all cases are ambiguous)

        Parameters:
            candidates (Dict[str, List[Wikidata Page]], optional): A dictionary where keys are synset IDs and values are lists of WikidataPage objects. Defaults to None.
            mode (str, optional): The mode of operation. Defaults to 'read'.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are synset IDs and values are lists
    '''
    file_prefix = type_bindings+ (model_name.replace('/', '') if type_bindings=='labse' else '') + '_'
    if type_bindings == 'labse':
            labse = SentenceBertTransformer(model_name=model_name, device="cuda")
            labse.load_model()
    elif type_bindings == 'fasttext':
        ft = fasttext.load_model(PATH_TO_FASTTEXT)
    if mode != 'read':
        with open(PATH_TO_TMP_FILE+f'log_bindings_{log_len}{file_prefix}.txt', 'w') as log_file:
            dictWn = create_info_about_sense()
            score_dict = defaultdict(list)
            i = 0
            for _, (title_synset, candidatess) in tqdm(enumerate(candidates.items())):
                if i < log_len:
                    print(f'-----Synset: {title_synset}--------',
                        sep='\n', end='\n', file=log_file)
                for wikipage in candidatess:
                    candidate, lemma = wikipage.page, wikipage.lemma
                    title = remove_non_ascii_cyrillic(candidate['label']['ru'])
                    if type_bindings == 'base':
                        if 'N' in wn[lemma][0].id:
                            synset_ctx = dictWn[wn[lemma][0].id].ctx
                            article_ctx = extract_ctx_wikidata(candidate)
                            article_ctx.update([candidate['label']['ru'].lower(), lemma.lower()])
                            score = get_score(article_ctx, synset_ctx)
                            score_dict[title_synset].append(score)
                            if i < log_len:
                                print(f'{title}: {score}', 'ctx-article:', article_ctx, f'synset_ctx: {synset_ctx}',
                                    sep='\n', end='\n', file=log_file)
                        else:
                            print(f'{title}: {0.0}', 'lemma is not N: lemma',
                                    sep='\n', end='\n', file=log_file)
                            score_dict[title_synset].append(0.0)
                    elif type_bindings == 'fasttext':
                        article_ctx = extract_ctx_wikidata(candidate)
                        article_ctx.update([candidate['label']['ru'].lower(), lemma.lower()])
                        embed_wiki_page = np.zeros(ft.get_dimension())
                        ctxw = set(map(remove_non_ascii_cyrillic, article_ctx))
                        for word in ctxw:
                            embed_wiki_page += ft.get_word_vector(word)
                        average_embedding_wiki_page = embed_wiki_page / len(ctxw)

                        embed_title_synset = ft.get_word_vector(title_synset)
                        score = cosine_similarity(average_embedding_wiki_page, embed_title_synset)
                        score_dict[title_synset].append(score)
                        if i < log_len:
                            print(f'{title}: {score}', 'ctx-article:', article_ctx, f'synset title for embed: {title_synset}',
                                sep='\n', end='\n', file=log_file)
                    elif type_bindings == 'labse':
                        
                        if candidate['descriptions']:
                            sentence = candidate['descriptions']['ru'] if 'ru' in candidate['descriptions'] else candidate['descriptions']['en']
                        else:
                            sentence = 'описания нет'
                        first =  title + '[SEP]' + remove_non_ascii_cyrillic( sentence)

                        sentence_hyper = f'{title} - это тоже, что и {title_synset}'
                        cosine_score = labse.cosine_similarity(sentence_hyper, first)
                        score_dict[title_synset].append(cosine_score)
                        if i < log_len:
                            print(f'{title}: {cosine_score}', first, sentence_hyper,
                                sep='\n', end='\n', file=log_file)
                i += 1
        
        write_pkl(score_dict, path=PATH_TO_TMP_FILE + file_prefix + 'score_wikidata.pkl')
    else:
        score_dict = read_pkl(path=PATH_TO_TMP_FILE + file_prefix + 'score_wikidata.pkl')
    return score_dict


def take_mapping(score:Dict[str, List[float]]=None, candidates:Dict[str, List[WikidataPage]]=None,
                  type_bindings:str='base', model_name:str='setu4993/LaBSE',
                 mode:str='read') -> Dict[str, DisplaySynset2Wikidata]:
    '''
    Description:
        This function maps the highest-scoring candidates to their corresponding synsets.

    Parameters:
        score (Dict[str, List[float]]): A dictionary where keys are synset IDs and values are lists of similarity scores.
        candidates (Dict[str, List[WikidataPage]]): A dictionary where keys are synset IDs and values are lists of WikidataPage objects.
        mode (str, optional): The mode of operation. Defaults to 'read'.

    Returns:
        Dict[str, DisplaySynset2Wikidata]: A dictionary where keys are synset IDs and values are DisplaySynset2Wikidata objects representing the highest-scoring mappings.
    '''
    file_prefix = type_bindings+ (model_name.replace('/', '') if type_bindings=='labse' else '') + '_'
    if mode != 'read':
        res_score = {}
        for _, (title_synset, candidatess) in tqdm(enumerate(candidates.items())):
            sorted_lst = sorted(zip(candidatess, score[title_synset]), key=lambda x: x[1], reverse=True)
            if sorted_lst and 'N' in title_synset:
                best:Tuple[WikidataPage, float] = sorted_lst[0]
                res_score[title_synset] = DisplaySynset2Wikidata(best[0].page['id'], best[0].page['label'], best[0].lemma, wn[best[0].lemma][0].id, best[1], title_synset)
        write_pkl(res_score, path=PATH_TO_TMP_FILE + file_prefix + 'score_wikidata_final.pkl')
    else:
        res_score = read_pkl(path=PATH_TO_TMP_FILE + file_prefix + 'score_wikidata_final.pkl')
    return res_score



def find_hyponym(articles, mapping, id2idx, set_candidates_id):
    mapping2id = {value.id:key for key, value in mapping.items()}
    candidates_for_elem  = defaultdict(set)
    counter = 0
    for _, elem in tqdm(enumerate(articles)):
        for rels in elem.page['rels']:
            if rels['type'] == 'subclass_of' and rels['rel_id'] in mapping2id:
                candidates_for_elem[rels['rel_id']].add(elem.page['id'])
                counter += 1

    df_res= pd.DataFrame({'hyperonym_id':candidates_for_elem.keys(), 'candidates_id':candidates_for_elem.values()})
    df_res['hyperonym_title'] = df_res['hyperonym_id'].apply(lambda x: articles[id2idx[x]].page['label'])
    def convert(lst):
        l = []
        for i in lst:
            try:
                l.append(articles[id2idx[i]].page['label']['ru'] if 'ru' in articles[id2idx[i]].page['label'] else articles[id2idx[i]].page['label']['en'])
            except:
                l.append('bad_name')
        return '; '.join(l)
    dictWn = create_info_about_sense()
    def in_wn(titles, ids):
        res = []
        for title, id in zip(titles.split( '; '), ids):
            if title != 'bad_name':
                lemma = get_lemma_by_title(title, dictWn)
                if not lemma and id in set_candidates_id:
                    res.append((title, 2))
                elif not lemma:
                    res.append((title, 1))
                else:
                    res.append((title, 0))
            else:
                res.append((title, 0))
        res = sorted(res, key=lambda x: -x[1])
        return res

    df_res['candidates_title'] = df_res['candidates_id'].apply(lambda x: convert(x))
    df_res['sort_by_wn'] = df_res.apply(lambda row: in_wn(row['candidates_title'], row['candidates_id']), axis=1)
    df_res['candidates_id'] = df_res['candidates_id'].apply(lambda x: '; '.join(x))
    df_res['wordnet_lemma'] = df_res['hyperonym_id'].apply(lambda x: mapping[mapping2id[x]].lemma)
    df_res['id_for_sort'] = df_res['hyperonym_id'].apply(lambda x: int(x[1:]))
    df_res = df_res.sort_values(by='id_for_sort').reset_index(drop=True)
    df_res.to_excel(PATH_TO_TMP_FILE+'candidates1.xlsx', index=False)
    return df_res
