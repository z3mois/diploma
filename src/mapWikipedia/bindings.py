from tqdm import tqdm
from .classes import Mapping, WnCtx, WikiSynset
from  .utils_local import (
    clear_title,
    read_pkl,
    write_pkl,
    extractCtxW,
    get_sense_id_by_title,
    get_lemma_by_title,
    count1,
    sort_dict_by_key,
    score,
    remove_non_ascii_cyrillic,
    cosine_similarity
)
from .model import SentenceBertTransformer
from collections import defaultdict
from ruwordnet import RuWordNet
from config.const import PATH_TO_TMP_FILE, PATH_TO_FASTTEXT
from typing import List, Tuple, Dict
import fasttext
import numpy as np

def unambiguous_bindings(wn:RuWordNet=None, dictWn:Dict[str, WnCtx]=None, wiki:List[WikiSynset]=None, mode:str='read') -> Tuple[Dict[str, Mapping], List[WikiSynset]]:
    '''
        The first stage is where we link to multi-valued Wikipedia pages 
        with unambiguous concepts in RuWordNet
        param:
            wn: RuWordNet 
            dictWn: dict from id ('101-N-100') to info about sense
            wiki: WikiSynset list
            mode: read or overwrite
        return:
            A linking dictionary and a new list without related concepts
    '''
    dictDisplay = {}
    new_wiki = []
    if mode!='read':
        print('Unambiguous bindings starts')
        for wikisyn in tqdm(wiki):
            if len(wikisyn.synset) != 1:
                new_wiki.append(wikisyn)
                continue
            title_clear = clear_title(wikisyn.page.title)
            id = get_sense_id_by_title(title_clear)
            if id and (id in dictWn):
                lemma = dictWn[id].lemmaInWn
                synset = wn.get_synsets(lemma)
                # print(lemma, synset)
                if  (not wikisyn.page.meaningPage) and (not wikisyn.page.multiPage) and (len(synset) == 1):
                    dictDisplay[wikisyn.page.title] = Mapping(wikisyn.page.id, wikisyn.page.revid, wikisyn.page.title, lemma, synset[0].id,
                                                            extractCtxW(wikisyn.page.links, wikisyn.page.categories), wikisyn.page.first_sentence)
                else:
                    new_wiki.append(wikisyn)
            else:
                new_wiki.append(wikisyn)
        write_pkl(dictDisplay, path=PATH_TO_TMP_FILE+'fst_dict.pkl')
        write_pkl(new_wiki, path=PATH_TO_TMP_FILE+'wiki_after_fst_stage.pkl')
    else:
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+'fst_dict.pkl')
        new_wiki = read_pkl(path=PATH_TO_TMP_FILE+'wiki_after_fst_stage.pkl')
    return dictDisplay, new_wiki


def create_candidates_index_dict(wiki:List[WikiSynset]=None, dictWn:Dict[str, WnCtx]=None, 
                                 name:str='lst_candidates_after_fst_stage.pkl', mode:str='read') -> Dict[str, List[int]]:
    '''
        Create dict_candidates in format Lemma -> idx in List[WikiSynset]
        param:
            wiki: List candidates
            dictWn: dictWn: castom dict from id to info about sense
            name: the name of the file to save
            mode: read or overwrite
    '''
    if mode != 'read':
        print('Start create dict candidates')
        dictLemmaInIndex = {}
        for index in tqdm(range(len(wiki))):
            lemma = get_lemma_by_title(wiki[index].page.title, dictWn)
            if lemma:
                if not lemma in dictLemmaInIndex:
                    dictLemmaInIndex[lemma] = []
                dictLemmaInIndex[lemma].append(index)
            else:
                for elem in wiki[index].synset:
                    lemma_redirect = get_lemma_by_title(elem.title, dictWn)
                    if lemma_redirect:
                        if not lemma_redirect in dictLemmaInIndex:
                            dictLemmaInIndex[lemma_redirect] = []
                        dictLemmaInIndex[lemma_redirect].append(index)
        print('Dict candidates created')
        write_pkl(dictLemmaInIndex, path=PATH_TO_TMP_FILE+name)
    else:
        dictLemmaInIndex = read_pkl(path=PATH_TO_TMP_FILE+name)
    return dictLemmaInIndex


def create_candidates_for_multi_stage(wiki:List[WikiSynset]=None, wn:RuWordNet=None, dictWn:Dict[str, WnCtx]=None,
                                      dictLemmaInIndex:Dict[str, int]=None, mode:str='read') -> Dict[str, List[WikiSynset]]:

    '''
        Creating a candidate list for a specific synset to solve the ambiguity problem
        param:
            wiki: List candidates
            dictWn: dictWn: castom dict from id to info about sense
            wn: RuWordNet 
            dictLemmaInIndex: dict_candidates in format Lemma -> idx in List[WikiSynset]
            mode: read or overwrite
        return:
            dict_candidates in format synset.id(121-N) -> List[WikiSynset]
    '''
    dictSynsetId = defaultdict(list)
    if mode != 'read':
        countLinksAdd = 0
        print('Start create candidates')
        for index in tqdm(range(len(wiki))):
            w = wiki[index]
            if not w.page.meaningPage:
                lemma = get_lemma_by_title(w.page.title, dictWn)
                if lemma:
                    for synset in wn.get_synsets(lemma):
                        dictSynsetId[synset.id].append(w)
            else:
                lemmaTitle = get_lemma_by_title(w.page.title, dictWn)
                for link in w.page.links:
                    lemma = get_lemma_by_title(link, dictWn)
                    if lemma and lemma in dictLemmaInIndex:
                        if lemmaTitle and get_sense_id_by_title(lemmaTitle) in dictWn:
                            for synset in wn.get_synsets(lemmaTitle):
                                countLinksAdd += 1
                                for indexElem in dictLemmaInIndex[lemma]:
                                    dictSynsetId[synset.id].append(wiki[indexElem]) 
        print(f'Was added from links {countLinksAdd}')
        write_pkl(dictSynsetId, path=PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl')
    else:
        dictSynsetId = read_pkl(path=PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl')
    return dictSynsetId

def add_multi_flag(wiki_pages:List[WikiSynset], dictLemmaInIndex:Dict[str, int]):
    '''
        Inplace is a method that sets additional ambiguity flags based on the list of candidates
        param:
            wiki_pages: The list of candidates to whom the flag will be added
            dictLemmaInIndex: dict_candidates in format Lemma -> idx in List[WikiSynset]
    '''
    print('Starts add flag')
    dictTitleInIndex = {}
    for index in tqdm(range(len(wiki_pages))):
        dictTitleInIndex[wiki_pages[index].page.title] = index
    add_counter = 0
    for _, lst_idx in dictLemmaInIndex.items():
        for idx in lst_idx:
            if not wiki_pages[idx].page.multiPage:
                if len(lst_idx)>1:
                    wiki_pages[dictTitleInIndex[wiki_pages[idx].page.title]].page.multiPage = True
                    add_counter +=1
    print(f'was added {add_counter} flag')


def second_stage_bindings(wn:RuWordNet=None, dictWn:Dict[str, WnCtx]=None, wiki:List[WikiSynset]=None,  
                          dictDisplay:Dict[str, Mapping]=None, mode:str='read') -> Tuple[Dict[str, Mapping], List[WikiSynset]]:
    '''
        The second stage of linking, where we link if the redirect is
        unambiguous and the concept corresponding to it in the RuWordNet is unambiguous
        param:
            wn: RuWordNet 
            dictWn: dict from id ('101-N-100') to info about sense
            wiki: WikiSynset list
            dictDisplay: answer dict after firse stage
            mode: read or overwrite
        return:
            Updated Dictdisplay and a new wiki list without pages that were linked
    '''
    wiki_for_multi = []
    if mode != 'read':
        countN = 0
        print("Start second stage bindings")
        for w in tqdm(wiki):
            flag = False
            if not w.page.meaningPage and not w.page.multiPage:
                for d in w.synset:
                    lemma = get_lemma_by_title(d.title, dictWn)
                    if lemma:
                        synset = wn.get_synsets(lemma)
                        if len(synset) == 1 and "N" in synset[0].id:
                            if w.page.title == "Abbath":
                                print(d.title)
                            flag = True
                            countN += 1
                            idd = synset[0].id
                            p = Mapping(w.page.id,w.page.revid,w.page.title, lemma, idd,
                                        extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
                            dictDisplay[w.page.title]=p
                            break
            if not flag:
                wiki_for_multi.append(w)
        print(f"There were {countN} entities connected")
        write_pkl(dictDisplay, path=PATH_TO_TMP_FILE+'snd_dict.pkl')
        write_pkl(wiki_for_multi, path=PATH_TO_TMP_FILE+'wiki_after_snd_stage.pkl')
    else:
        print("Start reading from file")
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+'snd_dict.pkl')
        wiki_for_multi = read_pkl(path=PATH_TO_TMP_FILE+'wiki_after_snd_stage.pkl')
    return dictDisplay, wiki_for_multi


def delete_double_in_candidates(dictSynsetId: Dict[str, List[WikiSynset]]) ->  Dict[str, List[WikiSynset]]:
    '''
        delete doubles in candidtes (by id page)
        param:
            dictSynsetId in format synset.id(121-N) -> List[WikiSynset]
        return:
            update dictSynsetId, without doubles
    '''
    doubles:int = 0
    print('Strt deleted doubles')
    for key in tqdm(dictSynsetId):
        tempList = []
        for elem in dictSynsetId[key]:
            if not elem.page.meaningPage:
                if count1(elem, tempList) < 1:
                    tempList.append(elem)
                else:
                    doubles += 1
        dictSynsetId[key] = tempList
    print(f'Was deleted {doubles} doubles')
    return dictSynsetId


def multi_bindings_stage(dictDisplay:Dict[str, Mapping]=None, dictSynsetId: Dict[str, List[WikiSynset]]=None, 
                         wn:RuWordNet=None, dictWn:Dict[str, WnCtx]=None, type_bindings:str='base',
                           model_name:str='setu4993/LaBSE', log_len:int=1000, mode:str='read') -> Tuple[Dict[str, Mapping], Dict[str, List[Tuple[WikiSynset, float]]]]:
    '''
        The stage of linking to a specific synset of a certain Wikipedia article
        param:
            dictDisplay:    answer dict after second stage
            dictSynsetId: dictSynsetId in format synset.id(121-N) -> List[WikiSynset]
            wn: RuWordNet 
            dictWn: dict from id ('101-N-100') to info about sense  
            model_name: name model from HF to type_bindings = labse
            type_bindings: base(method from Archiv), labse(check only cosine similarity), fasttext (check similarity fasttext embeddings)
            model_name: model_name for method labse(for exmaple: setu4993/LaBSE, intfloat/multilingual-e5-large)
            mode: read or overwrite
        return:
           Answer dict after multi stage
    '''
    file_prefix = '_'+type_bindings+ (model_name.replace('/', '') if type_bindings=='labse' else '')
    print(file_prefix)
    if mode != 'read':
        dictSortCandidates = {}
        badlemma, baddenominator, badmaxP, badsynsetlemma, badidWn = [], [], [], [], []
        sortCandidates = sort_dict_by_key(dictSynsetId)
        dictIdTitle = {}
        if type_bindings == 'labse':
            labse = SentenceBertTransformer(model_name=model_name, device="cuda")
            labse.load_model()
        elif type_bindings == 'fasttext':
            ft = fasttext.load_model(PATH_TO_FASTTEXT)
        for synset in wn.synsets:
            dictIdTitle[synset.id] = synset.title
        log_bindings = open(PATH_TO_TMP_FILE+f'log_bindings_{log_len}{file_prefix}.txt', 'w')
        for i, key in tqdm(enumerate(sortCandidates)):
            if i < log_len:
                print(key, file=log_bindings)
            if len(sortCandidates[key]) == 1:
                    w = sortCandidates[key][0]
                    p = Mapping(w.page.id,w.page.revid,w.page.title, dictIdTitle[key], key, extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
                    dictDisplay[w.page.title]=p
                    dictSortCandidates[key] = [(sortCandidates[key][0], 1)]
                    if i < log_len:
                        print(w.page.title,end='\n', file=log_bindings)
            else:
                maxP = -1
                maxagrument = 0
                lemmaSynset = get_lemma_by_title(dictIdTitle[key], dictWn)
                dictSortCandidates[key] = []
                if lemmaSynset:
                    idWn = wn.get_senses(lemmaSynset)[0].id
                    if "N" in idWn: #sometimes synset of sence N is not N
                        for elem in sortCandidates[key]:
                            if type_bindings == 'base':
                                ctxw = extractCtxW(elem.page.links, elem.page.categories)
                                lemma =  get_lemma_by_title(elem.page.title, dictWn)
                                denominator = 0
                                if lemma:
                                    numerator = score(dictWn[idWn].ctx, ctxw)
                                    for item in sortCandidates[key]:
                                        addctxw = extractCtxW(item.page.links, item.page.categories)
                                        denominator +=score(dictWn[idWn].ctx, addctxw)
                                else:
                                    badlemma.append(elem.page.title)
                                if denominator != 0:
                                    score_base = numerator / denominator
                                    dictSortCandidates[key].append((elem, score_base))
                                    if i < log_len:
                                        print(f'{elem.page.title}: {score_base}', dictWn[idWn].ctx, ctxw, 
                                              sep='\n', end='\n', file=log_bindings)
                                    if score_base > maxP:
                                        maxP = score_base
                                        maxagrument = elem
                                else:
                                    baddenominator.append(elem.page.title)
                                    dictSortCandidates[key].append((elem, -1))
                                    if i < log_len:
                                        print(f'{elem.page.title}: {-1}',dictWn[idWn].ctx, ctxw, 
                                              sep='\n', end='\n', file=log_bindings)
                            elif type_bindings == 'labse':
                                first =  remove_non_ascii_cyrillic(elem.page.title + '[SEP]' + elem.page.first_sentence)
                                sentence_hyper = f'{elem.page.title} - это то же, что и {lemmaSynset}'
                                cosine_score = labse.cosine_similarity(sentence_hyper, first)
                                dictSortCandidates[key].append((elem, cosine_score))
                                if cosine_score > maxP:
                                    maxP = cosine_score
                                    maxagrument = elem
                                if i < log_len:
                                    print(f'{elem.page.title}: {cosine_score}', first, sentence_hyper,
                                           sep='\n', end='\n', file=log_bindings)
                            elif type_bindings == 'fasttext':
                                embed_wiki_page = np.zeros(ft.get_dimension())
                                ctxw = extractCtxW(elem.page.links, elem.page.categories)
                                ctxw.add(elem.page.title)
                                ctxw = set(map(remove_non_ascii_cyrillic, ctxw))
                                for word in ctxw:
                                    embed_wiki_page += ft.get_word_vector(word)
                                average_embedding_wiki_page = embed_wiki_page / len(ctxw)
                                wn_embedding = ft.get_word_vector(lemmaSynset)
                                cosine_score = cosine_similarity(average_embedding_wiki_page, wn_embedding)
                                dictSortCandidates[key].append((elem, cosine_score))
                                if cosine_score > maxP:
                                    maxP = cosine_score
                                    maxagrument = elem
                                if i < log_len:
                                    print(f'{remove_non_ascii_cyrillic(elem.page.title)}: {cosine_score}', ctxw, lemmaSynset,
                                            sep='\n', end='\n', file=log_bindings)
                    else:
                        badidWn.append(wn.get_senses(lemmaSynset)[0].id)
                else:
                    badsynsetlemma.append(dictIdTitle[key])
                if maxP != - 1:
                    w = maxagrument
                    p = Mapping(w.page.id,w.page.revid,w.page.title,dictIdTitle[key], key,
                                extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
                    dictDisplay[w.page.title] = p
                else:
                    badmaxP.append(key)
        log_bindings.close()
        print("len(dictDisplay)", len(dictDisplay)) 
        print("len(badlemma)", len(badlemma))
        print("len(baddenominator)", len(baddenominator))
        print("len(badmaxP)", len(badmaxP))
        print("len(badsynsetlemma)", len(badsynsetlemma))
        print("len(badidWn)", len(badidWn))

        write_pkl(dictDisplay, path=PATH_TO_TMP_FILE+f'thr_dict_{file_prefix}.pkl')
        write_pkl(dictSortCandidates, path=PATH_TO_TMP_FILE+f'dictSortCandidates_thr_stage_{file_prefix}.pkl')
    else:
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+f'thr_dict_{file_prefix}.pkl')
        dictSortCandidates = read_pkl(path=PATH_TO_TMP_FILE+f'dictSortCandidates_thr_stage_{file_prefix}.pkl')     
    return dictDisplay, dictSortCandidates


def clear_bad_bindings_for_fourth_stage(dictDisplay):
    dict_wordId_in_display_and_key = defaultdict(list)
    for key, value in dictDisplay.items():
        dict_wordId_in_display_and_key[value.wordId].append((key,value))
    #удалим из dictDisplay повторяющиеся wordId
    for key, value in dict_wordId_in_display_and_key.items():
        if len(value) > 1:
            for item in value:
                del dictDisplay[item[0]]
    #удалим из dict_wordId_in_display_and_key однозначные отображенния
    dict_wordId_in_display_and_key_new = defaultdict(list)
    for key, value in dict_wordId_in_display_and_key.items():
        if len(value) > 1:
            dict_wordId_in_display_and_key_new[key] = value
    print(dict_wordId_in_display_and_key_new)

def mean_score(dict_first_method:Dict[str, List[Tuple[WikiSynset, float]]], dict_second_method:Dict[str, List[Tuple[WikiSynset, float]]], wn:RuWordNet) -> Dict[str, Mapping]:
    """
        Комбинируйте результаты, полученные по разным методикам, для повышения точности.

        Параметры:
        - dict_first_method (Dict[str, Any]): результаты первой методики, сохраненные в виде словаря.
        - dict_second_method (Dict[str, Any]): Результаты второй методики, сохраненные в виде словаря.
        - wn (WordNetCorpusReader): Экземпляр WordNetCorpusReader для получения информации о синхронизации WordNet.

        Возвращается:
        Dict[str, Union[str, Any]]: Комбинированные оценки и подробные сведения, представленные в виде словаря.
 
    """
    dictDisplay = {}
    dictIdTitle = {}
    for synset in wn.synsets:
        dictIdTitle[synset.id] = synset.title
    keys = list(dict_first_method.keys()) + list(dict_second_method.keys())
    for key in keys:
        maxP = -100
        maxagrument = None        
        if key in dict_first_method and key in dict_second_method:
            for wikisynset, score in dict_first_method[key]:
                for wikisynset2, score2 in dict_second_method[key]:
                    if wikisynset2.page.id == wikisynset.page.id:
                        if score != -1:
                            if score2 != -1:
                                score_tmp = score + score2
                            else:
                                score_tmp = score
                        else:
                            score_tmp = -1
                        if score_tmp > maxP:
                            maxagrument = wikisynset
                            maxP = score_tmp
        else:
            dict_for_iter =  dict_first_method if key in dict_first_method else  dict_second_method
            for wikisynset, score in dict_for_iter[key]:
                if score > maxP:
                    maxagrument = wikisynset
                    maxP = score        
        if maxagrument is not None:
            w = maxagrument
            p = Mapping(w.page.id,w.page.revid,w.page.title,dictIdTitle[key], key,
                        extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
            dictDisplay[w.page.title] = p
    return dictDisplay

    