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
)
from .model import SentenceBertTransformer
from collections import defaultdict
from ruwordnet import RuWordNet
from config.const import PATH_TO_TMP_FILE
from typing import List, Tuple, Dict


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
        print("Start writing in file")
        write_pkl(dictDisplay, PATH_TO_TMP_FILE+'fst_dict.pkl')
        write_pkl(new_wiki, PATH_TO_TMP_FILE+'wiki_after_fst_stage.pkl')
        print("Successful recording")
    else:
        print("Start reading from file")
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+'fst_dict.pkl')
        new_wiki = read_pkl(path=PATH_TO_TMP_FILE+'wiki_after_fst_stage.pkl')
        print("Successful reading")
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
        print('Dict candidates creatted')
        print("Start writing in file")
        write_pkl(dictLemmaInIndex, PATH_TO_TMP_FILE+name)
        print("Successful recording")
    else:
        print("Start reading from file")
        dictLemmaInIndex = read_pkl(path=PATH_TO_TMP_FILE+name)
        print("Successful reading")
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
        print(f"Start writing in file = {PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl'}")
        write_pkl(dictSynsetId, PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl')
        print("Successful recording")
    else:
        print(f"Start reading from file = {PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl'}")
        dictSynsetId = read_pkl(path=PATH_TO_TMP_FILE+'candidates_for_multi_stage.pkl')
        print("Successful reading")
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
        print("Start writing in file")
        write_pkl(dictDisplay, PATH_TO_TMP_FILE+'snd_dict.pkl')
        write_pkl(wiki_for_multi, PATH_TO_TMP_FILE+'wiki_after_snd_stage.pkl')
        print("Successful recording")
    else:
        print("Start reading from file")
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+'snd_dict.pkl')
        wiki_for_multi = read_pkl(path=PATH_TO_TMP_FILE+'wiki_after_snd_stage.pkl')
        print("Successful reading")
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
                         wn:RuWordNet=None, dictWn:Dict[str, WnCtx]=None, type_bindings:str='base', mode:str='read'):
    '''
        The stage of linking to a specific synset of a certain Wikipedia article
        param:
            dictDisplay:    answer dict after second stage
            dictSynsetId: dictSynsetId in format synset.id(121-N) -> List[WikiSynset]
            wn: RuWordNet 
            dictWn: dict from id ('101-N-100') to info about sense  
            type_bindings: base(method from Archiv), labse(check only cosine similarity)
            mode: read or overwrite
        return:
           Answer dict after multi stage
    '''
    if mode != 'read':
        dictSortCandidates = {}
        badlemma, baddenominator, badmaxP, badsynsetlemma, badidWn = [], [], [], [], []
        sortCandidates = sort_dict_by_key(dictSynsetId)
        dictIdTitle = {}
        labse = SentenceBertTransformer(device="cuda")
        labse.load_model()
        for synset in wn.synsets:
            dictIdTitle[synset.id] = synset.title
        for key in tqdm(sortCandidates):
            if len(sortCandidates[key]) == 1:
                    w = sortCandidates[key][0]
                    p = Mapping(w.page.id,w.page.revid,w.page.title, dictIdTitle[key], key, extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
                    dictDisplay[w.page.title]=p
                    dictSortCandidates[key] = [(sortCandidates[key][0], 1)]
            else:
                maxP = -1
                maxagrument = 0
                lemmaSynset = get_lemma_by_title(dictIdTitle[key], dictWn)
                dictSortCandidates[key] = []
                if lemmaSynset:
                    idWn = wn.get_senses(lemmaSynset)[0].id
                    if "N" in idWn: #sometimes synet of sence N is not N
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
                                    dictSortCandidates[key].append((elem, numerator / denominator))
                                    if numerator / denominator > maxP:
                                        maxP = numerator / denominator
                                        maxagrument = elem
                                else:
                                    baddenominator.append(elem.page.title)
                                    dictSortCandidates[key].append((elem, -1))
                            elif type_bindings == 'labse':
                                cosine_score = labse.cosine_similarity(f'{elem.page.title} - это тоже, что и {lemmaSynset}', elem.page.title + '[SEP]' + elem.page.first_sentence)
                                dictSortCandidates[key].append((elem, cosine_score))
                                if cosine_score > maxP:
                                    maxP = cosine_score
                                    maxagrument = elem
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
        print(dictDisplay)
        print("len(dictDisplay)", len(dictDisplay)) 
        print("len(badlemma)", len(badlemma))
        print("len(baddenominator)", len(baddenominator))
        print("len(badmaxP)", len(badmaxP))
        print("len(badsynsetlemma)", len(badsynsetlemma))
        print("len(badidWn)", len(badidWn))

        print("Start writing in file")
        write_pkl(dictDisplay, PATH_TO_TMP_FILE+f'thr_dict_{type_bindings}.pkl')
        write_pkl(dictSortCandidates, PATH_TO_TMP_FILE+f'dictSortCandidates_thr_stage_{type_bindings}.pkl')
        print("Successful recording")
    else:
        print("Start reading from file")
        dictDisplay = read_pkl(path=PATH_TO_TMP_FILE+f'thr_dict_{type_bindings}.pkl')
        dictSortCandidates = read_pkl(path=PATH_TO_TMP_FILE+f'dictSortCandidates_thr_stage_{type_bindings}.pkl')
        print("Successful reading")        
    return dictDisplay, dictSortCandidates


def multi_bindings_stage_labse(dictDisplay:Dict[str, Mapping], dictSynsetId: Dict[str, List[WikiSynset]], wn:RuWordNet, dictWn:Dict[str, WnCtx], mode:str='read'):
    badlemma, badmaxP, badsynsetlemma, badidWn = [], [], [], []
    print(len(dictDisplay), len(dictSynsetId))
    dictSortCandidates = {}
    sortCandidates = sort_dict_by_key(dictSynsetId)
    dictSortCandidates = {}
    dictIdTitle = {}
    labse = SentenceBertTransformer(device="cuda")
    labse.load_model()
    i = 0
    for synset in wn.synsets:
        dictIdTitle[synset.id] = synset.title
    for key in tqdm(sortCandidates):
        if len(sortCandidates[key]) == 1:
                w = sortCandidates[key][0]
                p = Mapping(w.page.id,w.page.revid,w.page.title, dictIdTitle[key], key,extractCtxW(w.page.links, w.page.categories), w.page.first_sentence)
                dictDisplay[w.page.title]=p
                dictSortCandidates[key] = [(sortCandidates[key][0], 1)]
        else:
            maxP = -1
            maxagrument = 0
            lemmaSynset = get_lemma_by_title(dictIdTitle[key], dictWn)
            dictSortCandidates[key] = []
            if lemmaSynset:
                idWn = wn.get_senses(lemmaSynset)[0].id
                if "N" in idWn: #sometimes synet of sence N is not N
                    for elem in sortCandidates[key]:
                        i +=1
                        cosine_score = labse.cosine_similarity(f'{elem.page.title} - это тоже, что и {lemmaSynset}', elem.page.title + '[SEP]' + elem.page.first_sentence)
                        # print(elem.page.title + '[SEP]' +elem.page.first_sentence, f'{elem.page.title} - это тоже, что и {lemmaSynset}', cosine_score, sep='::::')
                        dictSortCandidates[key].append((elem, cosine_score))
                        if cosine_score > maxP:
                            maxP = cosine_score
                            maxagrument = elem
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
        # if i > 10:
        #     return
    print("len(dictDisplay)",len(dictDisplay)) 
    print("len(badlemma)",len(badlemma))
    print("len(badmaxP)",len(badmaxP))
    print("len(badsynsetlemma)",len(badsynsetlemma))
    print("len(badidWn)",len(badidWn))

    print("Start writing in file")
    write_pkl(dictDisplay, PATH_TO_TMP_FILE+'thr_dict_labse.pkl')
    write_pkl(dictSortCandidates, PATH_TO_TMP_FILE+'dictSortCandidates_thr_stage_labse.pkl')
    print("Successful recording")