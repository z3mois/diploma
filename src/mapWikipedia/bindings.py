from tqdm import tqdm
from .classes import Mapping, WnCtx, WikiSynset
from  .utils_local import (
    clear_title,
    read_pkl,
    write_pkl,
    extractCtxW,
    get_sense_id_by_title,
    get_lemma_by_title
)
from ruwordnet import RuWordNet
from config.const import PATH_TO_TMP_FILE
from typing import List, Tuple, Dict


def unambiguous_bindings(wn:RuWordNet, dictWn:Dict[str, WnCtx], wiki:List[WikiSynset], mode:str='read') -> Tuple[Dict[str, Mapping], List[WikiSynset]]:
    dictDisplay = {} # словарь отображений
    new_wiki = [] #будут все викисинсеты, кроме однозначных
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


def create_candidates_index_dict(wiki:List[WikiSynset], dictWn:Dict[str, WnCtx], mode:str='read') -> Dict[str, int]:
    '''
        Create dict_candidates in format Lemma -> idx in List[WikiSynset]
        param:
            wiki: List candidates
            dictWn: dictWn: castom dict from id to info about sense
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
        write_pkl(dictLemmaInIndex, PATH_TO_TMP_FILE+'lst_candidates_after_fst_stage.pkl')
        print("Successful recording")
    else:
        print("Start reading from file")
        dictLemmaInIndex = read_pkl(path=PATH_TO_TMP_FILE+'lst_candidates_after_fst_stage.pkl')
        print("Successful reading")
    return dictLemmaInIndex


    