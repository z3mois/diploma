from tqdm import tqdm
from .classes import Mapping, WnCtx, WikiSynset
from  .extractor import Extractor
from  .utils_local import (
    wn, 
    get_normal_form,
    clear_title,
    includeTitleInWn,
    read_pkl,
    write_pkl,
    my_split,
    extractCtxW,
    get_sense_id_by_title
)
from ruwordnet import RuWordNet
from config.const import PATH_TO_TMP_FILE
from typing import List, Tuple, Dict


def unambiguous_bindings(wn:RuWordNet, dictWn:Dict[str, WnCtx], wiki:List[WikiSynset], mode:str='read') -> Tuple[Dict[str, Mapping], List[WikiSynset]]:
    dictDisplay = {} # словарь отображений
    new_wiki = [] #будут все викисинсеты, кроме однозначных
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
    write_pkl(dictDisplay, PATH_TO_TMP_FILE+'fst.pkl')
    return dictDisplay, new_wiki
    