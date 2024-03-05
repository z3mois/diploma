
from config.const import RUWORDNET_PATH
import pymorphy2
from functools import lru_cache
from ruwordnet import RuWordNet
from typing import Any, List, Union, Dict
from .classes import WnCtx, WikiSynset
import pickle
import re
import numpy as np


@lru_cache(maxsize=200000)
def get_normal_form(word:str)-> str:
    '''
        Return normal form word
        param word: The word from which we want to get a normal form
    '''
    return morph_analizer.parse(word)[0].normal_form.lower()


morph_analizer = pymorphy2.MorphAnalyzer()
wn = RuWordNet(filename_or_session=RUWORDNET_PATH)


def my_split(x:str)->str:
    '''
        Its own equivalent for clearing text
    '''
    s = ""
    for i in x:
        if i!= "(" and i != ")":
            s +=i
        elif i  == ")":
            s += ""
        else:
            s += ","
    return s


def clear_title(title:str) -> str:
    '''
        Special article cleaning function
        param:
            title: title for clear
        return:
            clear title
    '''
    title = title.lower()
    title = title.replace("—", "-")
    title = title.replace(",", "")
    return title


def includeTitleInWn(all_senses:set[str], title:str)->bool:
    '''
        Check if there is an article title in RuWordNet
        param
            all_senses: a list of sense from RuWordNet that have been lemmatized by words
            title: title article< which we check
    '''
    if title in all_senses:
        return True
    if "(" in title:
        text = my_split(title).split(",")
        if text[0] in all_senses:
            return True
    text = my_split(title).split(",")
    lemmatized = " ".join([get_normal_form(word)
                for word in text[0].split()])
    if lemmatized.upper() in all_senses:
        return True
    if "ё" in title:
        return includeTitleInWn(all_senses, title.replace("ё","е"))
    return False

def get_sense_id_by_title(key:str) -> Union[None, str]:
    '''
        We get a string as input and search for the session id in RuWordNet for it
        param
            key: input string
        return
            None if you haven't found the key otherwise the key
    '''
    ch = " "
    senses = wn.get_senses(key)
    if len(senses) > 0:
        # print(1)
        return senses[0].id
    if "(" in key:
        text = my_split(key).split(",")
        text[0] = text[0].rstrip(ch)
        senses = wn.get_senses(text[0])
        if  len(senses) > 0:
            # print(2)
            return senses[0].id
    text = my_split(key).split(",")
    lemmatized = " ".join([get_normal_form(word) for word in text[0].split()])
    lemmatized = lemmatized.rstrip(ch)
    senses = wn.get_senses(lemmatized)
    if len(senses) > 0:
        # print(3)
        return senses[0].id
    if "ё" in key:
        return get_sense_id_by_title(key.replace("ё", "e"))
    return None


def get_lemma_by_id_sense(id:str, dictWn:Dict[str, WnCtx]) -> Union[None, str]:
    '''
        Getting an ID sense lemma from a RuWordNet
        param
            id: sense id - format '000-N-000'
            dictWn: castom dict from id to info about sense
    '''
    return dictWn[id].lemmaInWn if id in dictWn else None


def get_lemma_by_title(title:str, dictWn:Dict[str, WnCtx]) -> Union[None, str]:
    '''
        Getting an word  lemma from a RuWordNet
        param
            title: some title
            dictWn: castom dict from id to info about sense
    '''
    title_with_s = clear_title(title)
    id = get_sense_id_by_title(title_with_s)
    if id:
        return get_lemma_by_id_sense(id, dictWn)
    return None


def read_pkl(path:str) -> Any:
    '''
        Read .pkl  file and returns what is in it
        params
            path: where are we loading it
    '''
    file = open(path, "rb")
    unpickler = pickle.Unpickler(file)
    varibles = unpickler.load()
    file.close()
    return varibles


def write_pkl(varibles:Any, path:str)->None:
    '''
        Write varibles  a .pkl file along the path
        params
            varibles: what we save
            path: where are we saving it
    '''
    file = open(path, "wb")
    pickle.dump(varibles, file=file)
    file.close()


def extractCtxW(links:List[str], categories:List[str])->set[str]:
    '''
        find context wikipage = links + categories
        param:
            links: all links on the page
            categories: all categories on the page
    '''
    ctx = set()
    for link in links:
        ctx.add(" ".join([get_normal_form(word) for word in link.split()]))
    for elem in categories:
        ctx.add(" ".join([get_normal_form(word) for word in elem.split()]))
    return ctx



def count1(value:WikiSynset, array:List[WikiSynset]) -> int:
    '''
        Count elem in array(count by id page)
    '''
    count:int = 0
    for elem in array:
        if elem.page.id == value.page.id:
            count += 1
    return count


def score(ctxS:set, ctxW:set) -> int:
    '''
        find len of intersection two set
        param:
            set1, set2
        return:
            len of intersection
    '''
    return len(ctxS.intersection(ctxW))

def sort_dict_by_key(dct:Dict[Any, Any]) -> Dict[Any, Any]:
    '''
        sort dict by dict key
    '''
    return dict(sorted(dct.items(), key=lambda x: x[0]))

def remove_non_ascii_cyrillic(text):
    """Удаляет из текста все символы, не являющиеся ни ASCII, ни русскими буквами."""
    return re.sub(r'[^\x00-\x7Fа-яА-ЯёЁ]+', '', text)


def cosine_similarity(emb1: np.ndarray[np.float64], emb2: np.ndarray[np.float64]) -> float:
    '''
    Calculating the cosine proximity between two vord represetaion(word or sent)
        Param: two embed
        return: cosine similarity range, -1 to 1
    '''    
    return np.dot(emb1, emb2) /(np.linalg.norm(emb2) * np.linalg.norm(emb1))