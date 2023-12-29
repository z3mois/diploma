
from config.const import RUWORDNET_PATH
import pymorphy2
from functools import lru_cache
from ruwordnet import RuWordNet
from typing import Any
import pickle

@lru_cache(maxsize=200000)
def get_normal_form(word:str)-> str:
    '''
        Return normal form word
        param word: The word from which we want to get a normal form
    '''
    return morph_analizer.parse(word)[0].normal_form


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
            clera title
    '''
    title = title.upper()
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
    if "Ё" in title:
        return includeTitleInWn(all_senses, title.replace("Ё","Е"))
    return False

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
