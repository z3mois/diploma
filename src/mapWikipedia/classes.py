from dataclasses import dataclass
import json

@dataclass
class Page:
    '''
        A class whose element contains all the necessary information about the page
        param id, revid: id in wikipedia,
            tilte: title Wilipedia,
            meaningPage:  a pointer to whether a page is a page of value,
            multiPage: - a pointer to whether a page is a multi-valued page,
            categories: all categories from wiki page,
            links: all links on this page,
            redirect: Is the page a redirect,
            first_sentence: first sentence in page
    '''
    id: int
    revid: int
    title:str
    meaningPage: bool
    multiPage: bool
    categories: list
    links: list
    redirect:bool
    first_sentence:str
    def __eq__(self, other):
        return (self.id, self.revid, self.title, self.meaningPage,
                self.multiPage, self.categories, self.links, 
                self.redirect) == (other.id, other.revid, 
                other.title, other.meaningPage,
                other.multiPage, other.categories, other.links, 
                other.redirect)


class PageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Page):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


@dataclass
class WnCtx:
    id: int
    ctx: set
    lemmaInWn: str
    name: str
class WnCtxEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, WnCtx):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)



@dataclass
class WikiSynset():
    def __init__(self, page:Page):
        self.page = page
        self.synset = [page]

    def append(self, redirect_title:Page):
        self.synset.append(redirect_title)


@dataclass
class Mapping:
    id:int
    revid:int
    title:str
    lemma:str
    wordId:int
    ctxW:set
    first_sentense:str
    def to_dict(self):
        return {
            "id": self.id,
            "revid": self.revid,
            "title": self.title,
            "lemma": self.lemma,
            "wordId": self.wordId,
            "ctxW": list(self.ctxW),
            "first_sentense": self.first_sentense
        }

class MappingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Mapping):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)
