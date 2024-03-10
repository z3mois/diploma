import os
import bz2
import os.path
import re
from tqdm import tqdm
from .classes import Page, WikiSynset, WnCtx
from  .extractor import Extractor
from  .utils_local import (
    wn, 
    get_normal_form,
    clear_title,
    includeTitleInWn,
    read_pkl,
    write_pkl,
    my_split
)
from ruwordnet import RuWordNet
from config.const import PATH_TO_TMP_FILE, PATH_TO_WN_XML
from typing import List, Tuple, Dict
from xml.dom import minidom

acceptedNamespaces = ['w', 'wiktionary', 'wikt']
templateNamespace = ''
tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')


def collect_pages(text):
    """
    :param text: the text of a wikipedia file dump.
    """
    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    page = []
    id = ''
    revid = ''
    last_id = ''
    inText = False
    redirect = False
    redirect_page = ''
    for line in text:
        if '<' not in line:     # faster than doing re.search()
            if inText:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id: # <revision> <id></id> </revision>
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'redirect':
            redirect = True
            redirectRE = re.compile(r'title=\"(.*?)\" />')
            redirect_page = re.findall(redirectRE, line)[0]
        elif tag == 'text':
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            colon = title.find(':')
            if (colon < 0 or (title[:colon] in acceptedNamespaces) and id != last_id and
                    not redirect and not title.startswith(templateNamespace)):
                yield (id, revid, title, page, redirect_page,redirect)
                last_id = id
            id = ''
            revid = ''
            page = []
            inText = False
            redirect = False
            redirect_page=''

def decode_open(filename, mode='rt', encoding='utf-8'):
    """
    Open a file, decode and decompress, depending on extension `gz`, or 'bz2`.
    :param filename: the file to open.
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.gz':
        import gzip
        return gzip.open(filename, mode, encoding=encoding)
    elif ext == '.bz2':
        return bz2.open(filename, mode=mode, encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)


def extract_cat(text) -> List[str]:
    """
    Find categories in text wikipedia page
    :param text: text wikipedia page.
    """
    matcher=re.compile(r"Категория:\s?([А-Яа-я\s?]+)")
    return matcher.findall(text)


def extract_links(text:str) -> List[str]:
    """
    Find links in text wikipedia page
    :param text: text wikipedia page.
    """
    matcher=re.compile(r"[\[\[]([А-Яа-я\s?]+)[\|,\]\]]")
    return matcher.findall(text)


def extract_first_links(text:str) -> List[str]:
    """
    Find first links in every paragraph in text wikipedia page
    :param text: text wikipedia page.
    """
    matcher=re.compile(r"[\[\[]([А-Яа-я\s?]+)[\|,\]\]]")
    answer = []
    for elem in text.split("\n"):
        item = matcher.findall(elem)
        if len(item) > 0:
            answer.append(item[0])
    return answer


def extractCtxS(wn:RuWordNet, lemma:str)-> set[str]:
    '''
    Find conetext of synset
    :param 
        wn: RuWordNet,
        lemma: synset lemma        
    '''
    ctx_s = set()
    #synonymy
    for sense in wn.get_synsets(lemma):
        for synonymy in sense.senses:
            ctx_s.update(my_split(synonymy.lemma).split(","))
    #Hypernymy/Hyponymy
    for sense in wn.get_senses(lemma):
        for hypernyms in sense.synset.hypernyms:
            ctx_s.update(my_split(hypernyms.title).split(","))
    for sense in wn.get_senses(lemma):
        for hyponyms in sense.synset.hyponyms:
            ctx_s.update(my_split(hyponyms.title).split(","))
    #Sisterhood:
    for sense in wn.get_senses(lemma):
        for hypernyms in sense.synset.hypernyms:
            for sister in hypernyms.hyponyms:
                ctx_s.update(my_split(sister.title).split(","))
    return ctx_s


def read_dump(path:str, mode:str = 'read') -> Tuple[List[Page], Dict[str, List[Page]], Dict[str, List[str]]]:
    """
    Read wikipedia dump and get list pages, redirected dicts if mode == over_read,
    else read this veribles
    :param path: path to dump,
            mode: mode to do(read or over_read).
    """
    if mode == 'over_read':
        input = decode_open(path)
        i = 0
        dictRedirect = {}
        pages = []
        redirectcount = 0
        dictPageRedirect = {}
        i = 0
        print('Start reading file')
        for id, revid, title, page, redirect_page, redirect in tqdm(collect_pages(input)):
            i += 1
            text = ''.join(page)
            text_lower = text.lower()
            multiPage = False
            if text_lower.find('{{другие значения') != -1:
                multiPage = True
            elif title.find("(") != -1 and (not "значения" in title.lower())and (not "значение" in title.lower()):
                multiPage = True
            elif text_lower.find("{{перенаправление") != -1:
                multiPage = True
            elif text_lower.find("{{другое значение") != -1:
                multiPage = True
            elif text_lower.find("{{значения") != -1:
                multiPage = True
            elif text_lower.find("{{redirect-multi") != -1:
                multiPage = True
            elif text_lower.find("{{redirect-multi") != -1:
                multiPage = True
            elif text_lower.find("{{see also") != -1:
                multiPage = True
            elif text_lower.find("{{о|") != -1:
                multiPage= True
            elif text_lower.find("{{список однофамильцев}}") != -1:
                multiPage= True
            categories = extract_cat(text)
            
            meaningPage = False
            if ("значения" in title.lower()) or ("значение" in title.lower()):
                meaningPage = True
            elif text_lower.find('{{неоднозначность') != -1:
                meaningPage = True
            elif text_lower.find('{{многозначность') != -1:
                meaningPage = True
            elif text_lower.find('{{disambig') != -1:
                meaningPage = True
            # redirects = extract_redirects(text)
            links =[]
            if not meaningPage:
                links = extract_links(text)
            else:
                links = extract_first_links(text)   
            first_sentense = ""
            if not redirect_page:
                ext = Extractor(id,revid,"",title,page)
                first_sentense = "\n".join(ext.clean_text(text)).split(".")[0] 
            if len(redirect_page) > 0:
                if redirect_page not in dictRedirect:
                    dictRedirect[redirect_page] = []
                    dictPageRedirect[redirect_page] = []
                dictPageRedirect[redirect_page].append(Page(id,revid,title,meaningPage,multiPage,categories,links,redirect, first_sentense))
                dictRedirect[redirect_page].append(title)
                redirectcount +=1
            pages.append(Page(id,revid,title,meaningPage,multiPage,categories,links,redirect, first_sentense))
        input.close()
        print('Finish read Wikipedia')
        write_pkl(pages, path=PATH_TO_TMP_FILE + "ctxw.pkl")
        write_pkl(dictRedirect, path=PATH_TO_TMP_FILE + PATH_TO_TMP_FILE + "dr.pkl")
        write_pkl(dictPageRedirect, path=PATH_TO_TMP_FILE + PATH_TO_TMP_FILE + "drp.pkl")
    else:
        pages = read_pkl(path=PATH_TO_TMP_FILE + "ctxw.pkl")
        dictRedirect = read_pkl(path=PATH_TO_TMP_FILE + "dr.pkl")
        dictPageRedirect = read_pkl(path=PATH_TO_TMP_FILE + "drp.pkl")
    return pages, dictPageRedirect, dictRedirect

def create_wikisynset(pages:List[Page]=None, dictPageRedirect:Dict[str, List[Page]]=None, mode:str='read') -> List[WikiSynset]:
    '''
        Creata special list of wikisynset
        param page: list of Page wikipedia,
            dictPageRedirect: displaying from the page to all referenced ones
            mode: type of to do(read or overwrite), if mode==overwrite you can use wiki=create_wikisynset()
        return:
            wiki: list of WikiSynset
    '''
    if mode != 'read':
        wiki = []
        meaningPageCounter = 0
        multiPageCounter = 0
        includeTitle = 0
        all_senses = set([' '.join([get_normal_form(w) for w in s.lemma.lower().split()]) for s in wn.senses])
        hashDict = {}
        print('Start create hash')
        for index in tqdm(range(len(pages)-1)):
            hashDict[pages[index].title.lower()] = index
        print('Create hash finished')
        #пройтись по всем значения со страницы-значения и всем значения поставить мульти
        i = 0
        print('Start add multiPage label based on meaningPage')
        for index in tqdm(range(len(pages)-1)):
            if pages[index].meaningPage:
                for link in pages[index].links:
                    if link.lower() in hashDict:
                        pages[hashDict[link.lower()]].multiPage = True
                        i += 1
        print(f'Was added multiPage label {i}')
        print('Start create WikiSynset list')
        for page in tqdm(pages):
            title_clear = clear_title(page.title)
            if page.redirect:
                if includeTitleInWn(all_senses, title_clear):
                    includeTitle += 1
                continue
            wikiSyn = WikiSynset(page)
            if page.title in dictPageRedirect:
                for redirect in dictPageRedirect[page.title]:
                    wikiSyn.append(redirect)
            if page.meaningPage:
                meaningPageCounter += 1
            if page.multiPage:
                multiPageCounter += 1
            wiki.append(wikiSyn)
            if includeTitleInWn(all_senses, title_clear):
                includeTitle += 1
        print('WikiSynset list was created')
        print(f'Count wikipage with title in RuWordNet {includeTitle}')
        print(f'Count meaning page {meaningPageCounter}')
        print(f'Count multi page {multiPageCounter}')
        write_pkl(wiki, path=PATH_TO_TMP_FILE+'WikiSynset.pkl')
    else:
        wiki=read_pkl(path=PATH_TO_TMP_FILE+'WikiSynset.pkl')
    return wiki

def create_info_about_sense(mode:str='read') -> Dict[str, WnCtx]:
    '''
        Parse file with all sense in RuWordNet and create dict from synset_id in context thos synset
        param:
            mode: read or overwrite
    '''
    if mode != 'read':
        mydoc = minidom.parse(PATH_TO_WN_XML)
        items = mydoc.getElementsByTagName('sense')
        countWn = 0
        dictWn = {}
        print('Start to read file and find synset')
        for elem in tqdm(items):
            countWn +=1
            text = elem.attributes['name'].value
            text_id = elem.attributes["id"].value
            lemma = elem.attributes["lemma"].value
            ctx_s = extractCtxS(wn, lemma)
            ctx = set()
            for ctx_elem in ctx_s:
                ctx.add(" ".join([get_normal_form(word) for word in ctx_elem.split()]))
            dictWn[text_id] = WnCtx(text_id, ctx, lemma, text)
        print(f'Count sene = {countWn}')
        print('Finish extract data ')
        write_pkl(dictWn, path=PATH_TO_TMP_FILE+'ctxS.pkl')
    else:
        dictWn=read_pkl(path=PATH_TO_TMP_FILE+'ctxS.pkl')
    return dictWn

    