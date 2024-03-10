from .classes import WikidataPage
from typing import (
    List,
    Dict
)
from ruwordnet import RuWordNet
from collections import defaultdict
from tqdm import tqdm
from ..mapWikipedia import (
    get_lemma_by_title,
    write_pkl,
    read_pkl,
    create_info_about_sense,
    wn,
    remove_non_ascii_cyrillic
)
from config.const import PATH_TO_TMP_FILE


def create_dict_candidates(articles:List[WikidataPage]=None,
                            mode:str='read') -> Dict[str, WikidataPage]:
    """
        Create candidates for bindings
    """
    if mode!='read':
        dictWn = create_info_about_sense()
        canidates = defaultdict(list)
        for wp in tqdm(articles):
            article, lemma = wp.page, wp.lemma
            if 'ru' in article['label']:
                if lemma:
                    synset_base_on_lemma = wn.get_synsets(lemma)
                    if synset_base_on_lemma:
                        for synset in wn.get_synsets(lemma):
                            canidates[synset.id].append((article, lemma))
                else:
                    lemma_new = get_lemma_by_title(article['label']['ru'], dictWn)
                    if lemma_new:
                        for synset in wn.get_synsets(lemma_new):
                            canidates[synset.id].append((article, lemma_new))
        write_pkl(canidates, path=PATH_TO_TMP_FILE+'candidates_wikidata.pkl')
    else:
        canidates = read_pkl(path=PATH_TO_TMP_FILE+'candidates_wikidata.pkl')
    return canidates