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
    remove_non_ascii_cyrillic
)
from config.const import PATH_TO_TMP_FILE
from .utils_local import (
    get_score,
    extract_ctx_wikidata
)


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


def bindings(candidates:Dict[str, List[WikidataPage]]=None, mode:str='read')-> Dict[str, List[float]]:
    '''
        This is the function of linking a wikidata page to a specific synset (similar to wikipedia, here we immediately assume that all cases are ambiguous)

        Parameters:
            candidates (Dict[str, List[Wikidata Page]], optional): A dictionary where keys are synset IDs and values are lists of WikidataPage objects. Defaults to None.
            mode (str, optional): The mode of operation. Defaults to 'read'.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are synset IDs and values are lists
    '''
    if mode != 'read':
        dictWn = create_info_about_sense()
        score_dict = defaultdict(list)
        for _, (title_synset, candidatess) in tqdm(enumerate(candidates.items())):
            for wikipage in candidatess:
                candidate, lemma = wikipage.page, wikipage.lemma 
                if 'N' in wn[lemma][0].id:
                    synset_ctx = dictWn[wn[lemma][0].id].ctx
                    article_ctx = extract_ctx_wikidata(candidate)
                    article_ctx.update([candidate['label']['ru'].lower(), lemma.lower()])
                    score = get_score(article_ctx, synset_ctx)
                    score_dict[title_synset].append(score)
                else:
                    score_dict[title_synset].append(0.0)
        write_pkl(score_dict, path=PATH_TO_TMP_FILE+'score_wikidata.pkl')
    else:
        score_dict = read_pkl(path=PATH_TO_TMP_FILE+'score_wikidata.pkl')
    return score_dict


def take_mapping(score:Dict[str, List[float]]=None, candidates:Dict[str, List[WikidataPage]]=None, 
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
    if mode != 'read':
        res_score = {}
        for _, (title_synset, candidatess) in tqdm(enumerate(candidates.items())):
            sorted_lst = sorted(zip(candidatess, score[title_synset]), key=lambda x: x[1], reverse=True)
            if sorted_lst and 'N' in title_synset:
                best:Tuple[WikidataPage, float] = sorted_lst[0]
                res_score[title_synset] = DisplaySynset2Wikidata(best[0].page['id'], best[0].page['label'], best[0].lemma, wn[best[0].lemma][0].id, best[1], title_synset)
        write_pkl(res_score, path=PATH_TO_TMP_FILE+'bindings.pkl')
    else:
        res_score = read_pkl(path=PATH_TO_TMP_FILE+'bindings.pkl')
    return res_score
