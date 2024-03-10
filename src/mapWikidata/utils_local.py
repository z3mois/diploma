from emoji import UNICODE_EMOJI
from collections import deque
import re
from ..mapWikipedia import (
    get_normal_form
)


chars = set(["-", " ", ':', '(', ')', '.', ',', '{', '}', '?', '!', ';', '\"', "\'", '+', '=', '*'])

def is_lat(s:str) -> bool:
    '''
        Checking that the string consists only of Unicode characters and numbers and other common characters
        param:
            s: str for check
        return:
            true if all chars in string - unicode
    '''
    for char in s:
        if char.isalpha() or char.isdigit() and (char not in UNICODE_EMOJI) or char in  chars:
            pass
        else:
            return False
    return True


def clean_text(text:str) -> str:
    '''
    This function takes a string of text and performs cleaning operations to 
    remove non-alphanumeric characters, digits, extra whitespaces, and converts the text to lowercase.
    Parameters:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    '''
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip().lower()
    return cleaned_text


def extract_ctx_wikidata(article:dict) -> set:
    '''
        This function extracts the context of an article from its description in Russian.
        Parameters:
            article (dict): A dictionary representing the article, typically containing a 'descriptions' field with a description in Russian.

        Returns:
            set: A set of normalized tokens representing the context of the article.
    '''
    ctx = set()
    if 'ru' in article['descriptions']:
        for token in clean_text(article['descriptions']['ru']).split():
            ctx.add(get_normal_form(token).lower())
    return ctx


def get_score(elem1:set, elem2:set) -> float:
    '''
        This function calculates the similarity score between two sets of elements based on a heuristic approach.

        Parameters:
            elem1 (set): The first set of elements.
            elem2 (set): The second set of elements.

        Returns:
            float: The similarity score between the two sets, ranging from 0 to 1
    '''
    return (len((elem1 & elem2)) + 1) / (len(elem1) + len(elem2) + 1)