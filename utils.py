from emoji import UNICODE_EMOJI
from const import RUWORDNET_PATH
from emoji import UNICODE_EMOJI
import pymorphy2
from functools import lru_cache
from ruwordnet import RuWordNet


def is_lat(s):
    for char in s:
        if char.isalpha() or char.isdigit() or char == " " or char == "-" or char == ':':
            pass
        else:
            return False
    return not(s in UNICODE_EMOJI)


def my_split(x):
    s = ""
    for i in x:
        if i!= "(" and i != ")":
            s +=i
        elif i  == ")":
            s += ""
        else:
            s += ","
    return s


def TitleInWn(all_senses, title):
    title = title.lower()
    title = title.replace("—", "-")
    title = title.replace(",", "")
    if title in all_senses:
        return title
    if "(" in title:
        text = my_split(title).split(",")
        if text[0] in all_senses:
            return text[0]
    text = my_split(title).split(",")
    lemmatized = " ".join([get_normal_form(word).lower()
                for word in text[0].split()])
    if lemmatized in all_senses:
        return lemmatized
    if "ё" in title:
        return TitleInWn(all_senses, title.replace("ё","е"))
    return None

@lru_cache(maxsize=200000)
def get_normal_form(word):
    return morph_analizer.parse(word)[0].normal_form
morph_analizer = pymorphy2.MorphAnalyzer()
wn = RuWordNet(filename_or_session=RUWORDNET_PATH)