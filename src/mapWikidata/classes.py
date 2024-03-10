from dataclasses import dataclass
from typing import (
    Dict,
    Any,
    Optional
)


@dataclass
class WikidataPage:
    page: Dict[str, Any]
    lemma: Optional[str]


@dataclass
class DisplaySynset2Wikidata:
    id:int
    label:int
    lemma:str
    sense_id:int
    score:float
    synset_lemma:str