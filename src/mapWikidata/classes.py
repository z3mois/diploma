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
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "lemma": self.lemma,
            "sense_id": self.sense_id,
            "score":  float(self.score),
            "synset_lemma": self.synset_lemma,
        }