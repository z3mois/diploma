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