from src.mapWikipedia import read_dump, create_wikisynset,create_info_about_sense, wn, unambiguous_bindings
from config.const import DAMP_OF_WIKIPEDIA_PATH
dictWn = create_info_about_sense()
wiki = create_wikisynset()
unambiguous_bindings(wn, dictWn, wiki)
