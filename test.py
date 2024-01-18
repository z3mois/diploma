from src.mapWikipedia import read_dump, create_wikisynset,create_info_about_sense, wn, unambiguous_bindings, create_candidates_index_dict
from config.const import DAMP_OF_WIKIPEDIA_PATH

dictWn = create_info_about_sense()
wiki = create_wikisynset()

dictt, new_wiki = unambiguous_bindings(wn, dictWn, wiki, mode='read')


create_candidates_index_dict(new_wiki, dictWn, mode='FstCreate')
