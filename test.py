from src.mapWikipedia import (
    read_dump,
    create_wikisynset,
    create_info_about_sense,
      wn, unambiguous_bindings, 
      create_candidates_index_dict, 
      add_multi_flag, second_stage_bindings)
from config.const import DAMP_OF_WIKIPEDIA_PATH

dictWn = create_info_about_sense()
wiki = create_wikisynset()

dictt, new_wiki = unambiguous_bindings(wn, dictWn, wiki, mode='read')


dictLemm = create_candidates_index_dict()

add_multi_flag(new_wiki, dictLemm)

dictt, new_wiki = second_stage_bindings(wn, dictWn, new_wiki, dictt, mode='overwrite')
