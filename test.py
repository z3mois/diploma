from src.mapWikipedia import (
    read_dump,
    create_wikisynset,
    create_info_about_sense,
      wn, unambiguous_bindings, 
      create_candidates_index_dict, 
      add_multi_flag, second_stage_bindings,
      create_candidates_for_multi_stage)
from config.const import DAMP_OF_WIKIPEDIA_PATH

dictWn = create_info_about_sense()
# wiki = create_wikisynset()

# dictt, new_wiki = unambiguous_bindings(wn, dictWn, wiki, mode='Ovewrite')

# dictLemm = create_candidates_index_dict(new_wiki, dictWn, mode='overwrite')

# add_multi_flag(new_wiki, dictLemm)

dictt, new_wiki = second_stage_bindings()

dictLemmNew = create_candidates_index_dict(name='lst_candidates_after_snd_stage.pkl', mode='read')


print(create_candidates_for_multi_stage(new_wiki, wn, dictWn, dictLemmNew, mode='read')['124964-N'][0])