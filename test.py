from src.mapWikipedia import (
    read_dump,
    create_wikisynset,
    create_info_about_sense,
      wn, unambiguous_bindings, 
      create_candidates_index_dict, 
      add_multi_flag, second_stage_bindings,
      create_candidates_for_multi_stage, 
      delete_double_in_candidates,
      multi_bindings_stage,
      scoring,
      scoring_wikipedia_bindings, clear_bad_bindings_for_fourth_stage, mean_score
)
from src.mapWikidata import (

    create_dataset_for_wikidata,extract_title_with_title_in_RuWordNet, extract_referring_pages, create_dict_candidates, bindings, take_mapping, scoring
)
from config.const import DAMP_OF_WIKIPEDIA_PATH, DAMP_OF_WIKIDATA_PATH, GOLD_DATA
import pandas as pd
pd.options.mode.chained_assignment = None

# wiki = create_wikisynset()

# dictt, new_wiki = unambiguous_bindings(wn, dictWn, wiki, mode='Ovewrite')

# dictLemm = create_candidates_index_dict(new_wiki, dictWn, mode='overwrite')

# add_multi_flag(new_wiki, dictLemm)
dictWn = create_info_about_sense()
dictt, new_wiki = second_stage_bindings()

dictLemmNew = create_candidates_index_dict(name='lst_candidates_after_snd_stage.pkl', mode='read')


dict_candidtes = create_candidates_for_multi_stage(new_wiki, wn, dictWn, dictLemmNew, mode='read')

dict_candidtes_update = delete_double_in_candidates(dict_candidtes)

dicttFinal, dictSortCandidates = multi_bindings_stage(dictt, dict_candidtes_update, wn, dictWn, type_bindings='labse', model_name='setu4993/LaBSE', mode='read')
dicttFinal_base, dictSortCandidates_base = multi_bindings_stage(dictt, dict_candidtes_update, wn, dictWn, type_bindings='base', model_name='setu4993/LaBSE', mode='read')


print(scoring_wikipedia_bindings(dictt, dicttFinal, GOLD_DATA))
print(scoring_wikipedia_bindings(dictt, dicttFinal_base, GOLD_DATA))
dict_new = mean_score(dictSortCandidates, dictSortCandidates_base, wn)
merged_dictionary = {**dictt, **dict_new}
print(scoring_wikipedia_bindings(dictt, merged_dictionary, GOLD_DATA))

# create_dataset_for_wikidata()

# to_add, articles = extract_title_with_title_in_RuWordNet(mode='read')




# a = extract_referring_pages(mode='read')

# print(len(create_dict_candidates(a, 'read')))
# candidates = create_dict_candidates()

# model_name='intfloat/multilingual-e5-large'
# type_bindings = 'labse'
# score = bindings(candidates, type_bindings=type_bindings, model_name=model_name, log_len=100, mode='overwrite' )
# mapping = take_mapping(score=score, candidates=candidates, type_bindings=type_bindings, model_name=model_name, mode='overwrite')
# print(scoring(mapping))