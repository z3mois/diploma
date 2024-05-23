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
      scoring_wikipedia_bindings, prepare_data_for_new_stage, mean_score, write_pkl,read_pkl
)
from src.mapWikidata import (

    create_dataset_for_wikidata,extract_title_with_title_in_RuWordNet, extract_referring_pages, create_dict_candidates, bindings, take_mapping, scoring, create_graph, find_hyponym
)
from config.const import DAMP_OF_WIKIPEDIA_PATH, DAMP_OF_WIKIDATA_PATH, GOLD_DATA, PATH_TO_TMP_FILE
import pandas as pd
pd.options.mode.chained_assignment = None

# wiki = create_wikisynset()

# dictt, new_wiki = unambiguous_bindings(wn, dictWn, wiki, mode='Ovewrite')

# dictLemm = create_candidates_index_dict(new_wiki, dictWn, mode='overwrite')

# add_multi_flag(new_wiki, dictLemm)
# dictWn = create_info_about_sense()
# dictt, new_wiki = second_stage_bindings()

# dictLemmNew = create_candidates_index_dict(name='lst_candidates_after_snd_stage.pkl', mode='read')


# dict_candidtes = create_candidates_for_multi_stage(new_wiki, wn, dictWn, dictLemmNew, mode='read')

# dict_candidtes_update = delete_double_in_candidates(dict_candidtes)

# dicttFinal, dictSortCandidates = multi_bindings_stage(type_bindings='labse', model_name='setu4993/LaBSE', mode='read')
# dicttFinal_base, dictSortCandidates_base = multi_bindings_stage(type_bindings='base', model_name='setu4993/LaBSE', mode='read')


# # print(scoring_wikipedia_bindings(dictt, dicttFinal, GOLD_DATA))
# print(scoring_wikipedia_bindings(dictt, dicttFinal_base, GOLD_DATA))
# dict_new = mean_score(dictSortCandidates, dictSortCandidates_base, wn)
# merged_dictionary = {**dictt, **dict_new}
anyy=read_pkl(path='base_labse_final_wikipeida.pkl')
values = [obj.to_dict() for obj in anyy.values()]

import json
with open('base_labse_final_wikipeida.json', 'w', encoding='utf-8') as f:
    json.dump(values, f, ensure_ascii=False, indent=4)
# print(values)
# # print(scoring_wikipedia_bindings(dictt, merged_dictionary, GOLD_DATA))





# test wikidata
# create_dataset_for_wikidata()

# to_add, articles = extract_title_with_title_in_RuWordNet(mode='read')
# a = extract_referring_pages(to_add=to_add, articles=articles, mode='read')
# graph_path_straight, graph_path_inverse, id_artircle2idx_article, idx_article2id_artircle = create_graph(a, 'overwrite')
# candidates = create_dict_candidates()
model_name= 'setu4993/LaBSE'#'intfloat/multilingual-e5-large' #
type_bindings = 'labse'
# score = bindings(candidates, type_bindings=type_bindings, model_name=model_name, log_len=100, mode='read' )
mapping = take_mapping(type_bindings=type_bindings, model_name=model_name, mode='read')

values = [obj.to_dict() for obj in mapping.values()]

import json
with open('labse_final_wikidata.json', 'w', encoding='utf-8') as f:
    json.dump(values, f, ensure_ascii=False, indent=4)
# print(mapping)
# print(scoring(mapping))
# unique_ids = set()
# for obj_list in candidates.values():
#     for obj in obj_list:
#         unique_ids.add(obj.page['id'])
# find_hyponym(a,  mapping, id_artircle2idx_article,  unique_ids)