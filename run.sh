
### This is just a log to keep track of models i have already evaluated --joao 


## 1 - hf_deepresearch_gpt-4o-mini
# python eval_quality_async.py --subfolder hf_deepresearch_gpt-4o-mini --open_ai_model gpt-4.1-mini
# python eval_citation_async.py --subfolder hf_deepresearch_gpt-4o-mini --open_ai_model gpt-4.1-mini
# python eval_merge.py --subfolder hf_deepresearch_gpt-4o-mini --open_ai_model gpt-4.1-mini


## 2 - gpt-4o-search-preview
# python eval_quality_async.py --subfolder gpt-4o-search-preview --open_ai_model gpt-4.1-mini
# python eval_citation_async.py --subfolder gpt-4o-search-preview --open_ai_model gpt-4.1-mini
# python eval_merge.py --subfolder gpt-4o-search-preview --open_ai_model gpt-4.1-mini
# cat /data/group_data/cx_group/deepsearch_benchmark/reports/gpt-4o-search-preview/evaluation_results_single_score_gpt-4.1-mini.json 

## 3- GPTResearcher
# python eval_quality_async.py --subfolder GPTResearcher --open_ai_model gpt-4.1-mini
# python eval_citation_async.py --subfolder GPTResearcher --open_ai_model gpt-4.1-mini
# python eval_merge.py --subfolder GPTResearcher --open_ai_model gpt-4.1-mini
# cat /data/group_data/cx_group/deepsearch_benchmark/reports/GPTResearcher/evaluation_results_single_score_gpt-4.1-mini.json 

## 4- open_deep_search
# python eval_quality_async.py --subfolder open_deep_search --open_ai_model gpt-4.1-mini
# python eval_citation_async.py --subfolder open_deep_search --open_ai_model gpt-4.1-mini
# python eval_merge.py --subfolder open_deep_search --open_ai_model gpt-4.1-mini
# cat /data/group_data/cx_group/deepsearch_benchmark/reports/open_deep_search/evaluation_results_single_score_gpt-4.1-mini.json 


## 5- GPTResearcher_custom
# python eval_quality_async.py --subfolder GPTResearcher_custom --open_ai_model gpt-4.1-mini
# python eval_citation_async.py --subfolder GPTResearcher_custom --open_ai_model gpt-4.1-mini
# python eval_merge.py --subfolder GPTResearcher_custom --open_ai_model gpt-4.1-mini
# cat /data/group_data/cx_group/deepsearch_benchmark/reports/GPTResearcher_custom/evaluation_results_single_score_gpt-4.1-mini.json 


# python eval_quality_async.py --subfolder GPTResearcher_custom --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder GPTResearcher --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder open_deep_search_custom --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder open_deep_search --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder hf_deepresearch_gpt-4o-mini_custom --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder hf_deepresearch_gpt-4o-mini --open_ai_model gpt-4.1-mini
# python eval_quality_async.py --subfolder gpt-4o-search-preview --open_ai_model gpt-4.1-mini

#python eval_quality_async.py --subfolder search-r1 --open_ai_model gpt-4.1-mini
#python eval_citation_async.py --subfolder open_deep_search --open_ai_model gpt-4.1-mini