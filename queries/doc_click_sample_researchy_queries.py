# This is how the queries were sampled. comment to avoid reruning and rewriting the file.

import random
import json
from datasets import load_dataset

dataset = load_dataset("corbyrosset/researchy_questions")
queries = dataset["test"]

def compute_researchy_queries_feasibility(item):
    item["feasibility_score"] = len(item["DocStream"])
    return item

queries_with_scores = queries.map(compute_researchy_queries_feasibility)
top_queries = queries_with_scores.sort('feasibility_score', reverse=True).select(range(1000))

for i, entry in enumerate(top_queries):
    if i == 2 or i == 5:
        continue
    print(entry["question"])
    for k in entry["DocStream"][:10]:
        print(k["Url"])
    print("========================")


    if i == 7: 
        break
    


# output_path = "queries/researchy_queries_sample_doc_click.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for entry in top_queries:
#         json.dump({"id": entry["id"], "query": entry["question"]}, f)
#         f.write("\n")