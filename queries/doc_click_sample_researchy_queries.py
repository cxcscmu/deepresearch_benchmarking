## This is how the queries were sampled. comment to avoid reruning and rewriting the file.

import random
import json
from datasets import load_dataset

dataset = load_dataset("corbyrosset/researchy_questions")
queries = dataset["test"]

def compute_researchy_queries_feasibility(item, 
                        w_multi=1.0, 
                        w_knowledge=1.0, 
                        w_reasoning=1.0, 
                        w_subjective=0.5, 
                        w_nonfactoid=1.0, 
                        w_clicks=0.5):
    """
    Compute a composite score for a query based on intrinsic scores, nonfactoid score, and click behavior.
    
    Parameters:
      item (dict): A dictionary representing a query from the dataset.
      w_multi (float): Weight for the 'multi-faceted' score.
      w_knowledge (float): Weight for the 'knowledge-intensive' score.
      w_reasoning (float): Weight for the 'reasoning-intensive' score.
      w_subjective (float): Weight for the 'subjective' score.
      w_nonfactoid (float): Weight for the nonfactoid score.
      w_clicks (float): Weight for the number of clicked documents (DocStream length).

    Returns:
      float: The composite score.
    """
    # intrinsic = item["intrinsic_scores"]
    # multi = intrinsic["multi-faceted"]
    # knowledge = intrinsic["knowledge-intensive"]
    # reasoning = intrinsic["reasoning-intensive"]
    # subjective = intrinsic["subjective"]
    
    # nonfactoid = item["nonfactoid_score"]
    
    click_count = len(item["DocStream"])
    
    # score = (
    #     w_multi * multi +
    #     w_knowledge * knowledge +
    #     w_reasoning * reasoning +
    #     w_subjective * subjective +
    #     w_nonfactoid * nonfactoid +
    #     w_clicks * click_count
    # )

    score = click_count

    item["feasibility_score"] = score
    return item


queries_with_scores = queries.map(compute_researchy_queries_feasibility)
top_queries = queries_with_scores.sort('feasibility_score', reverse=True).select(range(1000))

output_path = "queries/researchy_queries_sample_doc_click.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in top_queries:
        json.dump({"id": entry["id"], "query": entry["question"]}, f)
        f.write("\n")