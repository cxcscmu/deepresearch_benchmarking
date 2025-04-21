from datasets import load_dataset
from tqdm import tqdm
import os
import requests
import json

TOKEN="..."
MODEL="sonar-deep-research"

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
    intrinsic = item["intrinsic_scores"]
    multi = intrinsic["multi-faceted"]
    knowledge = intrinsic["knowledge-intensive"]
    reasoning = intrinsic["reasoning-intensive"]
    subjective = intrinsic["subjective"]
    
    nonfactoid = item["nonfactoid_score"]
    
    click_count = len(item["DocStream"])
    
    score = (
        w_multi * multi +
        w_knowledge * knowledge +
        w_reasoning * reasoning +
        w_subjective * subjective +
        w_nonfactoid * nonfactoid +
        w_clicks * click_count
    )

    item["feasibility_score"] = score
    return item

def query_pplx(query):


    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a deepsearch system, provide an in-depth report-like answer to the user query."
            },
            {
                "role": "user",
                "content": query
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print()
    
    all_info = json.loads(response.text)
   
    content = all_info["choices"][0]["message"]["content"]
    urls = all_info["citations"]
    formatted_refs = "\n".join([f"[{i + 1}] - {url}" for i, url in enumerate(urls)])

    final_answer = f"""{content}

## References
{formatted_refs}"""

    return final_answer



dataset = load_dataset("corbyrosset/researchy_questions")

queries = dataset["test"]
queries_with_scores = queries.map(compute_researchy_queries_feasibility)
top_queries = queries_with_scores.sort('feasibility_score', reverse=True).select(range(100))

out_path = f"answers/{MODEL}"
os.makedirs(out_path, exist_ok=True)

print(len(top_queries))
for example in tqdm(top_queries):
    query_id = example["id"]
    query = example["question"]

    answer = query_pplx(query)

    file_path = os.path.join(out_path, f"{query_id}.a")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(answer)
    
    file_path = os.path.join(out_path, f"{query_id}.q")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(query)

