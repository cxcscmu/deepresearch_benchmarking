from tqdm import tqdm
import os
import requests
import json
from dotenv import load_dotenv

MODEL="sonar-deep-research"

load_dotenv("keys.env")
TOKEN = os.getenv("PERPLEXITY_API_KEY")

QUERIES = "queries/researchy_queries_sample_doc_click.jsonl"
OUT_PATH = f"/data/group_data/cx_group/deepsearch_benchmark/reports/{MODEL}"
os.makedirs(OUT_PATH, exist_ok=True)

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

    
    all_info = json.loads(response.text)
   
    content = all_info["choices"][0]["message"]["content"]
    urls = all_info["citations"]
    formatted_refs = "\n".join([f"[{i + 1}] - {url}" for i, url in enumerate(urls)])

    final_answer = f"""{content}

## References
{formatted_refs}"""

    usage = all_info["usage"]

    return final_answer, usage


with open(QUERIES, "r", encoding="utf-8") as f:
    all_queries = [json.loads(line) for line in f]

# Filter out completed queries
pending_queries = []
for example in all_queries:
    query_id = str(example["id"])
    a_path = os.path.join(OUT_PATH, f"{query_id}.a")
    if not os.path.exists(a_path):
        pending_queries.append(example)

print(f"Total pending queries: {len(pending_queries)}")

pending_queries = pending_queries[:200]

for example in tqdm(pending_queries):
    query_id = example["id"]
    query = example["query"]

    answer, usage = query_pplx(query)

    file_path = os.path.join(OUT_PATH, f"{query_id}.a")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(answer)
    
    file_path = os.path.join(OUT_PATH, f"{query_id}.q")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(query)

    with open(os.path.join(OUT_PATH, f"{query_id}.u"), "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

