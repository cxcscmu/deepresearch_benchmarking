from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import os

client = OpenAI(api_key='...')
MODEL = "gpt-4o-search-preview"
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

def query_gpt(query):

    completion = client.chat.completions.create(
        model=MODEL,
        web_search_options={},
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You are a deepsearch system, providing in-depth reports to user's queries."},
            {"role": "user", "content": query}],
    )

    answer = completion.choices[0].message.content

    sources = []
    for ann in completion.choices[0].message.annotations:
        if ann.type == 'url_citation':
            sources.append(ann.url_citation.url)

    seen = set()
    unique_sources = []
    for url in sources:
        if url not in seen:
            seen.add(url)
            unique_sources.append(url)

    sources_markdown = "\n".join(unique_sources)

    final_answer = f"""{answer}

## References
{sources_markdown}"""

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

    answer = query_gpt(query)


    file_path = os.path.join(out_path, f"{query_id}.a")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(answer)
    
    file_path = os.path.join(out_path, f"{query_id}.q")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(query)

