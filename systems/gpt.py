import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os
import json
from dotenv import load_dotenv


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-search-preview"
#QUERIES = "queries/researchy_queries_sample_doc_click.jsonl"
QUERIES = "queries/subset.jsonl"
OUT_PATH = f"/data/group_data/cx_group/deepsearch_benchmark/reports/{MODEL}"
os.makedirs(OUT_PATH, exist_ok=True)

async def query_gpt(query):
    completion = await client.chat.completions.create(
        model=MODEL,
        web_search_options={},
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You act as a deepsearch system, generating in-depth, structured reports in response to user queries. Your goal is to synthesize information from the web and provide well-organized answers. Every factual claim should be supported by a citation, and you must perform web searches when needed."},
            {"role": "user", "content": query}
        ],
    )

    answer = completion.choices[0].message.content

    sources = []
    for ann in completion.choices[0].message.annotations or []:
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


    usage = {
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
    }

    return final_answer, usage

async def main():
    with open(QUERIES, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} queries")

    async def process_line(line):
        example = json.loads(line)
        query_id = example["id"]
        query = example["query"]
        answer, usage = await query_gpt(query)

        with open(os.path.join(OUT_PATH, f"{query_id}.a"), "w", encoding="utf-8") as fa:
            fa.write(answer)
        with open(os.path.join(OUT_PATH, f"{query_id}.q"), "w", encoding="utf-8") as fq:
            fq.write(query)
        with open(os.path.join(OUT_PATH, f"{query_id}.u"), "w", encoding="utf-8") as fu:
            json.dump(usage, fu, indent=2)

    await tqdm_asyncio.gather(*(process_line(line) for line in lines))

if __name__ == "__main__":
    asyncio.run(main())