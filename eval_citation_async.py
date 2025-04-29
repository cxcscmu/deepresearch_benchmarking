import asyncio
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from pydantic import create_model
import json
import os
from pathlib import Path
import argparse
from tqdm.asyncio import tqdm_asyncio
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
import re
from enum import Enum
from typing import List
from dotenv import load_dotenv
import logging

load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

crawl_config = CrawlerRunConfig(stream=False, verbose=False)
browser_config = BrowserConfig(verbose=False)

class CitationSupportValues(str, Enum):
    FULL = "full_support"
    PARTIAL = "partial_support"
    NONE = "no_support"

    @classmethod
    def score(cls, value):
        return {
            cls.NONE.value: 0.0,
            cls.PARTIAL.value: 0.5,
            cls.FULL.value: 1.0,
        }[value]

class CitationSupport(BaseModel):
    support: CitationSupportValues
    justification: str


class ClaimEntry(BaseModel):
    claim_id: int
    claim: str
    sources: List[str]

class ClaimsModel(BaseModel):
    claims: List[ClaimEntry]



async def crawl_urls(urls):
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=crawl_config)
        return [r.markdown if r.success else f"**Error for {r.url}:** {r.error_message}" for r in results]

def create_prompt_extractor(answer):
    return f"""You are an information extraction expert.

Given a structured report containing claims and their supporting sources (usually in the form of inline hyperlinks or referenced URLs), extract all distinct factual or argumentative claims that are explicitly supported by a specific reference in the text.

Return a JSON object like this:
{{
  "claims": [
    {{
      "claim_id": 1,
      "claim": "<claim_1>",
      "sources": [<url_1>,...,<url_n>]
    }},
    {{
      "claim_id": 2,
      "claim": "<claim_2>",
      "sources": [<url_1>,...,<url_n>]
    }},
    ...
  ]
}}

Where:

- The root is "claims", which contains a list of json claim objects.
- Each claim json object has: 
    - id, an identifier (sequential integer starting from 1).
    - claim, a concise but complete sentence restating the claim.
    - sources, a list of URLs, which are the sources that explicitly support the claim (**IMPORTANT**: must be taken directly from the report, can be one or more).

**IMPORTANT**: Only include claims that are directly and explicitly supported by a source in the report. Do not include general summaries, opinions, or claims that lack citation.

Process the full report carefully to ensure all source-supported claims are included and accurately captured.

Now extract the claims from the report below:

{answer}

Return the JSON object, nothing else.
"""

def create_prompt_citation_checker(claim, docs):
    # Adapted from: https://github.com/vectara/open-rag-eval/blob/dev/open_rag_eval/metrics/citation_metric.py
    citations_text = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))
    return f"""In this task, you will evaluate whether each statement is
        supported by its corresponding citations. Note that the system
        responses may appear very fluent and well-formed, but contain
        slight inaccuracies that are not easy to discern at first glance.
        Pay close attention to the text.

        You will be provided with a statement and its corresponding
        citations. It may be helpful to ask yourself whether it is
        accurate to say "according to the citation" with a
        statement following this phrase. Be sure to check all of the
        information in the statement. You will be given three options:

        - Full Support: All of the information in the statement is
        supported in the citations.

        - Partial Support: Some parts of the information are supported in
        the citations, but other parts are missing from the citations.

        - No Support: These citations does not support any part of the
        statement.

        Please provide your response based on the information in the
        citations. If you are unsure, use your best judgment. Respond as
        either ``full_support'', ``partial_support'', or ``no_support''
        with no additional information. 
        You should also provide a very brief justification to your assessment.

        Statement: {claim}

        Citations: {citations_text}
    """

async def extract_claims_and_url(openai_semaphore, answer, model):
    async with openai_semaphore:
        prompt = create_prompt_extractor(answer)
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
            response_format=ClaimsModel,
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)["claims"]
        except Exception:
            print("Could not parse JSON - extractor")
            return {}

async def check_citation_quality(openai_semaphore, claim, docs, model):
    async with openai_semaphore:
        prompt = create_prompt_citation_checker(claim, docs)
        response = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
            response_format=CitationSupport,
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            print("Could not parse JSON - quality")
            #print(response.choices[0].message.content)
            return {}

async def evaluate_query(openai_semaphore, query_id, answer_path, model):
    with open(answer_path, "r", encoding="utf-8") as f:
        answer = f.read().strip()

    claims_to_urls = await extract_claims_and_url(openai_semaphore, answer, model)

    if not claims_to_urls:
        return query_id, {"score": 0, "detailed": None}

    scores = {}
    for claim in claims_to_urls:
        claim_text = claim["claim"]
        urls = claim["sources"]
        claim_id = claim["claim_id"]
        if not urls:
            # sourceless citation, should not happen but capture still.
            continue

        docs = await crawl_urls(urls)
        if not docs:
            # This will skip a claim because crawler did not return text for that url.
            continue

        url_pattern = r'https?://\S+|www\.\S+'
        clean_docs = [re.sub(url_pattern, '', d) for d in docs if d.strip()]

        try:
            res = await check_citation_quality(openai_semaphore, claim_text, clean_docs, model)
            label = res["support"]
            justification = res["justification"]
            if label:
                score = CitationSupportValues.score(label)
                scores[f"claim_{claim_id}"] = {
                    "claim": claim_text,
                    "urls": urls,
                    "score": score,
                    "justification": justification,
                }
        except Exception as e:
            scores[f"claim_{claim_id}"] = {
                    "claim": claim_text,
                    "urls": urls,
                    "score": 0,
                    "justification": str(e),
                }

    final_score = sum(s["score"] for s in scores.values()) / len(scores) if scores else 0.0
    return query_id, {"score": final_score, "detailed": scores or None}

async def evaluate_folder_async(subfolder_name, model, path_to_reports):
    folder_path = Path(path_to_reports) / subfolder_name
    output_file = folder_path / f"evaluation_results_citation_{model}.json"

    all_results = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    print(f"Skipped {len(all_results)} queries.")

    openai_semaphore = asyncio.Semaphore(12)

    query_files = list(folder_path.glob("*.a"))
    tasks = []
    for file in query_files:
        query_id = file.stem
        if query_id in all_results:
            continue
        tasks.append(evaluate_query(openai_semaphore, query_id, file, model))


    results = await tqdm_asyncio.gather(*tasks)
    for query_id, result in results:
        all_results[query_id] = result

    average_score = sum(r["score"] for r in all_results.values()) / len(all_results) if all_results else 0
    return all_results, average_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder")
    parser.add_argument("--open_ai_model")
    args = parser.parse_args()

    path_to_reports = "/data/group_data/cx_group/deepsearch_benchmark/reports/"
    print(f"Evaluating {args.subfolder} using {args.open_ai_model}")
    results, avg = asyncio.run(evaluate_folder_async(args.subfolder, args.open_ai_model, path_to_reports))

    print(f"Evaluated {len(results)} queries.")

    output_path = Path(path_to_reports) / args.subfolder / f"evaluation_results_citation_{args.open_ai_model}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed evaluation results to {output_path}")
