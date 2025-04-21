import asyncio
from pydantic import BaseModel,HttpUrl
from openai import OpenAI
from pydantic import create_model
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
import re
from enum import Enum
from typing import List

from dotenv import load_dotenv
load_dotenv("keys.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

crawl_config = CrawlerRunConfig(
    stream=False,
    verbose=False
)
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
        markdown_outputs = []
        for result in results:
            if result.success:
                markdown_outputs.append(result.markdown)
            else:
                markdown_outputs.append(f"**Error for {result.url}:** {result.error_message}")

    return markdown_outputs

def create_prompt_extractor(answer):
    return f"""You are an information extraction expert.

Given a structured report containing claims and their supporting sources (usually in the form of inline hyperlinks or referenced URLs), extract all distinct factual or argumentative claims that are explicitly supported by a specific reference in the text.

Return a JSON object like this:
{{
  "claims": [
    {{
      "id": 1,
      "claim": "<claim_1>",
      "sources": [<url_1>,...,<url_n>]
    }},
    {{
      "id": 2,
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
    - sources, the URLs of the sources that explicitly supports it (must be taken directly from the report, can be one or more).

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


def extract_claims_and_url(answer, model):
    prompt = create_prompt_extractor(answer)
    chat_pattern = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    response = client.beta.chat.completions.parse(
        model=model,
        messages=chat_pattern,
        response_format=ClaimsModel,
        temperature=0
    )

    try:
        json_response = json.loads(response.choices[0].message.content)["claims"]
    except Exception:
        print("Could not parse JSON")

    return json_response


def check_citation_quality(claim, docs, model):
    prompt = create_prompt_citation_checker(claim, docs)
    chat_pattern = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    response = client.beta.chat.completions.parse(
        model=model,
        messages=chat_pattern,
        response_format=CitationSupport,
        temperature=0
    )

    try:
        json_response = json.loads(response.choices[0].message.content)
    except Exception:
        print("Could not parse JSON")

    return json_response


def evaluate_folder(subfolder_name, model):
    folder_path = Path("answers") / subfolder_name
    assert folder_path.exists(), f"Folder {folder_path} does not exist"

    output_file = Path("answers") / args.subfolder / f"evaluation_results_citation_{model}.json" 

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    print(f"Skipped queries: {len(all_results)}")

    query_files = list(folder_path.glob("*.q"))[:1]
    for file in tqdm(query_files, total=len(query_files)):
        query_id = file.stem
        q_path = folder_path / f"{query_id}.q"
        a_path = folder_path / f"{query_id}.a"

        if query_id in all_results:
            continue

        if not a_path.exists():
            print(f"Warning: Missing answer file for query {query_id}")
            continue

        with open(a_path, "r", encoding="utf-8") as f:
            answer = f.read().strip()

        try:
            claims_to_urls = extract_claims_and_url(answer, model)

            scores = {}
            num_claims = len(claims_to_urls)

            if num_claims == 0:
                final_score = 0

                all_results[query_id] = {
                    "score": final_score,
                    "detailed": None
                }
            
            else:
                for claim in claims_to_urls:
                    claim_text = claim["claim"]
                    urls = claim["sources"]

                    docs = asyncio.run(crawl_urls(urls))
                    url_pattern = r'https?://\S+|www\.\S+'
                    clean_docs = [re.sub(url_pattern, '', text) for text in docs]

                    if not clean_docs:
                        continue

                    try:
                        res = check_citation_quality(claim_text, clean_docs, model)
                        label = res["support"]
                        justification = res["justification"]
                        score = CitationSupportValues.score(label)
                        scores[f"claim_{claim['claim_id']}"] = {
                            "claim": claim_text,
                            "urls": urls,
                            "score": score, 
                            "justification": justification,
                        }

                    except Exception as e:
                        print(f"Error evaluating claim {claim['claim_id']}: {e}")

                final_score = sum([x["score"] for x in list(scores.values())]) / len(scores) if scores else 0.0

                all_results[query_id] = {
                    "score": final_score,
                    "detailed": scores
                }


        except Exception as e:
            import traceback
            print(f"Error evaluating {query_id}")
            traceback.print_exc()


    average_normalized_score = (
        sum(r["score"] for r in all_results.values()) / len(all_results)
        if all_results else 0
    )

    return all_results, average_normalized_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", help="Name of the subfolder inside 'answers/'")
    parser.add_argument("--open_ai_model", help="openai model name'")
    args = parser.parse_args()

    print(f"Evaluating {args.subfolder} using {args.open_ai_model}")

    results , avg = evaluate_folder(args.subfolder, args.open_ai_model)

    output_file = Path("answers") / args.subfolder / f"evaluation_results_citation_{args.open_ai_model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed evaluation results to {output_file}")