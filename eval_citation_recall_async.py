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
import re
from enum import Enum
from typing import List
from dotenv import load_dotenv
import logging

load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ClaimEntry(BaseModel):
    claim_id: int
    claim: str
    sources: List[str]

class ClaimsModel(BaseModel):
    claims: List[ClaimEntry]


def create_prompt_extractor(answer):
    return f"""You are an information extraction expert.

Given a structured report containing claims and their supporting sources (usually in the form of inline hyperlinks or referenced URLs), extract all distinct factual or argumentative claims in the text.
If a claim is supported by one or more sources, return the supporting URLs as sources.
If a claim is not supported by any source, return an empty list of sources.

Return a JSON object like this:
{{
  "claims": [
    {{
      "claim_id": 1,
      "claim": "<claim_1>",
      "sources": ["<url_1>", "<url_2>", ...]
    }},
    {{
      "claim_id": 2,
      "claim": "<claim_2>",
      "sources": []
    }},
    ...
  ]
}}

Where:

- The root is "claims", which contains a list of claim objects.
- Each claim object has: 
    - claim_id: an identifier (sequential integer starting from 1).
    - claim: a concise but complete sentence restating the claim.
    - sources: a list of URLs that explicitly support the claim, or an empty list if no URLs support it. 

(**IMPORTANT**: Only include URLs that are **explicitly present in the report text**, typically as inline hyperlinks or reference-style citations. Do not infer or fabricate URLs. Do not include non-URL citations such as book titles, paper references, or other non-URL sources.)

(**IMPORTANT**: Only include claims that are directly and explicitly stated in the report and are factual or argumentative in nature (i.e., statements that can be verified or refuted). Do not include general summaries, personal opinions, or meta-commentary.)

Process the full report carefully to ensure all claims are included and accurately captured.

Now extract the claims from the report below:

{answer}

Return the JSON object, and nothing else.
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

async def evaluate_query(openai_semaphore, query_id, answer_path, model):
    with open(answer_path, "r", encoding="utf-8") as f:
        answer = f.read().strip()

    url_pattern = r'https?://\S+|www\.\S+'
    if not re.search(url_pattern, answer):
        return query_id, {"score": 0, "detailed": "No URLs found in text."}

    claims_to_urls = await extract_claims_and_url(openai_semaphore, answer, model)

    if not claims_to_urls:
        return query_id, {"score": 0.0, "detailed": "No claims extracted."}

    total_claims = len(claims_to_urls)
    supported_claims = sum(1 for claim in claims_to_urls if claim.get("sources"))

    score = supported_claims / total_claims if total_claims > 0 else 0.0

    detailed = {
        f"claim_{claim['claim_id']}": {
            "claim": claim["claim"],
            "urls": claim["sources"],
            "supported": bool(claim["sources"])
        }
        for claim in claims_to_urls
    }

    return query_id, {"score": score, "detailed": detailed}

async def evaluate_folder_async(subfolder_name, model, path_to_reports):
    folder_path = Path(path_to_reports) / subfolder_name
    output_file = folder_path / f"evaluation_results_citation_recall_{model}.json"

    all_results = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    print(f"Skipped {len(all_results)} queries.")

    openai_semaphore = asyncio.Semaphore(100)

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

    output_path = Path(path_to_reports) / args.subfolder / f"evaluation_results_citation_recall_{args.open_ai_model}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_score = 0
    count = 0

    for query_id, result in results.items():
        query_score = result.get("score", 0)
        total_score += query_score * 100 
        count += 1

    average_score = total_score / count if count > 0 else 0
    print(f"\nAverage normalized citation recall score across {count} queries: {average_score:.2f}")

    print(f"\nSaved detailed evaluation results to {output_path}")
