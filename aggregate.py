import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, List
from tqdm.asyncio import tqdm_asyncio
from pydantic import create_model
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# class for a single point
class KeyPointAggregationSingle(BaseModel):
    point_number: int
    point_content: str
    original_point_number: List[int]

# class for all key point extraction results
class KeyPointAggregation(BaseModel):
    points: List[KeyPointAggregationSingle]

def create_prompt(original_points: List[str]):

    original_points_with_number = {i + 1: point for i, point in enumerate(original_points)}

    return f"""
You are given a list of points extracted from a piece of text. Your task is to aggregate these points according to the following instructions:

1. Identify and deduplicate any duplicated or highly similar points. Merge them into a single, representative point.
2. Identify contradictory points. Merge them into a single point that presents both sides. For example, if one point says "Covid vaccine is safe" and another says "Covid vaccine is not safe", merge them into: 
   "A claimed the Covid vaccine is safe, and B claimed the Covid vaccine is not safe."

IMPORTANT RULES:
- Every aggregated point must preserve all original information from the included points.
- Do not invent or add any new information. Only use what is already present.
- Do not provide any explanations or summaries beyond the aggregation itself.
- For each aggregated point, include a reference to the original point numbers it is based on, e.g., "original_point_number": [1, 3, 7]".
- Do not make a aggregated key point too lengthy. Do not aggregate too many points into a single one. Each aggregated point should be to the point.


Respond strictly in JSON format:
{{
    "points": [
        {{
            "point_number": point_number,
            "point_content": point_content,
            "original_point_number": [original_point_number1, original_point_number2, ...]
        }},
        ...
    ]
}}

[Original Points]
{original_points_with_number}
"""


    
    
async def aggregate_single_query(semaphore, key_point_dir, qid, model):

    key_point_dir = Path(key_point_dir)
    key_point_path = key_point_dir / f"{qid}.json"
    aggregated_key_point_path = key_point_dir / f"{qid}_aggregated.json"


    with open(key_point_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        key_points = data["key_points"]
        question = data["question"]
        if key_points is None:
            print(f"Key points are None for {key_point_path}")


    all_points = []

    for cluewebID, points in key_points.items():
        for point in points:
            all_points.append(point["point_content"])

    # aggregate key points
    prompt = create_prompt(all_points)
    chat_pattern = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    async with semaphore:
        response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=chat_pattern,
                    response_format=KeyPointAggregation,
                    temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        aggregated_points = result['points']

        results = {
            "question": question,
            "key_points": aggregated_points
        }

        with open(aggregated_key_point_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    


async def aggregate_all_queries(key_point_dir, model):

    semaphore = asyncio.Semaphore(100)

    query_ids = [
        p.stem 
        for p in Path(key_point_dir).glob("*.json") 
        if not p.name.endswith("_aggregated.json") and not p.name.endswith("_deduplicated.json")
    ]
    tasks = [aggregate_single_query(semaphore, key_point_dir, qid, model) for qid in query_ids]
    
    await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    
    key_point_dir = "key_point"
    model = "gpt-4.1-mini"

    asyncio.run(aggregate_all_queries(key_point_dir, model))
