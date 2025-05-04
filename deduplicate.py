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
class KeyPointDeduplicationSingle(BaseModel):
    point_number: int
    point_content: str
    original_point_number: List[int]

# class for all key point extraction results
class KeyPointDeduplication(BaseModel):
    points: List[KeyPointDeduplicationSingle]

def create_prompt(original_points: List[str]):

    original_points_with_number = {i + 1: point for i, point in enumerate(original_points)}

    return f"""Based on the points extracted from a piece of text, identify and remove any duplicate points to streamline the list.

REMEMBER:
- The de-duplicated points need to contain all the original points.
- An original point cannot exist in two different de-duplicated points at the same time.
- Do not add any new points, only de-duplicate the existing ones. And do not add any explanations.
- For each point, provide the corresponding original point numbers to indicate which original points are included in the de-duplicated point.

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


    
    
async def deduplicate_single_query(semaphore, key_point_dir, qid, model):

    key_point_dir = Path(key_point_dir)
    key_point_path = key_point_dir / f"{qid}.json"
    deduplicated_key_point_path = key_point_dir / f"{qid}_deduplicated.json"


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

    # deduplicate key points
    prompt = create_prompt(all_points)
    chat_pattern = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    async with semaphore:
        response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=chat_pattern,
                    response_format=KeyPointDeduplication,
                    temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        deduplicated_points = result['points']

        results = {
            "question": question,
            "key_points": deduplicated_points
        }

        with open(deduplicated_key_point_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    


async def deduplicate_all_queries(key_point_dir, model):

    semaphore = asyncio.Semaphore(100)

    query_ids = [
        p.stem 
        for p in Path(key_point_dir).glob("*.json") 
        if not p.name.endswith("_deduplicated.json")
    ]
    tasks = [deduplicate_single_query(semaphore, key_point_dir, qid, model) for qid in query_ids]
    
    await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    
    key_point_dir = "key_point"
    model = "gpt-4.1-mini"

    asyncio.run(deduplicate_all_queries(key_point_dir, model))
