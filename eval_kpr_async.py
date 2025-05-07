import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, List
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv("keys.env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def create_prompt(key_point, answer):

    return f"""You are given a **single key point** and a **report**.

    Your job is to determine whether the report:
    - **Supports** the key point (it affirms, explains, or reinforces the point),
    - **Omits** the key point (it does not mention or cover this point at all), or
    - **Contradicts** the key point (it says something that disagrees with or negates the point).

    Carefully read the key point and the report.

    Return your answer as a **JSON object** with two fields:
    - "label": One of "Supported", "Omitted", or "Contradicted".
    - "justification": Brief explanation on why you assigned this label.

    Respond strictly in JSON format:
    {{"label": label, "justification": justification}}
    Do **not** add any extra commentary or text outside the JSON.

    ---

    Key Point: {key_point}
    Report: {answer}
    """



class KeyPointRecall(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str


async def evaluate_single_key_point(semaphore, key_point, answer, model):
    async with semaphore:
        prompt = create_prompt(key_point["point_content"], answer)
        chat_pattern = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=chat_pattern,
                    response_format=KeyPointRecall,
                    temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return key_point["point_number"], (result['label'], result['justification'])

async def evaluate_answer(semaphore, answer, key_points, model):
    tasks = [evaluate_single_key_point(semaphore, key_point, answer, model) for key_point in key_points]
    results = await asyncio.gather(*tasks)
    return dict(results)

async def evaluate_query(semaphore, query_id, anwser_dir, key_point_dir, model):
    a_path = anwser_dir / f"{query_id}.a"
    p_path = key_point_dir / f"{query_id}_aggregated.json"

    if not a_path.exists():
        print(f"Warning: Missing answer file for query {query_id}")
        return query_id, None
    if not p_path.exists():
        print(f"Warning: Missing key point file for query {query_id}")
        return query_id, None

    answer = a_path.read_text().strip()
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        key_points = data["key_points"]

    try:
        evaluations = await evaluate_answer(semaphore, answer, key_points, model)
        return query_id, {
            "labels": evaluations,
        }
    except Exception as e:
        print(f"Error evaluating {query_id}: {e}")
        return query_id, None

async def evaluate_folder_async(subfolder_name, model, path_to_reports, key_point_dir):
    folder_path = Path(path_to_reports) / subfolder_name
    output_file = folder_path / f"evaluation_results_kpr_{model}.json"
    key_point_dir = Path(key_point_dir)

    all_results = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    print(f"Skipped queries: {len(all_results)}")

    semaphore = asyncio.Semaphore(100)

    query_ids = [p.stem for p in folder_path.glob("*.q") if p.stem not in all_results]
    tasks = [evaluate_query(semaphore, qid, folder_path, key_point_dir, model) for qid in query_ids]
    results = await tqdm_asyncio.gather(*tasks)

    for query_id, result in results:
        if result is not None:
            all_results[query_id] = result

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder")
    parser.add_argument("--open_ai_model")
    args = parser.parse_args()

    path_to_reports = "reports"
    path_to_results = "results"
    path_to_key_point = "key_point"

    print(f"Evaluating {args.subfolder} using {args.open_ai_model}")
    results = asyncio.run(evaluate_folder_async(args.subfolder, args.open_ai_model, path_to_reports, path_to_key_point))

    for query_id, labels in results.items():
        supported_count = 0
        omitted_count = 0
        contradicted_count = 0
        for point_number, label in labels["labels"].items():
            if label[0] == "Supported":
                supported_count += 1
            elif label[0] == "Omitted":
                omitted_count += 1
            elif label[0] == "Contradicted":
                contradicted_count += 1
        
        support_rate = supported_count / len(labels["labels"]) * 100
        ommitted_rate = omitted_count / len(labels["labels"]) * 100
        contradicted_rate = contradicted_count / len(labels["labels"]) * 100
        results[query_id]["support_rate"] = support_rate
        results[query_id]["ommitted_rate"] = ommitted_rate
        results[query_id]["contradicted_rate"] = contradicted_rate
        
    
    result_dir = Path(path_to_results) / args.subfolder
    result_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_path = result_dir / f"evaluation_results_kpr_{args.open_ai_model}.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed evaluation results to {detailed_path}")