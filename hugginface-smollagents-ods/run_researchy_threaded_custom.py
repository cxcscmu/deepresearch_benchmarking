import argparse
import os
import threading
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.text_inspector_tool import TextInspectorTool
from scripts.simple_search_tool import (
    StaticSearchTool,
    StaticTextViewer,
    SelectDocTool,
    ViewportTool,
    PageDownTool,
    PageUpTool,
    FindInDocTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)
import json


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--concurrency", type=int, default=16)
    return parser.parse_args()


def dummy_search_function(query, top_k):
    import requests
    import base64
    import json

    try:
        URL = "https://clueweb22.us"
        request_url = f"{URL}/search?query={query}&k={top_k}"
        response = requests.get(request_url, timeout=30)
        response.raise_for_status()

        json_data = response.json()
        results = json_data.get("results", [])
        parsed_docs = []

        for encoded_doc in results:
            decoded = base64.b64decode(encoded_doc).decode("utf-8")
            parsed = json.loads(decoded)

            text = parsed.get("Clean-Text", "")
            url = parsed.get("URL", "")
            doc_id = parsed.get("ClueWeb22-ID", "")

            parsed_docs.append({
                "title": f"{doc_id} ({url})",
                "text": text.strip()
            })

        return parsed_docs
    except Exception as e:
        print(f"Search API error: {e}")
        return []


def create_agent(model_id="o1"):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
        "max_completion_tokens": 8192,
    }
    if model_id in ["o1", "o3-mini"]:
        model_params["reasoning_effort"] = "high"
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    viewer = StaticTextViewer([])
    WEB_TOOLS = [
        StaticSearchTool(
        dummy_search_function,
        viewer),
        SelectDocTool(viewer),
        ViewportTool(viewer),
        PageUpTool(viewer),
        PageDownTool(viewer),
        FindInDocTool(viewer),
        TextInspectorTool(model, text_limit),
    ]
    search_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search a static document collection to answer your question.
Ask them for all your questions that require reading and synthesis.
Provide as much context as possible, including timeframes or focus topics.
Use real sentences as input, not just keyword queries.
""",
        provide_run_summary=True,
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[search_agent],
    )

    return manager_agent


def process_query(example, model_id, output_dir):
    agent = create_agent(model_id=model_id)
    query_id = example["id"]
    query = example["query"]

    augmented_question = "You'll act as a deepsearch system, generating in-depth, structured reports in response to user queries. Your goal is to synthesize information and provide well-organized answers with supported claims. Here is the question: " + query

    try:
        answer = agent.run(augmented_question)

        file_path_a = os.path.join(output_dir, f"{query_id}.a")
        file_path_q = os.path.join(output_dir, f"{query_id}.q")

        with append_answer_lock, open(file_path_a, "w", encoding="utf-8") as f:
            f.write(str(answer))
        with append_answer_lock, open(file_path_q, "w", encoding="utf-8") as f:
            f.write(str(query))
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")


def main():
    args = parse_args()
    path_to_queries = "../queries/researchy_queries_sample_doc_click.jsonl"
    with open(path_to_queries, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    print(f"Total of {len(queries)} queries.")

    output_dir = f"RQ_results/hf_deepresearch_custom_{args.model_id}"
    os.makedirs(output_dir, exist_ok=True)

    filtered_queries = []
    for example in queries:
        query_id = example["id"]
        file_path_a = os.path.join(output_dir, f"{query_id}.a")
        file_path_q = os.path.join(output_dir, f"{query_id}.q")
        if not (os.path.exists(file_path_a) and os.path.exists(file_path_q)):
            filtered_queries.append(example)

    print(f"Processing {len(filtered_queries)} queries (skipping already processed).")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(process_query, example, args.model_id, output_dir)
            for example in filtered_queries
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            f.result()


if __name__ == "__main__":
    main()