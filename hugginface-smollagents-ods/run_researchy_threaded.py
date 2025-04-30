import argparse
import os
import threading
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # HfApiModel,
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


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1"):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if model_id in ["o1", "o3-mini"]:
        model_params["reasoning_effort"] = "high"
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent

def process_query(example, model_id, output_dir):
    agent = create_agent(model_id=model_id)  # Moved inside the thread
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
