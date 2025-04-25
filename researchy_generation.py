# This is the code I used to generate the results for ODS. It should be placed under this folder: OpenDeepSearch/src/opendeepsearch/
#!/usr/bin/env python3
import asyncio
import json
import os
from pathlib import Path
from opendeepsearch import OpenDeepSearchAgent


SYSTEM_NAME = "ODS"                     # folder where .q/.a files go
OUTPUT_DIR = Path(SYSTEM_NAME)
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_SYSTEM_PROMPT = """
You are a research analyst. For each query, generate a markdown report with:
1. Executive Summary
2. Methodology (search strategies and tools used)
3. Detailed Findings (include ≥3 direct quotes with citations like [^1], [^2])
4. Sources (numbered list of URLs & titles)
5. Limitations
"""

agent = OpenDeepSearchAgent(
    model="gpt-4o",
    reranker="jina",
    search_provider="serper",
    temperature=0.2,
    top_p=0.3,
    source_processor_config={
        "top_results": 20,
        "filter_content": False,
    },
)
agent.system_prompt = REPORT_SYSTEM_PROMPT

# ────────────────────────────────────────────────────────────────
async def process_record(rec: dict, sem: asyncio.Semaphore):
    qid, qtext = rec["id"], rec["query"]

    # write the raw query
    (OUTPUT_DIR / f"{qid}.q").write_text(qtext, encoding="utf-8")

    # generate the report
    async with sem:
        report = await agent.ask(qtext, max_sources=20, pro_mode=True)

    # write the answer/report
    (OUTPUT_DIR / f"{qid}.a").write_text(report, encoding="utf-8")

# ────────────────────────────────────────────────────────────────
async def main(jsonl_path: str, concurrency: int = 4):
    # load all {id,query} records
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(process_record(r, sem) for r in records))

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deep-search report generator")
    parser.add_argument("-i","--input", required=True,
                        help="Path to JSONL file with {id,query} per line")
    parser.add_argument("-c","--concurrency", type=int, default=4,
                        help="Number of parallel requests")
    args = parser.parse_args()

    asyncio.run(main(args.input, args.concurrency))
