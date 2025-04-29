#!/usr/bin/env python3
import os
import sys
import asyncio
import json
from pathlib import Path


os.environ["RETRIEVER"]          = "custom"
os.environ["RETRIEVER_ENDPOINT"] = "https://clueweb22.us/search"
os.environ["RETRIEVER_ARG_K"]     = "3"

from gpt_researcher import GPTResearcher



# ────────────────────────────────────────────────────────────────
# 1) Paths & Config
# ────────────────────────────────────────────────────────────────
SYSTEM_NAME = "GPTResearcher_custom"                        # output folder
OUTPUT_DIR = Path(SYSTEM_NAME)
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG_PATH = "gptr_config.json"               
REPORT_TYPE = "research_report"

# ────────────────────────────────────────────────────────────────
# 2) Process a single record
# ────────────────────────────────────────────────────────────────
async def process_record(rec: dict, sem: asyncio.Semaphore):
    qid, qtext = rec["id"], rec["query"]

    # Write the raw query
    (OUTPUT_DIR / f"{qid}.q").write_text(qtext, encoding="utf-8")

    async with sem:
        try:
            # Initialize researcher with config (no hard-coded keys)
            researcher = GPTResearcher(
                query=qtext,
                report_type=REPORT_TYPE,
                config_path=CONFIG_PATH
            )

            # Conduct the deep research
            await researcher.conduct_research()

            # If Tavily has expired or failed, its retriever returns []
            sources = researcher.get_research_sources()
            if not sources:
                raise RuntimeError(
                    "No sources returned. Tavily API may have expired or rate-limited you."
                )

            # Generate the markdown report
            report_md = await researcher.write_report()

        except Exception as e:
            # Catch both tavily HTTP errors (502, 429) and empty results
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    # Write the answer/report
    (OUTPUT_DIR / f"{qid}.a").write_text(report_md, encoding="utf-8")

# ────────────────────────────────────────────────────────────────
# 3) Main: read JSONL & dispatch
# ────────────────────────────────────────────────────────────────
async def main(input_path: str, concurrency: int = 8):
    # Load up to 1 000 lines of {"id","query"}
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(process_record(r, sem) for r in records))

# ────────────────────────────────────────────────────────────────
# 4) CLI
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPTResearcher batch runner")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to JSONL file with one {id,query} per line"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=8,
        help="Parallel research tasks"
    )
    args = parser.parse_args()
    asyncio.run(main(args.input, args.concurrency))
