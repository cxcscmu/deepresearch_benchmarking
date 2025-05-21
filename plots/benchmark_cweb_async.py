import asyncio
import aiohttp
import time
import json
import random
import numpy as np
from tqdm import tqdm

random.seed(17121998)

# Load test queries
TEST_QUERIES = []
with open("/home/jmcoelho/deepresearch-eval/queries/researchy_queries_sample_doc_click.jsonl", 'r') as h:
    for line in h:
        data = json.loads(line)
        TEST_QUERIES.append(data["query"])
        if len(TEST_QUERIES) == 700:
            break

# Add 100 random samples again
sampled_queries = random.choices(TEST_QUERIES, k=100)
TEST_QUERIES += sampled_queries

# URL format
BASE_URL = "https://clueweb22.us/search"

async def fetch(session, query, semaphore):
    url = f"{BASE_URL}?query={query}&k=10"
    async with semaphore:
        start = time.time()
        try:
            async with session.get(url) as response:
                await response.text()  # Force content read
                latency = time.time() - start
                if response.status != 200:
                    print(f"Error {response.status} for query: {query}")
                return latency
        except Exception as e:
            print(f"Exception for query: {query} | {e}")
            return None

async def benchmark_cweb_concurrent(queries, concurrent_requests=1):
    latencies = []
    semaphore = asyncio.Semaphore(concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, query, semaphore) for query in queries]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            latency = await f
            if latency is not None:
                latencies.append(latency)
    return latencies

if __name__ == "__main__":
    print(f"Benchmarking ClueWeb concurrently with {len(TEST_QUERIES)} queries...")
    latencies = asyncio.run(benchmark_cweb_concurrent(TEST_QUERIES, concurrent_requests=800))
    print(f"Completed {len(latencies)} successful requests.")

    if latencies:
        median_latency = np.median(latencies)
        print(f"Median latency: {median_latency:.3f} seconds")
    else:
        print("No successful requests to compute median.")