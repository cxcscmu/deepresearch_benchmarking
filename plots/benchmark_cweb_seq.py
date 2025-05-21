import time
import requests
import json
import random
from tqdm import tqdm
import numpy as np

random.seed(17121998)

import random, nltk
nltk.download('words')
from nltk.corpus import words
TEST_QUERIES = [' '.join(random.sample(words.words(), random.randint(3,5))) for _ in range(800)]

# # Load test queries
# TEST_QUERIES = []
# with open("/home/jmcoelho/deepresearch-eval/queries/researchy_queries_sample_doc_click.jsonl", 'r') as h:
#     for line in h:
#         data = json.loads(line)
#         TEST_QUERIES.append(data["query"])
#         if len(TEST_QUERIES) == 700:
#             break

sampled_queries = random.choices(TEST_QUERIES, k=100)
TEST_QUERIES += sampled_queries

BASE_URL = "https://clueweb22.us/search"

def benchmark_cweb_sync(queries):
    latencies = []
    for query in tqdm(queries):
        url = f"{BASE_URL}?query={query}&k=10"
        start = time.time()
        response = requests.get(url)
        response.raise_for_status()
        latencies.append(time.time() - start)
    return latencies

if __name__ == "__main__":
    print(f"Benchmarking ClueWeb sequentially with {len(TEST_QUERIES)} queries...")
    latencies = benchmark_cweb_sync(TEST_QUERIES)
    print(f"Completed {len(latencies)} successful requests.")

    if latencies:
        median_latency = np.median(latencies)
        print(f"Median latency: {median_latency:.3f} seconds")
    else:
        print("No successful requests to compute median.")