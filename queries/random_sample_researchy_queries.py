## This is how the queries were sampled. comment to avoid reruning and rewriting the file.

# import random
# import json
# from datasets import load_dataset

# dataset = load_dataset("corbyrosset/researchy_questions")
# queries = dataset["test"]

# random.seed(42)
# indices = random.sample(range(len(queries)), 1000)
# sampled = [queries[i] for i in indices]

# output_path = "queries/researchy_queries_sample.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for entry in sampled:
#         json.dump({"id": entry["id"], "query": entry["question"]}, f)
#         f.write("\n")