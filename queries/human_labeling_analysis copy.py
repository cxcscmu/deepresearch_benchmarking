import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

# Paths
base_dir = Path("/data/group_data/cx_group/deepsearch_benchmark/reports/")
annotated_dir = Path("/home/jmcoelho/deepresearch-eval/annotations")
summary_csv = Path("/data/group_data/cx_group/deepsearch_benchmark/human_labeling/annotator_summary.csv")

# You maintain this list and can expand it as needed
annotators = ["pranav", "abhijay", "jingyuan", "joao"]

metrics = ['KPR', 'Citation', 'Clarity', 'Insightfulness']
agreement_counts = defaultdict(lambda: {'agree': 0, 'total': 0})
raw_labels = defaultdict(lambda: {'human': [], 'llm': []})
annotator_votes = defaultdict(lambda: defaultdict(dict))  # metric -> query -> annotator -> vote

# Load summary CSV to find queries where ALL annotators have completed

completed_queries = set()

with open(summary_csv, 'r') as f:
    header = f.readline().strip().split(",")
    for line in f:
        qid, *q_annotators = line.strip().split(",")
        # check that all annotators are in the list
        for ann in q_annotators:
            if ann not in annotators:
                break
        else:
            completed_queries.add(qid)       

print(f"Found {len(completed_queries)} queries with all annotators.")

def get_eval_metrics(qid, system, is_custom=False):
    suffix = "_custom" if is_custom else ""
    base_path = f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/'

    try:
        with open(f'{base_path}evaluation_results_detailed_gpt-4.1-mini.json') as f:
            detailed = json.load(f)
        with open(f'{base_path}evaluation_results_citation_gpt-4.1-mini.json') as f:
            citation = json.load(f)
        with open(f'{base_path}evaluation_results_kpr_gpt-4.1-mini.json') as f:
            kpr = json.load(f)
    except FileNotFoundError:
        return None

    if qid not in detailed or qid not in citation or qid not in kpr:
        return None

    metrics_out = {
        'KPR': kpr[qid].get('support_rate'),
        'Citation': citation[qid].get('score')
    }

    holistic_metrics = ["Clarity", "Depth", "Balance", "Breadth", "Support", "Insightfulness"]
    scores = detailed[qid].get('scores', {})
    for metric in holistic_metrics:
        val = scores.get(metric, [None])[0]
        metrics_out[metric] = val

    return metrics_out

# Collect votes and LLM predictions
for annotator in annotators:
    annotator_dir = annotated_dir / annotator
    if not annotator_dir.exists():
        print(f"Annotator folder {annotator_dir} not found, skipping.")
        continue

    for query_folder in annotator_dir.iterdir():
        qid = query_folder.name
        if qid not in completed_queries:
            continue

        mapping_path = query_folder / "mapping.txt"
        if not mapping_path.exists():
            continue

        with open(mapping_path, "r") as f:
            mapping = dict(line.strip().split(": ") for line in f)

        # Normalize system names
        mapping = {k: v.replace("search-r1-custom", "search-r1_custom") for k, v in mapping.items()}
        sys1_name = mapping['A'].replace("_custom", "")
        sys2_name = mapping['B'].replace("_custom", "")
        is_custom1 = "_custom" in mapping['A']
        is_custom2 = "_custom" in mapping['B']

        # Determine human preference
        if (query_folder / "A.txt").exists() and not (query_folder / "B.txt").exists():
            human_label = 1
        elif (query_folder / "B.txt").exists() and not (query_folder / "A.txt").exists():
            human_label = 0
        else:
            print(f"Missing label: {annotator}, {qid}")
            continue 

        # Store annotator vote
        for metric in metrics:
            annotator_votes[metric][qid][annotator] = human_label

        # Only store LLM label once per query (use first annotator pass)
        if annotator == annotators[0]:
            m1 = get_eval_metrics(qid, sys1_name, is_custom1)
            m2 = get_eval_metrics(qid, sys2_name, is_custom2)
            if not m1 or not m2:
                continue

            for metric in metrics:
                v1, v2 = m1.get(metric), m2.get(metric)
                if v1 is None or v2 is None or v1 == v2:
                    continue
                llm_label = 1 if v1 > v2 else 0
                raw_labels[metric]['llm'].append(llm_label)

# Compute majority-vote results
print("\nLLM-Human Agreement per Metric (+ Cohen's Kappa):")
for metric in metrics:
    qids = annotator_votes[metric].keys()
    llm_labels = []
    majority_labels = []
    for qid in qids:
        votes = list(annotator_votes[metric][qid].values())
        if len(votes) < 2:
            continue
        count = Counter(votes)
        if count[0] == count[1]:  # Tie in human votes
            continue
        majority = 1 if count[1] > count[0] else 0

        # Get LLM label: we can *safely* fetch from any annotator folder (e.g., the first)
        any_annotator = next(iter(annotators))
        query_folder = annotated_dir / any_annotator / qid
        mapping_path = query_folder / "mapping.txt"
        if not mapping_path.exists():
            continue

        with open(mapping_path, "r") as f:
            mapping = dict(line.strip().split(": ") for line in f)
        mapping = {k: v.replace("search-r1-custom", "search-r1_custom") for k, v in mapping.items()}
        sys1_name = mapping['A'].replace("_custom", "")
        sys2_name = mapping['B'].replace("_custom", "")
        is_custom1 = "_custom" in mapping['A']
        is_custom2 = "_custom" in mapping['B']

        m1 = get_eval_metrics(qid, sys1_name, is_custom1)
        m2 = get_eval_metrics(qid, sys2_name, is_custom2)
        if not m1 or not m2:
            continue

        v1, v2 = m1.get(metric), m2.get(metric)
        if v1 is None or v2 is None or v1 == v2:
            continue

        llm_label = 1 if v1 > v2 else 0

        # Both labels exist now:
        llm_labels.append(llm_label)
        majority_labels.append(majority)

    if not majority_labels:
        print(f"{metric:15s}: N/A")
        continue

    agree = sum(h == l for h, l in zip(majority_labels, llm_labels))
    total = len(majority_labels)
    pct = (agree / total) * 100
    kappa = cohen_kappa_score(majority_labels, llm_labels)
    print(f"{metric:15s}: {pct:.2f}% agreement, κ = {kappa:.3f} | {total}")

# Inter-annotator agreement
print("\nInter-Annotator Agreement (Cohen's Kappa):")
for metric in metrics:
    qids = annotator_votes[metric].keys()
    for a1, a2 in combinations(annotators, 2):
        y1, y2 = [], []
        for qid in qids:
            votes = annotator_votes[metric][qid]
            if a1 in votes and a2 in votes:
                y1.append(votes[a1])
                y2.append(votes[a2])
        if y1:
            kappa = cohen_kappa_score(y1, y2)
            print(f"{metric:15s} {a1:12s}-{a2:12s}: κ = {kappa:.3f} | {len(y1)}")