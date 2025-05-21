import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

# Paths
base_dir = Path("/data/group_data/cx_group/deepsearch_benchmark/reports/")
annotated_dir = Path("/home/jmcoelho/deepresearch-eval/annotations")
annotators = ["pranav", "abhijay", "jingyuan", "joao"]

# Metrics to track (KPC removed)
metrics = ['KPR', 'Citation Precision', 'Citation Recall', 'Clarity', 'Insightfulness']
agreement_counts = defaultdict(lambda: {'agree': 0, 'total': 0})
raw_labels = defaultdict(lambda: {'human': [], 'llm': []})
wins = Counter()
losses = Counter()

def normalize_sys_name(name):
    return name.replace("_custom", "")

def get_eval_metrics(qid, system, is_custom=False):
    suffix = "_custom" if is_custom else ""
    base_path = f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/'

    paths = {
        "detailed": f'{base_path}evaluation_results_detailed_gpt-4.1-mini.json',
        "citation": f'{base_path}evaluation_results_citation_gpt-4.1-mini.json',
        "citation_recall": f'{base_path}evaluation_results_citation_recall_gpt-4.1-mini.json',
        "kpr": f'{base_path}evaluation_results_kpr_gpt-4.1-mini.json',
    }

    try:
        with open(paths['detailed'], 'r') as f:
            detailed = json.load(f)
        with open(paths['citation'], 'r') as f:
            citation = json.load(f)
        with open(paths['citation_recall'], 'r') as f:
            citation_recall = json.load(f)
        with open(paths['kpr'], 'r') as f:
            kpr = json.load(f)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

    if qid not in detailed or qid not in citation or qid not in kpr or qid not in citation_recall:
        print(f"Query ID {qid} not found in one or more files for system {system}{suffix}.")
        return None

    metrics_out = {}
    metrics_out['KPR'] = kpr[qid].get('support_rate', None)
    metrics_out['Citation Precision'] = citation[qid].get('score', None)
    metrics_out['Citation Recall'] = citation_recall[qid].get('score', None)

    holistic_metrics = ["Clarity", "Depth", "Balance", "Breadth", "Support", "Insightfulness"]
    scores = detailed[qid].get('scores', {})
    for metric in holistic_metrics:
        val = scores.get(metric, [None])[0]
        metrics_out[metric] = val

    return metrics_out

# Collect human labels per query per annotator
human_labels = defaultdict(dict)  # qid -> annotator -> label

# Main loop
for annotator in annotators:
    annotator_dir = annotated_dir / annotator
    if not annotator_dir.exists():
        print(f"Annotator folder {annotator_dir} not found, skipping.")
        continue

    for query_folder in annotator_dir.iterdir():
        if not query_folder.is_dir():
            continue

        qid = query_folder.name
        mapping_path = query_folder / "mapping.txt"
        if not mapping_path.exists():
            print(f"Missing mapping.txt in {query_folder}, skipping.")
            continue

        mapping = {}
        with open(mapping_path, "r") as f:
            for line in f:
                label, sys_name = line.strip().split(": ")
                mapping[label] = sys_name

        if (query_folder / "A.txt").exists() and (query_folder / "B.txt").exists():
            continue
        if (query_folder / "A.txt").exists():
            human_pref = "A"
        elif (query_folder / "B.txt").exists():
            human_pref = "B"
        else:
            continue

        human_labels[qid][annotator] = 1 if human_pref == "A" else 0

        if mapping['A'] == "search-r1-custom":
            mapping['A'] = "search-r1_custom"
        if mapping['B'] == "search-r1-custom":
            mapping['B'] = "search-r1_custom"

        sys1_name = mapping['A'].replace("_custom", "")
        sys2_name = mapping['B'].replace("_custom", "")
        is_custom1 = "_custom" in mapping['A']
        is_custom2 = "_custom" in mapping['B']

        metrics_1 = get_eval_metrics(qid, sys1_name, is_custom=is_custom1)
        metrics_2 = get_eval_metrics(qid, sys2_name, is_custom=is_custom2)

        if metrics_1 is None or metrics_2 is None:
            continue

        for metric in metrics:
            val1 = metrics_1.get(metric)
            val2 = metrics_2.get(metric)
            if val1 is None or val2 is None or val1 == val2:
                continue

            human_label = 1 if human_pref == "A" else 0
            llm_label = 1 if val1 > val2 else 0

            raw_labels[metric]['human'].append(human_label)
            raw_labels[metric]['llm'].append(llm_label)

            agreement_counts[metric]['total'] += 1
            if human_label == llm_label:
                agreement_counts[metric]['agree'] += 1

        winner_sys = normalize_sys_name(mapping[human_pref])
        loser_sys = normalize_sys_name(mapping["B" if human_pref == "A" else "A"])
        wins[winner_sys] += 1
        losses[loser_sys] += 1

# LLM-Human agreement
print("\nLLM-Human Agreement per Metric (+ Cohen's Kappa, Win %, Loss %):")
for metric in metrics:
    total = agreement_counts[metric]['total']
    agree = agreement_counts[metric]['agree']
    human = raw_labels[metric]['human']
    llm = raw_labels[metric]['llm']

    if total == 0:
        print(f"{metric:20s}: N/A")
        continue

    kappa = cohen_kappa_score(human, llm)
    win = sum(1 for h, l in zip(human, llm) if h == l)
    lose = sum(1 for h, l in zip(human, llm) if h != l)
    win_pct = (win / total) * 100
    lose_pct = (lose / total) * 100
    agree_pct = (agree / total) * 100

    print(f"{metric:20s}: {agree_pct:.1f}% agree, κ = {kappa:.3f} | Win = {win_pct:.1f}%, Lose = {lose_pct:.1f}% | N = {total}")

# System-level win/loss summary
print("\nSystem-level Human Preference Counts:")
all_systems = sorted(set(wins.keys()).union(losses.keys()))
for system in all_systems:
    w = wins.get(system, 0)
    l = losses.get(system, 0)
    print(f"{system:25s}: Wins = {w:3d} | Losses = {l:3d} | Net = {w - l:+3d}")

# Human-Human agreement
print("\nPairwise Human Agreement (Cohen's Kappa):")
pairwise_kappas = []

for a1, a2 in combinations(annotators, 2):
    shared_qids = [qid for qid in human_labels if a1 in human_labels[qid] and a2 in human_labels[qid]]
    if not shared_qids:
        continue

    labels1 = [human_labels[qid][a1] for qid in shared_qids]
    labels2 = [human_labels[qid][a2] for qid in shared_qids]
    kappa = cohen_kappa_score(labels1, labels2)
    pairwise_kappas.append(kappa)
    print(f"{a1:10s} vs {a2:10s}: κ = {kappa:.3f} (N = {len(shared_qids)})")

if pairwise_kappas:
    avg_kappa = sum(pairwise_kappas) / len(pairwise_kappas)
    print(f"\nAverage Human-Human Cohen's Kappa: {avg_kappa:.3f}")
else:
    print("\nNo overlapping annotations found between annotators.")