import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

# Path setup
annotated_dir = Path("annotations")
annotators = ["pranav", "abhijay", "jingyuan", "joao"]

# Paper-ready label mapping
name_map = {
    "search-r1": "Search-R1",
    "search-o1": "Search-o1",
    "GPTResearcher": "GPT-Researcher",
    "open_deep_search": "OpenDeepSearch",
    "hf_deepresearch_gpt-4o-mini": "HF-DeepSearch",
}

# Final desired order (normalized keys)
ordered_systems = [
    "GPTResearcher",
    "open_deep_search",
    "hf_deepresearch_gpt-4o-mini",
    "search-o1",
    "search-r1",
]

def normalize(name):
    return (
        name.replace("_custom", "")
        .replace("search-r1-custom", "search-r1")
        .replace("search-o1-custom", "search-o1")
    )

# Aggregate head-to-head results
head2head_wins = defaultdict(lambda: defaultdict(int))
head2head_total = defaultdict(lambda: defaultdict(int))

for annotator in annotators:
    annotator_dir = annotated_dir / annotator
    if not annotator_dir.exists():
        continue

    for query_folder in annotator_dir.iterdir():
        qid = query_folder.name
        mapping_path = query_folder / "mapping.txt"
        if not mapping_path.exists():
            continue

        with open(mapping_path, "r") as f:
            mapping = dict(line.strip().split(": ") for line in f)

        sys1 = normalize(mapping['A'])
        sys2 = normalize(mapping['B'])

        if (query_folder / "A.txt").exists() and not (query_folder / "B.txt").exists():
            head2head_wins[sys1][sys2] += 1
        elif (query_folder / "B.txt").exists() and not (query_folder / "A.txt").exists():
            head2head_wins[sys2][sys1] += 1

        head2head_total[sys1][sys2] += 1
        head2head_total[sys2][sys1] += 1

# Matrix setup
n = len(ordered_systems)
matrix = np.zeros((n, n))

for i, s1 in enumerate(ordered_systems):
    for j, s2 in enumerate(ordered_systems):
        if s1 == s2:
            continue
        total = head2head_total[s1][s2]
        if total > 0:
            win_pct = 100 * head2head_wins[s1][s2] / total
            matrix[i, j] = win_pct

# Get display names in the same order
label_names = [name_map.get(s, s) for s in ordered_systems]

# Plotting
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)

fsize = 16
ax.set_xticks(np.arange(n))
ax.set_xticklabels(label_names, rotation=50, ha="left", fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_yticks(np.arange(n))
ax.set_yticklabels(label_names, fontsize=fsize)

# Annotate cells
for i in range(n):
    for j in range(n):
        if i != j:
            val = matrix[i, j]
            color = "black" if val < 70 else "white"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=fsize)

fig.tight_layout()
output_path = Path("model_head2head_matrix.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.close()
