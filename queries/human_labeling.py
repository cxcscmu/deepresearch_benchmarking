import os
import random
from pathlib import Path

base_dir = Path("/data/group_data/cx_group/deepsearch_benchmark/reports/") 
output_root = Path("/data/group_data/cx_group/deepsearch_benchmark/human_labeling/")
output_root.mkdir(parents=True, exist_ok=True)

original_systems = ["GPTResearcher", "hf_deepresearch_gpt-4o-mini", "open_deep_search", "search-r1", "search-o1"]
custom_systems = ["GPTResearcher_custom", "hf_deepresearch_gpt-4o-mini_custom", "open_deep_search_custom", "search-r1-custom", "search-o1_custom"]
annotators = ["joao", "ethan", "jingyuan", "karry", "pranav", "abhijay", "jiahe"]  

query_ids = [f.stem for f in (base_dir / original_systems[0]).glob("*.q")]
print(f"Found {len(query_ids)} queries.")

sample_size = min(210, len(query_ids))
random.seed(42)
sampled_query_ids = random.sample(query_ids, sample_size)

def pick_two_distinct(systems):
    return random.sample(systems, 2)

def pick_three_annotators(annotators):
    return random.sample(annotators, 3)

summary = []

for i, qid in enumerate(sampled_query_ids):
    if i < sample_size // 2:
        sys1, sys2 = pick_two_distinct(original_systems)
    else:
        sys1, sys2 = pick_two_distinct(custom_systems)

    a1_path = base_dir / sys1 / f"{qid}.a"
    a2_path = base_dir / sys2 / f"{qid}.a"
    q_path = base_dir / original_systems[0] / f"{qid}.q"

    if not (a1_path.exists() and a2_path.exists()):
        print(f"Warning: Missing files for query {qid}, skipping.")
        continue

    with open(a1_path, "r") as f:
        a1_text = f.read()
    with open(a2_path, "r") as f:
        a2_text = f.read()
    with open(q_path, "r") as f:
        q_text = f.read()

    systems = [("A", sys1, a1_text), ("B", sys2, a2_text)]

    ann1, ann2, ann3 = pick_three_annotators(annotators)
    summary.append((qid, ann1, ann2, ann3))

    for ann in [ann1, ann2, ann3]:
        ann_folder = output_root / ann / qid
        ann_folder.mkdir(parents=True, exist_ok=True)

        with open(ann_folder / "query.txt", "w") as f:
            f.write(q_text)

        for label, sys_name, text in systems:
            with open(ann_folder / f"{label}.txt", "w") as f:
                f.write(text)

        with open(ann_folder / "label.csv", "w") as f:
            f.write("label\n")

        with open(ann_folder / "mapping.txt", "w") as f:
            for label, sys_name, _ in systems:
                f.write(f"{label}: {sys_name}\n")

with open(output_root / "annotator_summary.csv", "w") as f:
    f.write("query_id,annotator1,annotator2,annotator3\n")
    for qid, ann1, ann2, ann3 in summary:
        f.write(f"{qid},{ann1},{ann2},{ann3}\n")

print(f"Created folder structure under {output_root} for annotators and saved summary file.")
