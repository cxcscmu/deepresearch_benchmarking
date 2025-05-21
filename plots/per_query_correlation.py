import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns  # ✅ Import seaborn

# ✅ Top-level parameters you can tweak
FIGSIZE_SCALE = 1   # e.g., 0.7 makes everything smaller; 1.0 keeps original size
FONT_SCALE = 1.8     # e.g., 1.4 makes fonts bigger; 1.0 keeps original size

# ✅ Set Seaborn theme for clean styling & scaled fonts
sns.set_theme(style="whitegrid", font_scale=FONT_SCALE)

# ✅ Keep only the main three systems
SYSTEMS = ["GPTResearcher", "open_deep_search", "hf_deepresearch_gpt-4o-mini"]
COLORS = ["#007BC0", "#FDB515", "#009647"]

# ✅ Simplified row labels (bold on left side)
ROW_LABELS = ["Relevance\n(KPR)", "Quality\n(Average)", "Faithfulness\n(F1-score)"]

# Paths pattern
def build_paths(system, is_custom=False):
    suffix = "_custom" if is_custom else ""
    return {
        "detailed": f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/evaluation_results_detailed_gpt-4.1-mini.json',
        "citation": f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/evaluation_results_citation_gpt-4.1-mini.json',
        "citation_recall": f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/evaluation_results_citation_recall_gpt-4.1-mini.json',
        "kpr": f'/data/group_data/cx_group/deepsearch_benchmark/reports/{system}{suffix}/evaluation_results_kpr_gpt-4.1-mini.json',
    }

# ✅ New: only these two metrics for quality
holistic_metrics = ["Clarity", "Insightfulness"]

# # Prepare the figure (3 rows × 3 columns), scaled by FIGSIZE_SCALE
fig_width = 3 * 5 * FIGSIZE_SCALE
fig_height = 3 * 3.8 * FIGSIZE_SCALE  # reduced from 5 to 3.8
fig, axes = plt.subplots(3, 3, figsize=(fig_width*1.3, fig_height),
                       gridspec_kw={'wspace': 0.8, 'hspace': 0.5})

for idx, (sys, color) in enumerate(zip(SYSTEMS, COLORS)):
    # Load original
    paths_orig = build_paths(sys, is_custom=False)
    with open(paths_orig['detailed'], 'r') as f:
        detailed_orig = json.load(f)
    with open(paths_orig['citation'], 'r') as f:
        citation_orig = json.load(f)
    with open(paths_orig['citation_recall'], 'r') as f:
        citrec_orig = json.load(f)
    with open(paths_orig['kpr'], 'r') as f:
        kpr_orig = json.load(f)

    # Load custom
    paths_custom = build_paths(sys, is_custom=True)
    with open(paths_custom['detailed'], 'r') as f:
        detailed_custom = json.load(f)
    with open(paths_custom['citation'], 'r') as f:
        citation_custom = json.load(f)
    with open(paths_custom['citation_recall'], 'r') as f:
        citrec_custom = json.load(f)
    with open(paths_custom['kpr'], 'r') as f:
        kpr_custom = json.load(f)

    # Find common IDs
    common_ids = (
        set(detailed_orig.keys())
        & set(detailed_custom.keys())
        & set(citation_orig.keys())
        & set(citation_custom.keys())
        & set(citrec_orig.keys())
        & set(citrec_custom.keys())
        & set(kpr_orig.keys())
        & set(kpr_custom.keys())
    )

    print(f"[{sys}] Found {len(common_ids)} common IDs")

    # Collect scores
    kpr_orig_scores = []
    kpr_custom_scores = []
    quality_orig_scores = []
    quality_custom_scores = []
    f1_orig_scores = []
    f1_custom_scores = []

    for qid in common_ids:
        # KPR (no scaling)
        kpr_val_orig = kpr_orig[qid].get('support_rate', 0)
        kpr_val_custom = kpr_custom[qid].get('support_rate', 0)
        kpr_orig_scores.append(kpr_val_orig)
        kpr_custom_scores.append(kpr_val_custom)

        # Quality (Clarity + Insightfulness avg ×10)
        detailed_scores_orig = detailed_orig[qid]['scores']
        detailed_scores_custom = detailed_custom[qid]['scores']
        quality_vals_orig = [detailed_scores_orig[metric][0] for metric in holistic_metrics]
        quality_vals_custom = [detailed_scores_custom[metric][0] for metric in holistic_metrics]
        quality_avg_orig = np.mean(quality_vals_orig) * 10
        quality_avg_custom = np.mean(quality_vals_custom) * 10
        quality_orig_scores.append(quality_avg_orig)
        quality_custom_scores.append(quality_avg_custom)

        # Faithfulness F1 (×100)
        precision_orig = citation_orig[qid]['score'] * 100
        recall_orig = citrec_orig[qid]['score'] * 100
        precision_custom = citation_custom[qid]['score'] * 100
        recall_custom = citrec_custom[qid]['score'] * 100

        def compute_f1(p, r):
            if p + r == 0:
                return 0.0
            return 2 * (p * r) / (p + r)

        f1_orig = compute_f1(precision_orig, recall_orig)
        f1_custom = compute_f1(precision_custom, recall_custom)
        f1_orig_scores.append(f1_orig)
        f1_custom_scores.append(f1_custom)

    # Convert to np arrays
    x_kpr = np.array(kpr_orig_scores)
    y_kpr = np.array(kpr_custom_scores)
    x_quality = np.array(quality_orig_scores)
    y_quality = np.array(quality_custom_scores)
    x_f1 = np.array(f1_orig_scores)
    y_f1 = np.array(f1_custom_scores)

    # Row 1: KPR
    corr_kpr, _ = spearmanr(x_kpr, y_kpr)
    axes[0, idx].scatter(x_kpr, y_kpr, alpha=0.6, color=color)
    axes[0, idx].set_xlabel("Commercial Search API")
    axes[0, idx].set_ylabel("DRGym Search API")
    axes[0, idx].set_xlim(0, 100)
    axes[0, idx].set_ylim(0, 100)
    axes[0, idx].text(0.05, 0.95, f"Spearman={corr_kpr:.4f}",
                      transform=axes[0, idx].transAxes,
                      fontsize=int(11 * FONT_SCALE),
                      ha='left', va='top',
                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

    # Row 2: Quality
    corr_quality, _ = spearmanr(x_quality, y_quality)
    axes[1, idx].scatter(x_quality, y_quality, alpha=0.6, color=color)
    axes[1, idx].set_xlabel("Commercial Search API")
    axes[1, idx].set_ylabel("DRGym Search API")
    axes[1, idx].set_xlim(0, 100)
    axes[1, idx].set_ylim(0, 100)
    axes[1, idx].text(0.05, 0.95, f"Spearman={corr_quality:.4f}",
                      transform=axes[1, idx].transAxes,
                      fontsize=int(11 * FONT_SCALE),
                      ha='left', va='top',
                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

    # Row 3: Faithfulness F1
    corr_f1, _ = spearmanr(x_f1, y_f1)
    axes[2, idx].scatter(x_f1, y_f1, alpha=0.6, color=color)
    axes[2, idx].set_xlabel("Commercial Search API")
    axes[2, idx].set_ylabel("DRGym Search API")
    axes[2, idx].set_xlim(0, 100)
    axes[2, idx].set_ylim(0, 100)
    axes[2, idx].text(0.05, 0.95, f"Spearman={corr_f1:.4f}",
                      transform=axes[2, idx].transAxes,
                      fontsize=int(11 * FONT_SCALE),
                      ha='left', va='top',
                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

# Add global column titles
for idx, sys in enumerate(SYSTEMS):
    display_names = {
        "GPTResearcher": "GPTResearcher",
        "open_deep_search": "OpenDeepSearch",
        "hf_deepresearch_gpt-4o-mini": "HF-DeepResearch",
    }
    axes[0, idx].set_title(f"{display_names[sys]}",
                           fontsize=int(14 * FONT_SCALE),
                           fontweight='bold',
                           pad=18)  # reduced from 30

# Add global row labels
for row_idx, label in enumerate(ROW_LABELS):
    axes[row_idx, 0].annotate(label,
                              xy=(0, 0.5),
                              xytext=(-axes[row_idx, 0].yaxis.labelpad - 18, 0),
                              xycoords=axes[row_idx, 0].yaxis.label,
                              textcoords='offset points',
                              size=int(14 * FONT_SCALE),
                              ha='right', va='center',
                              fontweight='bold',
                              rotation=90,
                              multialignment='center')

plt.tight_layout()
output_pdf = 'plots/system_custom_comparison_scaled_updated.pdf'
plt.savefig(output_pdf)
plt.close()

print(f"Saved grid plot to {output_pdf}")