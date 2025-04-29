import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress

with open('/data/group_data/cx_group/deepsearch_benchmark/reports/GPTResearcher/evaluation_results_per_query_score_gpt-4.1-mini.json', 'r') as f:
    data1 = json.load(f)

with open('/data/group_data/cx_group/deepsearch_benchmark/reports/GPTResearcher_custom/evaluation_results_per_query_score_gpt-4.1-mini.json', 'r') as f:
    data2 = json.load(f)

x = []
y = []
for qid in data1:
    if qid in data2:
        x.append(data1[qid]["with_citations"])
        y.append(data2[qid]["with_citations"])

x_np = np.array(x)
y_np = np.array(y)

# Compute linear regression
slope, intercept, r_value, _, _ = linregress(x_np, y_np)
regression_line = slope * x_np + intercept

# Compute correlations
pearson_corr, _ = pearsonr(x_np, y_np)
spearman_corr, _ = spearmanr(x_np, y_np)

# Create scatter plot
plt.figure(figsize=(6,6))
plt.scatter(x_np, y_np, alpha=0.6, label='Data points')
plt.plot(x_np, regression_line, color='red', label='Regression line')
plt.xlabel('Original API')
plt.ylabel('ClueWeb22 API')
plt.title('Scatter Plot of Without-Citation Scores')
plt.grid(True)

# Annotate correlation coefficients
plt.text(0.05, 0.95,
         f'Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}',
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

plt.legend()
plt.tight_layout()
plt.savefig('scatter_plot.pdf', format='pdf')
plt.close()