import matplotlib.pyplot as plt
import numpy as np

# Data
k_values = np.array([1, 10, 25, 50, 100])
p50_latency = np.array([0.1263, 0.1731, 0.2608, 0.4537, 0.8996])
p90_latency = np.array([0.1629, 0.2481, 0.4295, 0.7657, 1.5762])
p95_latency = np.array([0.1710, 0.3031, 0.5319, 0.9385, 1.8879])
p99_latency = np.array([0.1986, 0.3925, 0.7643, 1.2305, 2.4799])

# Font size settings
fsize = 16
plt.rcParams.update({
    "font.size": fsize,
    "axes.titlesize": fsize,
    "axes.labelsize": fsize,
    "xtick.labelsize": fsize,
    "ytick.labelsize": fsize,
    "legend.fontsize": fsize,
})

# Initialize figure
plt.figure(figsize=(4, 4))  # Half-column width

# Define consistent blue shades and distinct markers
colors = {
    'P50 (Median)': '#007BC0',
    'P90': '#3399CC',
    'P95': '#66ADD3',
    'P99': '#99C2DB'
}
markers = {
    'P50 (Median)': 'o',  # Circle
    'P90': 's',           # Square
    'P95': '^',           # Triangle up
    'P99': 'X'            # X-shaped
}
linewidths = {
    'P50 (Median)': 2.5,
    'P90': 2.0,
    'P95': 1.7,
    'P99': 1.5
}

# Plot each percentile with uniform line style but different markers and widths
plt.plot(k_values, p50_latency, label='P50 (Median)', color=colors['P50 (Median)'],
         linestyle='-', marker=markers['P50 (Median)'], linewidth=linewidths['P50 (Median)'])

plt.plot(k_values, p90_latency, label='P90', color=colors['P90'],
         linestyle='-', marker=markers['P90'], linewidth=linewidths['P90'])

plt.plot(k_values, p95_latency, label='P95', color=colors['P95'],
         linestyle='-', marker=markers['P95'], linewidth=linewidths['P95'])

plt.plot(k_values, p99_latency, label='P99', color=colors['P99'],
         linestyle='-', marker=markers['P99'], linewidth=linewidths['P99'])

# Axis and legend
plt.xticks(k_values)
plt.xlabel('K (#documents)')
plt.ylabel('Latency (seconds)')
plt.legend(loc='upper left')
plt.grid(True)

# Save
plt.tight_layout()
plt.savefig('plots/latency_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')