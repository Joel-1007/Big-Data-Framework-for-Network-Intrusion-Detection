import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Results
try:
    with open("cross_data_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Run ids_cross_data.py first.")
    exit(1)

labels = results["labels"]
accuracy = results["accuracy"]
f1_score = results["f1_score"]

x = np.arange(len(labels))
width = 0.30

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 6))

# Color palette: blue for in-domain, orange for moderate drift, red for heavy drift
colors_acc = ['#2b8cbe', '#fdae6b', '#e6550d']
colors_f1 = ['#3182bd', '#fd8d3c', '#d94701']

rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color=colors_acc, edgecolor='white', linewidth=1.2)
rects2 = ax.bar(x + width/2, f1_score, width, label='F1 Score', color=colors_f1, edgecolor='white', linewidth=1.2)

ax.set_ylabel('Score (0.0 to 1.0)', fontweight='bold', fontsize=12)
ax.set_title('Zero-Shot Cross-Domain Generalization Under Concept Drift', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold', fontsize=10)
ax.set_ylim(0, 1.15)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add drift severity annotations
ax.annotate('Baseline', xy=(0, 0.05), fontsize=9, ha='center', color='#2b8cbe', fontweight='bold')
ax.annotate('Moderate Drift', xy=(1, 0.05), fontsize=9, ha='center', color='#fdae6b', fontweight='bold')
ax.annotate('Heavy Drift +\nGaussian Noise', xy=(2, 0.05), fontsize=9, ha='center', color='#e6550d', fontweight='bold')

ax.legend(loc='upper right', ncol=2, fontsize=11)
plt.tight_layout()

output_file = "generalization_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved publication-quality plot to {output_file}")
