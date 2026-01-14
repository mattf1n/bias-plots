from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from plot_utils import compute_q_values, configure_matplotlib

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
LOGPROB_PATH = ROOT / "data_files" / "logprobs" / "covert_majority_llama_scores.csv"

TRAITS = [
    "Intelligence",
    "Stupidity",
    "Calmness",
    "Aggression",
    "Sophistication",
    "Unsophistication",
    "Politeness",
    "Rudeness",
    "Articulation",
    "Incoherence",
    "Determination",
    "Laziness",
]
SCORES = [1, 2, 3, 4, 5]

q_values = compute_q_values(LOGPROB_PATH, TRAITS, SCORES)
positive = ["Intelligence", "Calmness", "Sophistication", "Politeness", "Articulation", "Determination"]
negative = ["Stupidity", "Aggression", "Unsophistication", "Rudeness", "Incoherence", "Laziness"]
q_values = q_values.reindex(positive + negative)

fig, ax = plt.subplots(figsize=(5.95, 2.75))
pastel_diverging = LinearSegmentedColormap.from_list(
    "pastel_blue_orange", ["#E7B7C8", "#F7F7F7", "#8EC6C5"], N=256
)

sns.heatmap(
    q_values,
    annot=True,
    fmt=".2f",
    cmap=pastel_diverging,
    center=0,
    vmin=-0.3,
    vmax=0.3,
    linewidths=0.5,
    ax=ax,
)

ax.set_xlabel("Score")
ax.set_ylabel("Trait")

plt.tight_layout()
outfile = ROOT / "plots" / "Q_values_covert.pdf"
outfile.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outfile)
plt.close(fig)
