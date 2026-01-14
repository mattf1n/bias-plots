from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import configure_matplotlib

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
ABSOLUTE_CSV = ROOT / "data_files" / "cohens_d" / "finetune_llama" / "covert_absolute_llama.csv"
RELATIVE_CSV = ROOT / "data_files" / "cohens_d" / "finetune_llama" / "covert_relative_llama.csv"
FINETUNED_REL_CSV = ROOT / "data_files" / "cohens_d" / "finetune_llama" / "covert_relative_llama_finetuning.csv"
FINETUNED_ABS_CSV = ROOT / "data_files" / "cohens_d" / "finetune_llama" / "covert_absolute_llama_finetuning.csv"

TRAIT_ORDER = [
    "Intelligence",
    "Calmness",
    "Sophistication",
    "Politeness",
    "Articulation",
    "Determination",
    "Stupidity",
    "Aggression",
    "Unsophistication",
    "Rudeness",
    "Incoherence",
    "Laziness",
]

df_abs = pd.read_csv(ABSOLUTE_CSV).set_index("Trait").loc[TRAIT_ORDER]
df_rel = pd.read_csv(RELATIVE_CSV).set_index("Trait").loc[TRAIT_ORDER]
df_ft_r = pd.read_csv(FINETUNED_REL_CSV).set_index("Trait").loc[TRAIT_ORDER]
df_ft_a = pd.read_csv(FINETUNED_ABS_CSV).set_index("Trait").loc[TRAIT_ORDER]

delta_abs = df_ft_a["Cohens_d"] - df_abs["Cohens_d"]
delta_rel = df_ft_r["Cohens_d"] - df_rel["Cohens_d"]

ABS_DELTA_COLOR = "#B084CC"
REL_DELTA_COLOR = "#4C9A61"

x = np.arange(len(TRAIT_ORDER))
width = 0.35
fig, ax = plt.subplots(figsize=(5.95, 3.0))

bars1 = ax.bar(x - width / 2, delta_abs, width, label="Absolute", color=ABS_DELTA_COLOR)
bars2 = ax.bar(x + width / 2, delta_rel, width, label="Relative", color=REL_DELTA_COLOR)


def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        ypos = h + 0.02 if h >= 0 else h - 0.02
        va = "bottom" if h >= 0 else "top"
        ax.text(xpos, ypos, f"{h:.2f}", ha="center", va=va, fontsize=6)


add_labels(bars1)
add_labels(bars2)

ax.axhline(0, color="black", linewidth=1)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(-0.5, None)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(TRAIT_ORDER, rotation=30, ha="right")
ax.set_ylabel(r"$\Delta$ Cohen's $d$ (Finetuned - Original)")
ax.legend()

plt.tight_layout()
outfile = ROOT / "plots" / "cohens_d_finetuned.pdf"
outfile.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outfile)
plt.close(fig)
