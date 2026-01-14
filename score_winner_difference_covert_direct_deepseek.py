from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from plot_utils import configure_matplotlib

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_files" / "score_count" / "score_covert_direct_deepseek.csv"

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    traits = df["Trait"].tolist()

    winner_matrix = []
    difference_matrix = []
    winner_labels = []

    for _, row in df.iterrows():
        w_row, d_row, l_row = [], [], []
        for score in range(1, 6):
            aave = int(row[f"Score_{score}_AAVE"])
            sae = int(row[f"Score_{score}_SAE"])
            diff = aave - sae

            if diff > 0:
                w_row.append(1)
                l_row.append("AAVE")
            elif diff < 0:
                w_row.append(-1)
                l_row.append("SAE")
            else:
                w_row.append(0)
                l_row.append("Equal")

            d_row.append(diff)
        winner_matrix.append(w_row)
        difference_matrix.append(d_row)
        winner_labels.append(l_row)

    winner_array = np.array(winner_matrix)
    diff_array = np.array(difference_matrix)

    cat_colors = ["#F6C49A", "#FFFFFF", "#A7C7E7"]
    cmap_cat = ListedColormap(cat_colors)
    norm_cat = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_cat.N)

    pastel_diverging = LinearSegmentedColormap.from_list(
        "pastel_blue_orange",
        ["#E7B7C8", "#F7F7F7", "#8EC6C5"],
    )

    max_abs = int(max(abs(diff_array.min()), abs(diff_array.max())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 6.0), sharey=True)

    ax1.imshow(winner_array, cmap=cmap_cat, norm=norm_cat, aspect="auto")
    for i in range(len(traits)):
        for j in range(5):
            ax1.text(j, i, winner_labels[i][j], ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels([f"Score {s}" for s in range(1, 6)], fontsize=10)
    ax1.set_yticks(np.arange(len(traits)))
    ax1.set_yticklabels(traits, fontsize=10, fontweight="bold")
    ax1.set_title("Which Dialect Received Each Score More Often?\n(Blue = AAVE, Orange = SAE)", fontsize=12, pad=14)
    ax1.set_xlabel("Score", fontsize=10)
    ax1.set_ylabel("Trait", fontsize=10)
    ax1.set_xticks(np.arange(5) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(traits)) - 0.5, minor=True)
    ax1.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax1.tick_params(which="minor", bottom=False, left=False)

    im2 = ax2.imshow(diff_array, cmap=pastel_diverging, aspect="auto", vmin=-max_abs, vmax=max_abs)
    for i in range(len(traits)):
        for j in range(5):
            diff = int(difference_matrix[i][j])
            ax2.text(
                j,
                i,
                f"{diff:+d}" if diff != 0 else "0",
                ha="center",
                va="center",
                color="white" if abs(diff) > 0.6 * max_abs else "black",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels([f"Score {s}" for s in range(1, 6)], fontsize=10)
    ax2.set_yticks(np.arange(len(traits)))
    ax2.set_yticklabels(traits, fontsize=10, fontweight="bold")
    ax2.set_title("Count Difference (AAVE - SAE)\n(Blue = AAVE higher, Orange = SAE higher)", fontsize=12, pad=14)
    ax2.set_xlabel("Score", fontsize=10)
    ax2.set_xticks(np.arange(5) - 0.5, minor=True)
    ax2.set_yticks(np.arange(len(traits)) - 0.5, minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Difference (AAVE - SAE)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    outfile = ROOT / "plots" / "score_winner_difference_covert_direct_deepseek.pdf"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
