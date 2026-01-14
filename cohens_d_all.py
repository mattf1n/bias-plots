from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    MODEL_COLORS,
    MODEL_ORDER,
    VALENCE_TRAIT_PAIRS,
    VALENCE_TRAIT_ORDER,
    configure_matplotlib,
    darken_color,
    load_cohens_d_excel,
)


configure_matplotlib()

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "data_files" / "ALL_results_combined-3.xlsx"


def load_trait_ordered(sheet_map):
    df = load_cohens_d_excel(EXCEL_PATH, sheet_map)
    return df.loc[VALENCE_TRAIT_ORDER]


def draw_panel(ax, df, title):
    models_in_order = MODEL_ORDER
    neg_colors = {model: darken_color(MODEL_COLORS[model], factor=0.70) for model in models_in_order}
    x = np.arange(len(VALENCE_TRAIT_PAIRS))
    bar_width = 0.22
    bars = {model: {"pos": [], "neg": []} for model in models_in_order}

    for i, model in enumerate(models_in_order):
        offset = (i - 1) * bar_width
        pos_vals = [df.loc[pos, model] for pos, _ in VALENCE_TRAIT_PAIRS]
        neg_vals = [df.loc[neg, model] for _, neg in VALENCE_TRAIT_PAIRS]
        # negative first to keep darker/lighter layering consistent
        neg_bars = ax.bar(x + offset, neg_vals, bar_width, color=neg_colors[model])
        pos_bars = ax.bar(x + offset, pos_vals, bar_width, color=MODEL_COLORS[model], label=model)
        bars[model]["pos"] = pos_bars
        bars[model]["neg"] = neg_bars

    def annotate(bar, val, hshift=0.0, flip=False):
        x_text = bar.get_x() + bar.get_width() / 2 + hshift
        if flip:
            y_text = 0.04 if val < 0 else -0.04
        else:
            y_text = val + (0.04 if val >= 0 else -0.04)
        va = "bottom" if y_text > 0 else "top"
        ax.text(x_text, y_text, f"{val:.2f}", ha="center", va=va, rotation=90, fontsize=6)

    for idx, (pos_trait, neg_trait) in enumerate(VALENCE_TRAIT_PAIRS):
        pos_vals = [df.loc[pos_trait, m] for m in models_in_order]
        neg_vals = [df.loc[neg_trait, m] for m in models_in_order]
        pos_bars = [bars[m]["pos"][idx] for m in models_in_order]
        neg_bars = [bars[m]["neg"][idx] for m in models_in_order]
        offsets = [-0.04, 0.0, 0.04] if (max(pos_vals) - min(pos_vals) < 0.10) else [0.0, 0.0, 0.0]

        for j, (b, v, sh) in enumerate(zip(pos_bars, pos_vals, offsets)):
            flip = v * neg_vals[j] > 0 and abs(v) <= abs(neg_vals[j])
            annotate(b, v, sh, flip=flip)

        for j, (m, b, v) in enumerate(zip(models_in_order, neg_bars, neg_vals)):
            flip = v * pos_vals[j] > 0 and abs(v) < abs(pos_vals[j])
            if flip:
                y_text = 0.04 if v < 0 else -0.04
            else:
                y_text = v - 0.05
                if neg_trait == "Laziness" and m == "Llama":
                    y_text = v - 0.12
            va = "bottom" if y_text > 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, y_text, f"{v:.2f}", ha="center", va=va, rotation=90, fontsize=6)

    for y in np.arange(-2, 2.1, 0.5):
        ax.axhline(y, color="lightgray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(VALENCE_TRAIT_PAIRS)))
    ax.set_xticklabels([""] * len(VALENCE_TRAIT_PAIRS))
    ax.set_ylim(-2, 2)
    # y-label set outside to avoid duplicate labels
    ax.set_title(title, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("lightgray")
        spine.set_linewidth(1.0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(5.95, 4.0), sharey=True, gridspec_kw={"wspace": 0.05})

    covert_abs = load_trait_ordered(
        {"Llama": "covert_indirect_llama", "Deepseek": "covert_indirect_deepseek", "GPT": "covert_indirect_GPT"}
    )
    overt_abs = load_trait_ordered(
        {"Llama": "overt_indirect_llama", "Deepseek": "overt_indirect_deepseek", "GPT": "overt_indirect_GPT"}
    )
    covert_rel = load_trait_ordered(
        {"Llama": "covert_direct_llama", "Deepseek": "covert_direct_deepseek", "GPT": "covert_direct_GPT"}
    )
    overt_rel = load_trait_ordered(
        {"Llama": "overt_direct_llama", "Deepseek": "overt_direct_deepseek", "GPT": "overt_direct_GPT"}
    )
    # Deepseek overt relative has reversed sign convention; flip to align
    overt_rel["Deepseek"] *= -1

    panels = [
        ("Covert Absolute", covert_abs, axes[0, 0]),
        ("Covert Relative", covert_rel, axes[0, 1]),
        ("Overt Absolute", overt_abs, axes[1, 0]),
        ("Overt Relative", overt_rel, axes[1, 1]),
    ]
    for row_idx, (title, df, ax) in enumerate(panels):
        draw_panel(ax, df, title)
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel(r"Cohen's $d$")
        # Only top row shows x-axis labels (positive traits) at the top
        if ax in (axes[0, 0], axes[0, 1]):
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.set_xticklabels([p for p, _ in VALENCE_TRAIT_PAIRS], ha="left")
        else:
            # Bottom row shows negative traits on the bottom
            ax.xaxis.tick_bottom()
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.set_xticklabels([n for _, n in VALENCE_TRAIT_PAIRS], ha="right")

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "plots" / "COHENS_D_all.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
