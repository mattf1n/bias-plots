from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import (
    CF_TRAITS_ORDER,
    MODEL_COLORS,
    MODEL_ORDER,
    VALENCE_TRAIT_PAIRS,
    VALENCE_TRAIT_ORDER,
    build_cf_annotations,
    configure_matplotlib,
    darken_color,
    load_cf_gap_table,
    load_ci_ranges,
    load_cohens_d_excel,
)


configure_matplotlib()

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "data_files" / "ALL_results_combined-3.xlsx"


# ---------------------------------------------------------------------------
# CF Gap comparison (covert vs overt)
# ---------------------------------------------------------------------------
def compare_cf_gaps():
    covert_direct = {
        "Llama": "covert_direct_llama",
        "GPT": "covert_direct_GPT",
        "Deepseek": "covert_direct_deepseek",
    }
    covert_indirect = {
        "Llama": "covert_indirect_llama",
        "GPT": "covert_indirect_GPT",
        "Deepseek": "covert_indirect_deepseek",
    }
    overt_direct = {
        "Llama": "overt_direct_llama",
        "GPT": "overt_direct_GPT",
        "Deepseek": "overt_direct_deepseek",
    }
    overt_indirect = {
        "Llama": "overt_indirect_llama",
        "GPT": "overt_indirect_GPT",
        "Deepseek": "overt_indirect_deepseek",
    }

    covert_ci_dir = ROOT / "scores_confidence_significance"
    overt_ci_dir = ROOT / "scores_confidence_significance_overt"

    covert_ci_direct = {
        "Llama": covert_ci_dir / "llama_rel-Table 1.csv",
        "GPT": covert_ci_dir / "chatgpt_rel-chat_relative.csv",
        "Deepseek": covert_ci_dir / "deepseek_rel-Table 1.csv",
    }
    covert_ci_indirect = {
        "Llama": covert_ci_dir / "llama_abs-Table 1.csv",
        "GPT": covert_ci_dir / "chatgpt_abs-Table 1.csv",
        "Deepseek": covert_ci_dir / "deepseek_abs-Table 1.csv",
    }
    overt_ci_direct = {
        "Llama": overt_ci_dir / "llama_relative_overt.csv",
        "GPT": overt_ci_dir / "chatgpt_relative_overt.csv",
        "Deepseek": overt_ci_dir / "deepseek_relative_overt.csv",
    }
    overt_ci_indirect = {
        "Llama": overt_ci_dir / "llama_absolute_overt.csv",
        "GPT": overt_ci_dir / "chatgpt_absolute_overt.csv",
        "Deepseek": overt_ci_dir / "deepseek_absolute_overt.csv",
    }

    cov_direct_df = load_cf_gap_table(EXCEL_PATH, covert_direct, trait_order=CF_TRAITS_ORDER)
    cov_indirect_df = load_cf_gap_table(EXCEL_PATH, covert_indirect, trait_order=CF_TRAITS_ORDER)
    ov_direct_df = load_cf_gap_table(EXCEL_PATH, overt_direct, trait_order=CF_TRAITS_ORDER)
    ov_indirect_df = load_cf_gap_table(EXCEL_PATH, overt_indirect, trait_order=CF_TRAITS_ORDER)

    cov_direct_ci = load_ci_ranges(covert_ci_direct, trait_order=CF_TRAITS_ORDER)[cov_direct_df.columns]
    cov_indirect_ci = load_ci_ranges(covert_ci_indirect, trait_order=CF_TRAITS_ORDER)[cov_indirect_df.columns]
    ov_direct_ci = load_ci_ranges(overt_ci_direct, trait_order=CF_TRAITS_ORDER)[ov_direct_df.columns]
    ov_indirect_ci = load_ci_ranges(overt_ci_indirect, trait_order=CF_TRAITS_ORDER)[ov_indirect_df.columns]

    cov_direct_ann = build_cf_annotations(cov_direct_df, cov_direct_ci)
    cov_indirect_ann = build_cf_annotations(cov_indirect_df, cov_indirect_ci)
    ov_direct_ann = build_cf_annotations(ov_direct_df, ov_direct_ci)
    ov_indirect_ann = build_cf_annotations(ov_indirect_df, ov_indirect_ci)

    # shared color range across all four panels
    combined = [
        cov_indirect_df,
        cov_direct_df,
        ov_indirect_df,
        ov_direct_df,
    ]
    vmin = min(df.min().min() for df in combined)
    vmax = max(df.max().max() for df in combined)
    cmap = sns.light_palette("#4FA3A5", as_cmap=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), sharey=True)
    panels = [
        ("Covert Absolute", cov_indirect_df, cov_indirect_ann, axes[0, 0]),
        ("Covert Relative", cov_direct_df, cov_direct_ann, axes[0, 1]),
        ("Overt Absolute", ov_indirect_df, ov_indirect_ann, axes[1, 0]),
        ("Overt Relative", ov_direct_df, ov_direct_ann, axes[1, 1]),
    ]
    for title, data, annot, ax in panels:
        sns.heatmap(
            data,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=False,
            annot_kws={"fontsize": 6},
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        if "Relative" in title:
            ax.tick_params(axis="y", length=0)
    axes[0, 0].set_ylabel("Trait")
    axes[1, 0].set_ylabel("Trait")
    # single colorbar
    cbar = fig.colorbar(axes[0, 0].collections[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out = ROOT / "plots" / "compare_cf_gaps_covert_overt.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cohen's d comparisons (absolute and relative)
# ---------------------------------------------------------------------------
def _plot_cohens_d_on_axis(ax, df, title):
    models_in_order = MODEL_ORDER
    neg_colors = {model: darken_color(MODEL_COLORS[model], factor=0.70) for model in models_in_order}

    x = np.arange(len(VALENCE_TRAIT_PAIRS))
    bar_width = 0.22
    bars = {m: {"pos": [], "neg": []} for m in models_in_order}

    for i, model in enumerate(models_in_order):
        offset = (i - 1) * bar_width
        pos_vals = [df.loc[pos, model] for pos, _ in VALENCE_TRAIT_PAIRS]
        neg_vals = [df.loc[neg, model] for _, neg in VALENCE_TRAIT_PAIRS]
        pos_bars = ax.bar(x + offset, pos_vals, bar_width, color=MODEL_COLORS[model], label=model if i == 0 else None)
        neg_bars = ax.bar(x + offset, neg_vals, bar_width, color=neg_colors[model])
        bars[model]["pos"] = pos_bars
        bars[model]["neg"] = neg_bars

    def annotate(bar, val, hshift=0.0):
        x_text = bar.get_x() + bar.get_width() / 2 + hshift
        y_text = val + (0.04 if val >= 0 else -0.04)
        va = "bottom" if val >= 0 else "top"
        ax.text(x_text, y_text, f"{val:.2f}", ha="center", va=va, fontsize=6)

    for idx, (pos_trait, neg_trait) in enumerate(VALENCE_TRAIT_PAIRS):
        pos_vals = [df.loc[pos_trait, m] for m in models_in_order]
        pos_bars = [bars[m]["pos"][idx] for m in models_in_order]
        offsets = [-0.04, 0.0, 0.04] if (max(pos_vals) - min(pos_vals) < 0.10) else [0.0, 0.0, 0.0]
        for b, v, sh in zip(pos_bars, pos_vals, offsets):
            annotate(b, v, sh)
        neg_vals = [df.loc[neg_trait, m] for m in models_in_order]
        neg_bars = [bars[m]["neg"][idx] for m in models_in_order]
        for m, b, v in zip(models_in_order, neg_bars, neg_vals):
            y_text = v - 0.12 if neg_trait == "Laziness" and m == "Llama" else v - 0.05
            ax.text(b.get_x() + b.get_width() / 2, y_text, f"{v:.2f}", ha="center", va="top", fontsize=6)

    for y in np.arange(-2, 2.1, 0.5):
        ax.axhline(y, color="lightgray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1.2)
    for i, (pos, neg) in enumerate(VALENCE_TRAIT_PAIRS):
        ax.text(i, 1.02, pos, ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.text(i, -0.02, neg, ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks([])
    ax.set_ylim(-2, 2)
    ax.set_ylabel("Cohen's d")
    ax.set_title(title)


def compare_cohens_d():
    # Absolute: covert vs overt
    covert_abs_sheets = {"Llama": "covert_indirect_llama", "GPT": "covert_indirect_GPT", "Deepseek": "covert_indirect_deepseek"}
    overt_abs_sheets = {"Llama": "overt_indirect_llama", "GPT": "overt_indirect_GPT", "Deepseek": "overt_indirect_deepseek"}

    covert_abs = load_cohens_d_excel(EXCEL_PATH, covert_abs_sheets).loc[VALENCE_TRAIT_ORDER]
    overt_abs = load_cohens_d_excel(EXCEL_PATH, overt_abs_sheets).loc[VALENCE_TRAIT_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    _plot_cohens_d_on_axis(axes[0], covert_abs, "Covert Absolute")
    _plot_cohens_d_on_axis(axes[1], overt_abs, "Overt Absolute")
    axes[1].legend(title="Model", loc="upper right")
    fig.tight_layout()
    out = ROOT / "plots" / "compare_cohens_d_absolute.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # Relative: covert vs overt
    covert_rel_sheets = {"Llama": "covert_direct_llama", "GPT": "covert_direct_GPT", "Deepseek": "covert_direct_deepseek"}
    overt_rel_sheets = {"Llama": "overt_direct_llama", "GPT": "overt_direct_GPT", "Deepseek": "overt_direct_deepseek"}

    covert_rel = load_cohens_d_excel(EXCEL_PATH, covert_rel_sheets).loc[VALENCE_TRAIT_ORDER]
    overt_rel = load_cohens_d_excel(EXCEL_PATH, overt_rel_sheets).loc[VALENCE_TRAIT_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    _plot_cohens_d_on_axis(axes[0], covert_rel, "Covert Relative")
    _plot_cohens_d_on_axis(axes[1], overt_rel, "Overt Relative")
    axes[1].legend(title="Model", loc="upper right")
    fig.tight_layout()
    out = ROOT / "plots" / "compare_cohens_d_relative.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    compare_cf_gaps()
    compare_cohens_d()
