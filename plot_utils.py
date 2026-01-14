"""Shared helpers for generating plots from the graph notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_FONT_FAMILY = "Libertinus Serif"
DEFAULT_FONT_SIZE = 9

MODEL_COLORS = {
    "Deepseek": "#A8D0E6",
    "GPT": "#F6A8B2",
    "Llama": "#FFEBA8",
}
MODEL_ORDER = ["Llama", "Deepseek", "GPT"]

CF_TRAITS_ORDER = [
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

VALENCE_TRAIT_PAIRS = [
    ("Intelligence", "Stupidity"),
    ("Calmness", "Aggression"),
    ("Sophistication", "Unsophistication"),
    ("Politeness", "Rudeness"),
    ("Articulation", "Incoherence"),
    ("Determination", "Laziness"),
]
VALENCE_TRAIT_ORDER = [trait for pair in VALENCE_TRAIT_PAIRS for trait in pair]

SELF_CONSISTENCY_TRAITS = [
    "Aggression",
    "Articulation",
    "Calmness",
    "Determination",
    "Incoherence",
    "Intelligence",
    "Laziness",
    "Politeness",
    "Rudeness",
    "Sophistication",
    "Stupidity",
    "Unsophistication",
]


def configure_matplotlib(font_family: str = DEFAULT_FONT_FAMILY, font_size: int = DEFAULT_FONT_SIZE) -> None:
    """Apply consistent Matplotlib defaults with LaTeX rendering and Libertine fonts."""
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{libertinus}"


def darken_color(hex_color: str, factor: float = 0.70) -> str:
    """Return a darker shade of a hex color."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = np.clip(rgb * factor, 0, 255).astype(int)
    return "#" + "".join(f"{v:02X}" for v in rgb)


# ---------------------------------------------------------------------------
# CF gap heatmaps
# ---------------------------------------------------------------------------
def load_cf_gap_table(
    excel_path: Path,
    sheet_map: Dict[str, str],
    value_col: str = "CTF_Gap",
    trait_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load CTF gap values from the Excel workbook."""
    xls = pd.ExcelFile(excel_path)
    table = {model: xls.parse(sheet).set_index("Trait")[value_col] for model, sheet in sheet_map.items()}
    df = pd.DataFrame(table)
    df = df[[c for c in MODEL_ORDER if c in df.columns]]  # enforce consistent model order
    if trait_order:
        df = df.reindex(trait_order)
    return df


def load_ci_ranges(
    csv_map: Dict[str, Path],
    low_col: str = "CF_Gap_CI_Low",
    high_col: str = "CF_Gap_CI_High",
    trait_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load CI ranges and format them as strings `(low, high)`, tolerating missing files."""
    ci = {}
    for model, path in csv_map.items():
        if not path.exists():
            ci[model] = pd.Series(dtype=float)
            continue
        frame = pd.read_csv(path)
        frame["CI_Range"] = frame.apply(lambda row: f"({row[low_col]:.2f}, {row[high_col]:.2f})", axis=1)
        ci[model] = frame.set_index("Trait")["CI_Range"]
    df = pd.DataFrame(ci)
    if trait_order:
        df = df.reindex(trait_order)
    return df


def build_cf_annotations(values: pd.DataFrame, ci_ranges: pd.DataFrame) -> pd.DataFrame:
    """Create `value (low, high)` annotations for the CF heatmaps."""
    annotations = pd.DataFrame(index=values.index, columns=values.columns)
    for col in values.columns:
        for idx in values.index:
            val = values.loc[idx, col]
            ci = ci_ranges.loc[idx, col]
            if pd.isna(val):
                annotations.loc[idx, col] = ""
                continue
            if pd.isna(ci):
                annotations.loc[idx, col] = f"{val:.2f}"
            else:
                annotations.loc[idx, col] = f"{val:.2f} {ci}"
    return annotations


def plot_cf_gap_heatmaps(
    indirect_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    indirect_annots: pd.DataFrame,
    direct_annots: pd.DataFrame,
    outfile: Path,
    titles: Tuple[str, str] = ("Absolute Prompting", "Relative Prompting"),
    figsize: Tuple[float, float] = (5.95, 3.0),
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render the paired CF heatmaps to a single figure."""
    if vmin is None or vmax is None:
        combined = pd.concat([indirect_df, direct_df])
        vmin = combined.min().min()
        vmax = combined.max().max()
    cmap = sns.light_palette("#4FA3A5", as_cmap=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    sns.heatmap(
        indirect_df,
        annot=indirect_annots,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        cbar=False,
        annot_kws={"fontsize": 6},
    )
    axes[0].set_title(titles[0])
    axes[0].set_ylabel("Trait")
    axes[0].set_xlabel("Model")
    axes[0].tick_params(axis="x")

    sns.heatmap(
        direct_df,
        annot=direct_annots,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
        cbar=False,
        annot_kws={"fontsize": 6},
    )
    axes[1].set_title(titles[1])
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Model")
    axes[1].tick_params(axis="y", length=0)

    for axis in axes:
        axis.set_xticklabels(axis.get_xticklabels())

    plt.tight_layout()
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, fraction=0.05, pad=0.02)
    cbar.outline.set_visible(False)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Valence bar charts (Cohen's d)
# ---------------------------------------------------------------------------
def plot_valence_bars(
    combined_df: pd.DataFrame,
    outfile: Path,
    trait_pairs: Sequence[Tuple[str, str]] = VALENCE_TRAIT_PAIRS,
    colors: Dict[str, str] = MODEL_COLORS,
    figsize: Tuple[float, float] = (5.95, 3.0),
    ylim: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    """Plot paired positive/negative trait bars for each model."""
    models_in_order = MODEL_ORDER
    neg_colors = {m: darken_color(c, factor=0.70) for m, c in colors.items()}

    x = np.arange(len(trait_pairs))
    bar_width = 0.22
    fig, ax = plt.subplots(figsize=figsize)
    bars = {model: {"pos": [], "neg": []} for model in models_in_order}

    for i, model in enumerate(models_in_order):
        offset = (i - 1) * bar_width
        pos_vals = [combined_df.loc[pos, model] for pos, _ in trait_pairs]
        neg_vals = [combined_df.loc[neg, model] for _, neg in trait_pairs]

        # Draw negative bars first so positive bars (lighter) stay visible when overlapping
        neg_bars = ax.bar(x + offset, neg_vals, bar_width, color=neg_colors[model])
        pos_bars = ax.bar(x + offset, pos_vals, bar_width, color=colors[model], label=model)
        bars[model]["pos"] = pos_bars
        bars[model]["neg"] = neg_bars

    def annotate_bar(bar, value, hshift: float = 0.0, flip: bool = False) -> None:
        x_text = bar.get_x() + bar.get_width() / 2 + hshift
        if flip:
            # put label on opposite side of zero to avoid collisions
            y_text = 0.04 if value < 0 else -0.04
            va = "bottom" if y_text > 0 else "top"
        else:
            y_text = value + (0.04 if value >= 0 else -0.04)
            va = "bottom" if value >= 0 else "top"
        ax.text(x_text, y_text, f"{value:.2f}", ha="center", va=va, fontsize=6)

    for idx, (pos_trait, neg_trait) in enumerate(trait_pairs):
        pos_vals = [combined_df.loc[pos_trait, m] for m in models_in_order]
        pos_bars = [bars[m]["pos"][idx] for m in models_in_order]
        pos_offsets = [-0.04, 0.00, 0.04] if (max(pos_vals) - min(pos_vals) < 0.10) else [0.00, 0.00, 0.00]
        neg_vals = [combined_df.loc[neg_trait, m] for m in models_in_order]
        neg_bars = [bars[m]["neg"][idx] for m in models_in_order]

        for j, (bar, val, shift) in enumerate(zip(pos_bars, pos_vals, pos_offsets)):
            flip = False
            if val * neg_vals[j] > 0:  # same sign, avoid overlap
                flip = abs(val) <= abs(neg_vals[j])
            annotate_bar(bar, val, shift, flip=flip)

        for j, (model, bar, val) in enumerate(zip(models_in_order, neg_bars, neg_vals)):
            flip = False
            if val * pos_vals[j] > 0:
                flip = abs(val) < abs(pos_vals[j])
            x_text = bar.get_x() + bar.get_width() / 2
            if flip:
                y_text = 0.04 if val < 0 else -0.04
                va = "bottom" if y_text > 0 else "top"
            else:
                y_text = val - 0.12 if neg_trait == "Laziness" and model == "Llama" else val - 0.05
                va = "top"
            ax.text(x_text, y_text, f"{val:.2f}", ha="center", va=va, fontsize=6)

    for y in np.arange(ylim[0], ylim[1] + 0.1, 0.5):
        ax.axhline(y, color="lightgray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1.2)

    for i, (pos, neg) in enumerate(trait_pairs):
        ax.text(i, 1.02, pos, ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.text(i, -0.02, neg, ha="center", va="top", transform=ax.get_xaxis_transform())

    ax.set_xticks([])
    ax.set_ylim(ylim)
    ax.set_ylabel(r"Cohen's $d$")
    ax.tick_params(axis="y")
    ax.legend(title=None, loc="upper right")
    ax.text(0.5, 1.08, "Positive-Valence Traits", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, -0.10, "Negative-Valence Traits", ha="center", va="top", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def load_cohens_d_excel(excel_path: Path, sheet_map: Dict[str, str]) -> pd.DataFrame:
    """Load Cohen's d values for models from specific sheets."""
    workbook = pd.read_excel(excel_path, sheet_name=None)
    data = {}
    for model, sheet in sheet_map.items():
        frame = workbook[sheet]
        data[model] = frame.set_index("Trait")["Cohens_d"]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Self-consistency helpers
# ---------------------------------------------------------------------------
def self_consistency_chatgpt(csv_path: Path, dialect: str = "SAE") -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    results = {}
    for trait in SELF_CONSISTENCY_TRAITS:
        col = f"{trait}_{dialect}"
        consistency = df.groupby("Tweet_Index")[col].nunique() == 1
        results[trait] = consistency.mean()
    return results


def self_consistency_deepseek(csv_path: Path, dialect: str = "SAE") -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    suffix = "1" if dialect == "SAE" else "2"
    results = {}
    for trait in SELF_CONSISTENCY_TRAITS:
        col = f"{trait}{suffix}"
        scores = pd.to_numeric(df[col], errors="coerce")
        consistency = scores.groupby(df["Item"]).nunique() == 1
        results[trait] = consistency.mean()
    return results


def self_consistency_llama(csv_path: Path, dialect: str = "SAE") -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    filtered = df[df["RowType"] == dialect].copy()
    results = {}
    for trait in SELF_CONSISTENCY_TRAITS:
        consistency = filtered.groupby("OriginalIndex")[trait].nunique() == 1
        results[trait] = consistency.mean()
    return results


def plot_self_consistency_heatmap(data: pd.DataFrame, title: str, outfile: Path) -> None:
    """Plot a single self-consistency heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data.values, vmin=0.6, vmax=1.0, aspect="auto", cmap=sns.light_palette("#4FA3A5", as_cmap=True))
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.tick_params(axis="x", rotation=15)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Self-Consistency")

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_self_consistency_frames(chatgpt_path: Path, deepseek_path: Path, llama_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute self-consistency DataFrames for AAVE and SAE."""
    chatgpt_sae = self_consistency_chatgpt(chatgpt_path, "SAE")
    chatgpt_aave = self_consistency_chatgpt(chatgpt_path, "AAVE")

    deepseek_sae = self_consistency_deepseek(deepseek_path, "SAE")
    deepseek_aave = self_consistency_deepseek(deepseek_path, "AAVE")

    llama_sae = self_consistency_llama(llama_path, "SAE")
    llama_aave = self_consistency_llama(llama_path, "AAVE")

    def ordered(series_dict: Dict[str, float]) -> pd.Series:
        return pd.Series(series_dict).reindex(SELF_CONSISTENCY_TRAITS)

    sae_df = pd.DataFrame(
        {
            "DeepSeek": ordered(deepseek_sae),
            "ChatGPT": ordered(chatgpt_sae),
            "LLaMA": ordered(llama_sae),
        }
    )
    aave_df = pd.DataFrame(
        {
            "DeepSeek": ordered(deepseek_aave),
            "ChatGPT": ordered(chatgpt_aave),
            "LLaMA": ordered(llama_aave),
        }
    )

    return aave_df, sae_df


# ---------------------------------------------------------------------------
# Q-values helper
# ---------------------------------------------------------------------------
def compute_q_values(logprob_csv: Path, traits: Sequence[str], scores: Sequence[int]) -> pd.DataFrame:
    """Compute Q-values from log probability CSV."""
    df = pd.read_csv(logprob_csv)
    rows: List[pd.DataFrame] = []
    for trait in traits:
        tmp = pd.DataFrame(
            {
                "Trait": trait,
                "SAE_Score": df[f"{trait}_SAE_Score"],
                "AAVE_Score": df[f"{trait}_AAVE_Score"],
                "SAE_LogProb": df[f"{trait}_SAE_LogProb"],
                "AAVE_LogProb": df[f"{trait}_AAVE_LogProb"],
            }
        )
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    long_df["LogRatio"] = long_df["AAVE_LogProb"] - long_df["SAE_LogProb"]
    q_values = long_df.groupby(["Trait", "AAVE_Score"])[["LogRatio"]].mean().unstack()
    q_values.columns = q_values.columns.droplevel(0)
    q_values = q_values.reindex(columns=scores, fill_value=0.0)
    q_values.columns = [f"Score {s}" for s in scores]
    return q_values
