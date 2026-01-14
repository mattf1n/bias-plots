from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import configure_matplotlib

configure_matplotlib()

ROOT = Path(__file__).resolve().parent

chatgpt_csv = ROOT / "data_files" / "pearson_r" / "covert_abs" / "chatgpt_pearson_covert_absolute.csv"
deepseek_csv = ROOT / "data_files" / "pearson_r" / "covert_abs" / "deepseek_pearson_covert_absolute.csv"
llama_csv = ROOT / "data_files" / "pearson_r" / "covert_abs" / "llama_pearson_covert_absolute.csv"

df_gpt = pd.read_csv(chatgpt_csv)
df_ds = pd.read_csv(deepseek_csv)
df_ll = pd.read_csv(llama_csv)

df_gpt["Model"] = "GPT-4.0-mini"
df_ds["Model"] = "DeepSeek-V3"
df_ll["Model"] = "LLaMA-3.1-8B"

df_all = pd.concat([df_gpt, df_ds, df_ll], ignore_index=True)

long_df = df_all.melt(
    id_vars=["Valence Pair", "Model"],
    value_vars=["Pearson_r_SAE", "Pearson_r_AAVE"],
    var_name="Dialect",
    value_name="Correlation",
)
long_df["Dialect"] = long_df["Dialect"].replace({"Pearson_r_SAE": "SAE", "Pearson_r_AAVE": "AAVE"})
long_df["Model-Dialect"] = long_df["Model"] + " (" + long_df["Dialect"] + ")"

heatmap_df = long_df.pivot(index="Valence Pair", columns="Model-Dialect", values="Correlation")
heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)

fig, ax = plt.subplots(figsize=(8.0, 4.5))
sns.heatmap(
    heatmap_df,
    annot=True,
    cmap=sns.light_palette("#4FA3A5", as_cmap=True),
    center=0,
    fmt=".2f",
    linewidths=0.5,
cbar_kws={"label": "Pearson $r$"},
    ax=ax,
)

ax.set_title("Pearson Correlation Between Positive/Negative Trait Pairs Across Models")
ax.set_xlabel(r"Model $\times$ Dialect")
ax.set_ylabel("Valence Pair")

plt.tight_layout()
outfile = ROOT / "plots" / "pearson_correlations_covert_abs.pdf"
outfile.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outfile, dpi=300)
plt.close(fig)
