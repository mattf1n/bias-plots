from pathlib import Path

from plot_utils import (
    CF_TRAITS_ORDER,
    build_cf_annotations,
    configure_matplotlib,
    load_cf_gap_table,
    load_ci_ranges,
    plot_cf_gap_heatmaps,
)

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "data_files" / "ALL_results_combined-3.xlsx"
CI_DIR = ROOT / "scores_confidence_significance"
CI_DIR_OVERT = ROOT / "scores_confidence_significance_overt"

direct_sheets = {
    "Llama": "covert_direct_llama",
    "GPT": "covert_direct_GPT",
    "Deepseek": "covert_direct_deepseek",
}
indirect_sheets = {
    "Llama": "covert_indirect_llama",
    "GPT": "covert_indirect_GPT",
    "Deepseek": "covert_indirect_deepseek",
}

ci_direct_files = {
    "Llama": CI_DIR / "llama_rel-Table 1.csv",
    "GPT": CI_DIR / "chatgpt_rel-chat_relative.csv",
    "Deepseek": CI_DIR / "deepseek_rel-Table 1.csv",
}
ci_indirect_files = {
    "Llama": CI_DIR / "llama_abs-Table 1.csv",
    "GPT": CI_DIR / "chatgpt_abs-Table 1.csv",
    "Deepseek": CI_DIR / "deepseek_abs-Table 1.csv",
}

direct_df = load_cf_gap_table(EXCEL_PATH, direct_sheets, trait_order=CF_TRAITS_ORDER)
indirect_df = load_cf_gap_table(EXCEL_PATH, indirect_sheets, trait_order=CF_TRAITS_ORDER)

direct_ci = load_ci_ranges(ci_direct_files).reindex(CF_TRAITS_ORDER)[direct_df.columns]
indirect_ci = load_ci_ranges(ci_indirect_files).reindex(CF_TRAITS_ORDER)[indirect_df.columns]

direct_annots = build_cf_annotations(direct_df, direct_ci)
indirect_annots = build_cf_annotations(indirect_df, indirect_ci)

# Compute shared scale across covert + overt to keep figures comparable
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
ov_direct_df = load_cf_gap_table(EXCEL_PATH, overt_direct, trait_order=CF_TRAITS_ORDER)
ov_indirect_df = load_cf_gap_table(EXCEL_PATH, overt_indirect, trait_order=CF_TRAITS_ORDER)

all_vals = [direct_df, indirect_df, ov_direct_df, ov_indirect_df]
vmin = min(df.min().min() for df in all_vals)
vmax = max(df.max().max() for df in all_vals)

plot_cf_gap_heatmaps(
    indirect_df,
    direct_df,
    indirect_annots,
    direct_annots,
    outfile=ROOT / "plots" / "CF_Gap_Heatmaps_covert.pdf",
    vmin=vmin,
    vmax=vmax,
)
