from pathlib import Path

from plot_utils import configure_matplotlib, load_self_consistency_frames, plot_self_consistency_heatmap

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
CHATGPT_PATH = ROOT / "data_files" / "self_consistency" / "raw_scores_covert_indirect_chatgpt.csv"
DEEPSEEK_PATH = ROOT / "data_files" / "self_consistency" / "parsed_score_covert_indirect_deepseek.csv"
LLAMA_PATH = ROOT / "data_files" / "self_consistency" / "covert_indirect_raw_response_parsed_full_llama.csv"

_, sae_df = load_self_consistency_frames(CHATGPT_PATH, DEEPSEEK_PATH, LLAMA_PATH)

plot_self_consistency_heatmap(
    sae_df,
    title="Self-Consistency Heatmap - SAE",
    outfile=ROOT / "plots" / "self_consistency_SAE.pdf",
)
