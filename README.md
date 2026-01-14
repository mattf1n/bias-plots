Plots for the covert/overt analysis pulled out of the original notebook. Each script reads the data in `data_files/` (and related subfolders), writes a PDF to `plots/`, and can be run directly with `uv`.

## Setup
- Install [uv](https://docs.astral.sh/uv/).
- From the repo root, install dependencies and create the virtual environment:
  - `uv sync`
- Run any script through uv so it picks up the managed environment:
  - `uv run python <script.py>`

## Plotting scripts
- `cf_gap_heatmaps_covert.py`: Heatmaps of covert absolute vs. relative CTF gaps with confidence-interval annotations (`plots/CF_Gap_Heatmaps_covert.pdf`).
- `cf_gap_heatmaps_overt.py`: Overt version of the CTF gap heatmaps (`plots/CF_Gap_Heatmaps_overt.pdf`).
- `compare_main_plots.py`: Side-by-side covert/overt CTF gap panels and Cohen’s d comparisons (absolute + relative) written to `plots/compare_cf_gaps_covert_overt.pdf`, `plots/compare_cohens_d_absolute.pdf`, and `plots/compare_cohens_d_relative.pdf`.
- `cohens_d_all.py`: Four-panel Cohen’s d grid (covert/overt × absolute/relative) saved as `plots/COHENS_D_all.pdf`.
- `cohens_d_covert_absolute.py`, `cohens_d_covert_relative.py`, `cohens_d_overt_absolute.py`, `cohens_d_overt_relative.py`: Per-setting bar charts of positive/negative valence traits for each model (outputs prefixed with `plots/COHENS_D_...pdf`).
- `cohens_d_finetuned.py`: Finetuning deltas for LLaMA Cohen’s d scores (`plots/cohens_d_finetuned.pdf`).
- `q_values_covert.py`: Q-value heatmap for covert prompting (`plots/Q_values_covert.pdf`).
- `pearson_correlations_covert_abs.py`: Pearson r heatmap for positive/negative trait correlations across models (`plots/pearson_correlations_covert_abs.pdf`).
- `score_winner_difference_covert_direct_deepseek.py`: Winner grid and score count differences for DeepSeek covert direct prompting (`plots/score_winner_difference_covert_direct_deepseek.pdf`).
- `self_consistency_aave.py`, `self_consistency_sae.py`: Self-consistency heatmaps for AAVE and SAE dialects (`plots/self_consistency_AAVE.pdf`, `plots/self_consistency_SAE.pdf`).
- `Graph_code.py`: Legacy notebook-style scratchpad; prefer the dedicated scripts above for reproducible plots.

### Notes
- Scripts expect the Excel and CSV inputs already present under `data_files/` and related folders. No downloads are performed at runtime.
- Matplotlib is configured to use LaTeX + Libertinus (`plot_utils.configure_matplotlib`). Ensure the fonts/LaTeX package are available in your environment if you see font warnings.
