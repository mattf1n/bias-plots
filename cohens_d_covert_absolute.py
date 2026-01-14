from pathlib import Path

from plot_utils import VALENCE_TRAIT_ORDER, configure_matplotlib, load_cohens_d_excel, plot_valence_bars

configure_matplotlib()

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "data_files" / "ALL_results_combined-3.xlsx"

sheets = {
    "Llama": "covert_indirect_llama",
    "GPT": "covert_indirect_GPT",
    "Deepseek": "covert_indirect_deepseek",
}

cohens_d = load_cohens_d_excel(EXCEL_PATH, sheets).loc[VALENCE_TRAIT_ORDER]

def main() -> None:
    plot_valence_bars(
        cohens_d,
        outfile=ROOT / "plots" / "COHENS_D_covert_absolute_models.pdf",
    )


if __name__ == "__main__":
    main()
