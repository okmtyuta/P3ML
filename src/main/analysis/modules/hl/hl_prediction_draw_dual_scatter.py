import matplotlib.pyplot as plt
import polars as pl
from scipy import stats

from src.modules.helper.helper import Helper


def hl_prediction_draw_dual_scatter(esm2_code: str, saprot_code: str, dataset_name: str):
    esm2_df = pl.read_csv(Helper.ROOT / "logs" / esm2_code / "version_0" / "test_results.csv")
    esm2_df = esm2_df.with_columns(pl.col("log_halflife_pred").exp().alias("halflife_pred"))

    saprot_df = pl.read_csv(Helper.ROOT / "logs" / saprot_code / "version_0" / "test_results.csv")
    saprot_df = saprot_df.with_columns(pl.col("log_halflife_pred").exp().alias("halflife_pred"))

    plt.figure(figsize=(7, 7))
    plt.scatter(esm2_df["halflife"], esm2_df["halflife_pred"], s=15, label="ESM2", alpha=0.6)
    plt.scatter(saprot_df["halflife"], saprot_df["halflife_pred"], s=15, label="SaProt", alpha=0.6)

    plt.xscale("log")
    plt.yscale("log")

    esm2_halflife = esm2_df["halflife"].to_list() + esm2_df["halflife_pred"].to_list()
    saprot_halflife = saprot_df["halflife"].to_list() + saprot_df["halflife_pred"].to_list()
    halflife = esm2_halflife + saprot_halflife
    xymin, xymax = min(halflife), max(halflife)

    plt.plot([xymin, xymax], [xymin, xymax], color="red", linestyle="--", linewidth=1, label="Diagonal y = x")
    plt.grid()

    plt.xlabel("True $T_{1/2}$")
    plt.ylabel("Predicted $T_{1/2}$")

    esm2_y = esm2_df["log_halflife"].to_numpy()
    esm2_y_hat = esm2_df["log_halflife_pred"].to_numpy()
    esm2_pearsonr = stats.pearsonr(esm2_y, esm2_y_hat).correlation

    saprot_y = saprot_df["log_halflife"].to_numpy()
    saprot_y_hat = saprot_df["log_halflife_pred"].to_numpy()
    saprot_pearsonr = stats.pearsonr(saprot_y, saprot_y_hat).correlation

    plt.title(f"ESM2 Pearson = {round(esm2_pearsonr, 4)}, SaProt Pearson = {round(saprot_pearsonr, 4)}")

    plt.legend()

    dir = Helper.ROOT / "output" / "figures" / "hl" / "dual"
    dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(dir / f"{dataset_name}.png")


if __name__ == "__main__":
    configs = [
        ("EXP1-2", "EXP2-2", "schwanhausser"),
        # ('EXP3-2', 'EXP4-2', 'zecha'),
    ]

    for config in configs:
        esm2_code, saprot_code, dataset_name = config
        hl_prediction_draw_dual_scatter(esm2_code=esm2_code, saprot_code=saprot_code, dataset_name=dataset_name)
