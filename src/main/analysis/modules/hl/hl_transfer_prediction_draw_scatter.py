import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from src.modules.helper.helper import Helper


def draw_scatter_hl_transfer_prediction(code: str, from_dataset_name: str, to_dataset_name: str, language_name: str):
    df = pl.read_csv(Helper.ROOT / "logs" / code / "version_0" / "test_results.csv")
    df = df.with_columns(pl.col("log_halflife_pred").exp().alias("halflife_pred"))

    plt.figure(figsize=(7, 7))
    plt.scatter(df["halflife"], df["halflife_pred"], s=20)

    plt.xscale("log")
    plt.yscale("log")

    xymin = min(df["halflife"].to_list() + df["halflife_pred"].to_list())
    xymax = max(df["halflife"].to_list() + df["halflife_pred"].to_list())

    plt.plot([xymin, xymax], [xymin, xymax], color="red", linestyle="--", linewidth=1, label="Diagonal y = x")
    plt.grid()

    plt.xlabel("True $T_{1/2}$")
    plt.ylabel("Predicted $T_{1/2}$")

    y = df["log_halflife"].to_numpy()
    y_hat = df["log_halflife_pred"].to_numpy()

    pearsonr = stats.pearsonr(y, y_hat).correlation
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))

    plt.title(f"Pearson = {round(pearsonr, 4)}, RMSE = {round(rmse, 4)}")

    plt.legend()

    dir = Helper.ROOT / "output" / "figures" / "hl" / code
    dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(dir / f"{from_dataset_name}_to_{to_dataset_name}_{language_name}.png")


if __name__ == "__main__":
    items = [
        ("EXP5-1", "schwanhausser", "zecha", "esm2"),
        ("EXP6-1", "schwanhausser", "zecha", "saprot"),
        ("EXP7-1", "zecha", "schwanhausser", "esm2"),
        ("EXP8-1", "zecha", "schwanhausser", "saprot"),
    ]

    for code, from_dataset_name, to_dataset_name, language_name in items:
        draw_scatter_hl_transfer_prediction(code, from_dataset_name, to_dataset_name, language_name)
