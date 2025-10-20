import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats

from src.modules.helper.helper import Helper


def draw_scatter_hl_prediction(code: str, dataset_name: str, language_name: str):
    with open(Helper.ROOT / "logs" / code / "note.json", mode="r") as f:
        note = json.load(f)
        basic_version = note["basic_version"]

    df = pl.read_csv(Helper.ROOT / "logs" / code / basic_version / "test_results.csv")
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

    dir = Helper.ROOT / "output" / "figures" / "scatter_prediction_hl" / code
    dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(dir / f"{dataset_name}_{language_name}.png")


if __name__ == "__main__":
    items = [
        ("EXP1", "schwanhausser", "esm2"),
        ("EXP2", "schwanhausser", "saprot"),
        ("EXP3", "zecha", "esm2"),
        ("EXP4", "zecha", "saprot"),
    ]

    for code, dataset_name, language_name in items:
        draw_scatter_hl_prediction(code, dataset_name, language_name)
