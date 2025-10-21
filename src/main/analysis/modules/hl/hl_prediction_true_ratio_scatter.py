import json

import matplotlib.pyplot as plt
import polars as pl

from src.modules.helper.helper import Helper

# 上位nパーセントの部分
ACTUAL_THRESHOLD_PERCENTILE = 90
PREDICTION_THRESHOLD_PERCENTILE = 90

COLORS = {
    "upper-right": {"color": "tab:red", "label": ""},
    "upper-left": {"color": "tab:orange", "label": ""},
    "lower-right": {"color": "tab:green", "label": ""},
    "lower-left": {"color": "tab:gray", "label": ""},
}


def get_threshold(df: pl.DataFrame):
    actual_threshold = df.select(pl.col("halflife").quantile(ACTUAL_THRESHOLD_PERCENTILE / 100)).item()
    prediction_threshold = df.select(pl.col("halflife_pred").quantile(PREDICTION_THRESHOLD_PERCENTILE / 100)).item()

    return actual_threshold, prediction_threshold


def create_region(df: pl.DataFrame):
    actual_threshold, prediction_threshold = get_threshold(df=df)

    def region(row: dict):
        actual = row["halflife"]
        prediction = row["halflife_pred"]

        if actual >= actual_threshold and prediction >= prediction_threshold:
            return "upper-right"
        elif actual < actual_threshold and prediction >= prediction_threshold:
            return "upper-left"
        elif actual >= actual_threshold and prediction < prediction_threshold:
            return "lower-right"
        else:
            return "lower-left"

    return region


def create_region_df(df: pl.DataFrame):
    region = create_region(df=df)
    df = df.with_columns(pl.struct(["halflife", "halflife_pred"]).map_elements(region, return_dtype=pl.Utf8).alias("region"))

    return df


def hl_prediction_true_ratio_scatter(code: str):
    with open(Helper.ROOT / "logs" / code / "note.json", mode="r") as f:
        note = json.load(f)
        basic_version = note["basic_version"]

    df = pl.read_csv(Helper.ROOT / "logs" / code / basic_version / "test_results.csv")
    df = df.with_columns(pl.col("log_halflife_pred").exp().alias("halflife_pred"))

    df = create_region_df(df=df)

    plt.figure(figsize=(7, 7))
    for reg, config in COLORS.items():
        sub = df.filter(pl.col("region") == reg)
        plt.scatter(sub["halflife"], sub["halflife_pred"], label=config["label"], color=config["color"], s=20, edgecolors="none")

    plt.xscale("log")
    plt.yscale("log")

    actual_threshold, prediction_threshold = get_threshold(df=df)
    plt.axvline(x=actual_threshold, color="black", linestyle="--", linewidth=1)
    plt.axhline(y=prediction_threshold, color="black", linestyle="--", linewidth=1)

    xymin = min(df["halflife"].to_list() + df["halflife_pred"].to_list())
    xymax = max(df["halflife"].to_list() + df["halflife_pred"].to_list())

    plt.plot([xymin, xymax], [xymin, xymax], color="red", linestyle="--", linewidth=1, label="Diagonal y = x")
    plt.grid()

    plt.xlabel("True $T_{1/2}$")
    plt.ylabel("Predicted $T_{1/2}$")
    # plt.title("Log-scale Scatter Plot with 4 Regions and Diagonal", fontsize=16, pad=10)
    plt.legend()
    plt.tight_layout()
    # plt.tick_params(axis="both", which="major", labelsize=16)
    # plt.subplots_adjust(left=0.12, right=0.96, bottom=0.08, top=0.92)
    # plt.savefig("clss.png")
    dir = Helper.ROOT / "output" / "figures" / "hl" / code / f"basic_version_{basic_version}"
    dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(dir / f"{ACTUAL_THRESHOLD_PERCENTILE}_{PREDICTION_THRESHOLD_PERCENTILE}_true_ratio.png")


if __name__ == "__main__":
    for code in ["EXP1", "EXP2", "EXP3", "EXP4"]:
        hl_prediction_true_ratio_scatter(code="EXP2")
