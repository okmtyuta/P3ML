import math
from pathlib import Path
from src.modules.helper.helper import Helper

import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def draw_histogram(df: pl.DataFrame, path: Path):
    hls = df["log_halflife"].to_numpy()

    median = np.median(hls)
    p1, p99 = np.percentile(hls, [1, 99])

    plt.figure(figsize=(7, 6))
    plt.hist(hls, bins=50, color="lightblue", edgecolor="black", alpha=0.7)

    plt.axvline(
        median,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median = {round(median, 2)} ({round(math.exp(median), 2)} [h])",
    )

    plt.axvline(
        p1,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"1 percentile = {round(p1, 2)} ({round(math.exp(p1), 2)} [h])",
    )
    plt.axvline(
        p99,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"99 percentile = {round(p99, 2)} ({round(math.exp(p99), 2)} [h])",
    )
    plt.fill_betweenx(
        [0, plt.gca().get_ylim()[1]], p1, p99, color="green", alpha=0.1, label="1-99% range"
    )

    # plt.title("Histogram with Median and 99% Range")
    plt.xlabel("$T_{1/2}$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)


s_df = pl.read_csv(Helper.ROOT / "data" / "schwanhausser" / "data.csv")
z_df = pl.read_csv(Helper.ROOT / "data" / "zecha" / "data.csv")

draw_histogram(s_df, Helper.ROOT / 'output' / 'figures' / 'log_schwanhausser_histoguram.png')
draw_histogram(z_df, Helper.ROOT / 'output' / 'figures' / 'log_zecha_histoguram.png')
