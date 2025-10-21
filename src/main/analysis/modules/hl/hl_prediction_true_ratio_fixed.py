import json
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.modules.helper.helper import Helper

ACTUAL_THRESHOLD_PERCENTILE = 90


def hl_prediction_true_ratio_fixed(code: str, version: Optional[str] = None):
    if version is None:
        with open(Helper.ROOT / "logs" / code / "note.json", mode="r") as f:
            note = json.load(f)
            version = note["basic_version"]

    Ns: list[float] = np.round(np.arange(0.01, 1.01, 0.01), 2).tolist()
    Ns = [N * 100 for N in Ns]
    probs: list[float] = []
    random_probs: list[float] = []

    df = pl.read_csv(Helper.ROOT / "logs" / code / version / "test_results.csv")

    actual_threshold = df.select(pl.col("log_halflife").quantile(ACTUAL_THRESHOLD_PERCENTILE / 100)).item()
    actual_keys = df.filter(pl.col("log_halflife") >= actual_threshold)["key"].to_list()
    actual_key_set = set(actual_keys)

    for N in Ns:
        top_N_predicted_keys = (
            df.sort("log_halflife_pred", descending=True).head(int(len(df) * N / 100))["key"].to_list()
        )
        top_N_predicted_key_set = set(top_N_predicted_keys)

        random_keys = df.sample(fraction=N / 100, with_replacement=False)["key"].to_list()
        random_key_set = set(random_keys)

        prob = round(len((actual_key_set & top_N_predicted_key_set)) / len(top_N_predicted_key_set), 2)
        random_prob = round(len((actual_key_set & random_key_set)) / len(random_key_set), 2)

        probs.append(prob)
        random_probs.append(random_prob)

    plt.figure(figsize=(9, 7))
    plt.ylim(0, 1)
    plt.plot(Ns, probs, label="Use predicted value")
    plt.plot(Ns, random_probs, label="Random")
    plt.xlabel("$N$", fontsize=16, labelpad=10)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.legend()

    dir = Helper.ROOT / "output" / "figures" / "hl" / code / version
    dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(dir / "prediction_true_ratio_fixed.png")


if __name__ == "__main__":
    hl_prediction_true_ratio_fixed(code="EXP2-2", version="version_0")
