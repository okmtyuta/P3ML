import json

import polars as pl

from src.modules.helper.helper import Helper


def maxmin_scatter_hl_prediction(code: str):
    with open(Helper.ROOT / "logs" / code / "note.json", mode="r") as f:
        note = json.load(f)
        basic_version = note["basic_version"]

    df = pl.read_csv(Helper.ROOT / "logs" / code / basic_version / "test_results.csv")

    max_row = df.sort("log_halflife").row(index=0, named=True)
    min_row = df.sort("log_halflife", descending=True).row(index=0, named=True)

    return {"max_index": max_row["key"], "min_index": min_row["key"]}


if __name__ == "__main__":
    for code in ['EXP1-1', 'EXP2-1', 'EXP3-1', 'EXP4-1']:
        result = maxmin_scatter_hl_prediction(code)
        print(code, result)
