import json

import polars as pl

from src.modules.helper.helper import Helper


def create_note_hl_prediction(code: str):
    summary_csv = Helper.ROOT / "logs" / code / "summary.csv"
    df = pl.read_csv(summary_csv)
    mean_pearsonr = df.select(pl.col("pearsonr").mean()).item()
    result = df.with_columns((pl.col("pearsonr") - mean_pearsonr).abs().alias("abs_diff")).sort("abs_diff").row(0)

    note = {"basic_version": result[0]}

    with open(Helper.ROOT / "logs" / code / "note.json", mode="w") as f:
        f.write(json.dumps(note))


if __name__ == "__main__":
    for code in ["EXP1", "EXP2", "EXP3", "EXP4"]:
        create_note_hl_prediction(code=code)
