from pathlib import Path

import polars as pl

DATA_PATH = Path("data/meier/_data.csv")


def main() -> None:
    df = pl.read_csv(DATA_PATH)

    clean_seq = pl.col("seq").str.replace_all("_", "").str.replace_all(r"\((?:ac|ox)\)", "")

    df = df.with_columns(
        [
            clean_seq.alias("seq"),
            clean_seq.str.len_chars().alias("length"),
        ]
    )

    df.write_csv(Path("data/meier/__data.csv"))


if __name__ == "__main__":
    main()
