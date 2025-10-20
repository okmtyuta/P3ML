import numpy as np
import polars as pl
from scipy import stats

from src.modules.helper.helper import Helper


def aggregate_hl_prediction(code: str):
    experiments = Helper.ROOT / "logs" / code
    results = list(experiments.glob("**/test_results.csv"))
    dfs = [(result.parent.name, pl.read_csv(result)) for result in results]

    data = []
    for version, df in dfs:
        y = df["log_halflife"].to_numpy()
        y_hat = df["log_halflife_pred"].to_numpy()

        pearsonr = stats.pearsonr(y, y_hat).correlation

        error = y - y_hat
        abs_error = np.abs(error)
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(abs_error)
        delta95 = np.percentile(abs_error, 95)

        datum = {"version": version, "pearsonr": pearsonr, "rmse": rmse, "mae": mae, "delta95": delta95}
        data.append(datum)

    pl.from_dicts(data).sort("version").write_csv(Helper.ROOT / "logs" / code / "summary.csv")
