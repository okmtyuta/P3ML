import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import scipy.stats as stat
from matplotlib.axes import Axes


class Visualizer:
    @classmethod
    def save_scatter(self, input_path: Path, output_path: Path, prop_name: str) -> None:
        df = pl.read_csv(input_path)

        x: list[float] = df[prop_name].to_list()
        y: list[float] = df[f"{prop_name}_pred"].to_list()

        xy_min = min(x + y)
        xy_max = max(x + y)
        pad = 0.05 * (xy_max - xy_min if xy_max > xy_min else 1.0)

        fig = plt.figure(dpi=100, figsize=(8, 8))
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
        ax: Axes = fig.add_subplot(1, 1, 1)

        charges = sorted(df["charge"].unique().to_list())
        cmap = plt.get_cmap("tab10")
        for i, charge in enumerate(charges):
            sub = df.filter(pl.col("charge") == charge)
            xi = sub[prop_name].to_list()
            yi = sub[f"{prop_name}_pred"].to_list()
            ax.scatter(xi, yi, color=cmap(i % 10), s=4, alpha=0.7, label=f"charge={charge}")

        ax.legend(loc="best", fontsize=9)

        correlation = stat.pearsonr(x, y)[0]
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)) / len(x))

        ax.set_title(f"Observed vs Predicted {prop_name.capitalize()} (Pearson R={correlation:.4f}, RMSE={rmse:.2f})")

        ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], color="black", linewidth=1)

        ax.set_xlabel(f"Observed {prop_name.capitalize()}")
        ax.set_ylabel(f"Predicted {prop_name.capitalize()}")

        plt.savefig(output_path)
        plt.close(fig)

    # @classmethod
    # def save_training_metrics_curves(self, result_dir: Path, output_path: Path) -> None:
    #     files = {
    #         "train": result_dir / "train_ccs.csv",
    #         "validate": result_dir / "validate_ccs.csv",
    #         "evaluate": result_dir / "evaluate_ccs.csv",
    #     }

    #     series: dict[str, dict[str, list[float]]] = {}
    #     for name, path in files.items():
    #         df = pl.read_csv(path)
    #         series[name] = {
    #             "epoch": df["epoch"].to_list(),
    #             "rmse": df["root_mean_squared_error"].to_list(),
    #             "pearsonr": df["pearsonr"].to_list(),
    #         }

    #     fig = plt.figure(dpi=100, figsize=(12, 5))

    #     fig.subplots_adjust(left=0.12, right=0.70, bottom=0.12, top=0.9)
    #     ax_rmse: Axes = fig.add_subplot(1, 1, 1)
    #     ax_r: Axes = ax_rmse.twinx()

    #     colors = {"train": "C0", "validate": "C1", "evaluate": "C2"}

    #     for name, data in series.items():
    #         ax_rmse.plot(
    #             data["epoch"],
    #             data["rmse"],
    #             label=f"RMSE/{name}",
    #             color=colors[name],
    #             linestyle="-",
    #             linewidth=1.8,
    #         )
    #     ax_rmse.set_xlabel("Epoch")
    #     ax_rmse.set_ylabel("RMSE")
    #     ax_rmse.grid(True, linestyle=":", alpha=0.5)

    #     for name, data in series.items():
    #         ax_r.plot(
    #             data["epoch"],
    #             data["pearsonr"],
    #             label=f"r/{name}",
    #             color=colors[name],
    #             linestyle="--",
    #             linewidth=1.8,
    #         )
    #     ax_r.set_ylabel("Pearson r")

    #     handles_l, labels_l = ax_rmse.get_legend_handles_labels()
    #     handles_r, labels_r = ax_r.get_legend_handles_labels()
    #     ax_rmse.legend(
    #         handles_l + handles_r,
    #         labels_l + labels_r,
    #         loc="upper left",
    #         bbox_to_anchor=(1.15, 1.00),
    #         fontsize=9,
    #         frameon=True,
    #         borderaxespad=0.0,
    #     )

    #     plt.savefig(output_path)
    #     plt.close(fig)
