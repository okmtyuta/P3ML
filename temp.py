from pathlib import Path

from src.main.exp.EXP0004 import EXP0004
from src.main.exp.EXP0003 import EXP0003
from src.modules.visualizer.visualizer import Visualizer

if __name__ == "__main__":
    EXP0004()
    # EXP0003()
    # Visualizer.save_scatter(
    #     Path("logs/EXP0002/version_0/test_results.csv"),
    #     output_path=Path("logs/EXP0002/version_0/scatter.png"),
    #     prop_name="ccs",
    # )
    # Visualizer.save_scatter(
    #     Path("logs/EXP0003/version_0/test_results.csv"),
    #     output_path=Path("logs/EXP0003/version_0/scatter.png"),
    #     prop_name="ccs",
    # )
