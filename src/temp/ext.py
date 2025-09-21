import os
from pathlib import Path

from src.main.extraction import ExtractionRunnerConfig

DIR = "source/meier"

os.makedirs(DIR, exist_ok=True)


# cfg = ExtractionRunnerConfig(
#     csv_path=Path("data/ishihama/data.csv"),
#     output_path=Path(f"{DIR}/esm2.h5"),
#     protein_language_name="esm2",
#     batch_size=32,
# )
# cfg.run()

# cfg = ExtractionRunnerConfig(
#     csv_path=Path("data/ishihama/data.csv"),
#     output_path=Path(f"{DIR}/pwesm1b.h5"),
#     protein_language_name="pwesm1b",
#     batch_size=32,
# )
# cfg.run()

cfg = ExtractionRunnerConfig(
    csv_path=Path("data/meier/data.csv"),
    output_path=Path(f"{DIR}/onehot.h5"),
    protein_language_name="onehot",
    batch_size=32,
)
cfg.run()
