from pathlib import Path

from src.main.extraction import ExtractionRunnerConfig

cfg = ExtractionRunnerConfig(
    csv_path=Path("data/ishihama/data.csv"),
    output_path=Path("source/ishihama/onehot/data.h5"),
    protein_language_name='onehot',
    batch_size=32,
)

if __name__ == "__main__":
    cfg.run()
