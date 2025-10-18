from pathlib import Path

from src.main.extraction import ExtractionRunnerConfig

cfg = ExtractionRunnerConfig(
    csv_path=Path("data/schwanhausser/logarithm.csv"),
    output_path=Path("source/schwanhausser/esm2/logarithm.h5"),
    protein_language_name="esm2",
    batch_size=1,
)

if __name__ == "__main__":
    cfg.run()
