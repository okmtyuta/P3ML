from pathlib import Path

from src.main.extraction import ExtractionRunnerConfig

cfg = ExtractionRunnerConfig(
    csv_path=Path("data/schwanhausser/saprot_logarithm.csv"),
    output_path=Path("source/schwanhausser/saprot/saprot_logarithm.h5"),
    protein_language_name="saprot",
    batch_size=1,
)

if __name__ == "__main__":
    cfg.run()
