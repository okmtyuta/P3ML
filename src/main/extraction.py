from dataclasses import dataclass
from pathlib import Path

from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language.esm.esm1b import ESM1bLanguage
from src.modules.extract.language.esm.esm2 import ESM2Language
from src.modules.extract.language.onehot.onehot_language import OnehotLanguage
from src.modules.extract.language.pwesm.pwesm1b import PWESM1bLanguage
from src.modules.extract.language.pwesm.pwesm2 import PWESM2Language
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.types import ProteinLanguageName


@dataclass
class ExtractionRunnerConfig:
    csv_path: Path
    output_path: Path
    protein_language_name: ProteinLanguageName
    batch_size: int = 32

    def _create_language(self):
        if self.protein_language_name == "esm2":
            return ESM2Language()
        elif self.protein_language_name == "esm1b":
            return ESM1bLanguage()
        elif self.protein_language_name == "pwesm1b":
            return PWESM1bLanguage()
        elif self.protein_language_name == "pwesm2":
            return PWESM2Language()
        elif self.protein_language_name == "onehot":
            return OnehotLanguage()
        else:
            raise ValueError(f"Unsupported language model: {self.protein_language_name}")

    def run(self) -> ProteinList:
        print(f"Loading protein data from: {self.csv_path}")
        protein_list = ProteinList.from_csv(path=self.csv_path)
        proteins = protein_list.proteins
        print(f"Loaded {len(protein_list.proteins)} proteins")

        print(f"Initializing language model: {self.protein_language_name}")
        language = self._create_language()
        extractor = Extractor(language=language)

        print(f"Running extraction with batch_size={self.batch_size}")

        extractor(proteins=proteins, batch_size=self.batch_size)

        print(f"Saving results to: {self.output_path}")
        protein_list.to_hdf5(self.output_path)
        print(f"ExtractionRunner completed successfully for {len(proteins)} proteins")

        return protein_list
