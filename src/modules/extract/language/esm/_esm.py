from src.modules.extract.language._language import _Language
from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.extract.language.esm.esm_types import ESMModelName
from src.modules.protein.protein import Protein


class _ESMLanguage(_Language):
    def __init__(self, model_name: ESMModelName):
        super().__init__()
        self._converter = ESMConverter(model_name=model_name)

    def __call__(self, proteins: list[Protein]) -> list[Protein]:
        self._set_representations(proteins=proteins)
        return proteins

    def _set_representations(self, proteins: list[Protein]) -> list[Protein]:
        seqs = [protein.seq for protein in proteins]

        sequence_representations = self._converter(seqs=seqs)

        for i, protein in enumerate(proteins):
            representations = sequence_representations[i]
            protein.set_representations(representations=representations)

        return proteins
