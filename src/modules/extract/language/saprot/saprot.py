from src.modules.extract.language._language import _Language
from src.modules.extract.language.saprot.saprot_converter import SaProtConverter
from src.modules.protein.protein import Protein


class SaProtLanguage(_Language):
    name = "saprot"

    def __init__(self):
        super().__init__()
        self._converter = SaProtConverter()

    def __call__(self, proteins: list[Protein]):
        self._set_representations(proteins=proteins)
        return proteins

    def _set_representations(self, proteins: list[Protein]) -> list[Protein]:
        seqs = [protein.seq for protein in proteins]

        sequence_representations = self._converter(seqs=seqs)

        for i, protein in enumerate(proteins):
            representations = sequence_representations[i]
            protein.set_representations(representations=representations)

        return proteins
