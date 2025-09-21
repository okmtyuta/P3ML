from src.modules.extract.language._language import _Language
from src.modules.extract.language.onehot.onehot_converter import OnehotConverter
from src.modules.protein.protein import Protein


class OnehotLanguage(_Language):
    def __init__(self):
        super().__init__()
        self._encoder = OnehotConverter()

    def __call__(self, proteins: list[Protein]) -> list[Protein]:
        self._set_representations(proteins=proteins)
        return proteins

    def _set_representations(self, proteins: list[Protein]) -> list[Protein]:
        seqs = [protein.seq for protein in proteins]

        sequence_representations = self._encoder(seqs=seqs)

        for i, protein in enumerate(proteins):
            representations = sequence_representations[i]
            protein.set_representations(representations=representations)

        return proteins
