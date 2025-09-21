
from src.modules.extract.language._language import _Language
from src.modules.protein.protein import Protein


class Extractor:
    def __init__(self, language: _Language):
        self._language = language

    def __call__(self, proteins: list[Protein], batch_size: int) -> list[Protein]:
        for index in range(0, len(proteins), batch_size):
            batch = proteins[index : index + batch_size]
            self._language(proteins=batch)

        return proteins
