import abc

from src.modules.protein.protein import Protein


class _Language(metaclass=abc.ABCMeta):
    name = "_language"

    def __call__(self, proteins: list[Protein]) -> list[Protein]:
        raise NotImplementedError()
