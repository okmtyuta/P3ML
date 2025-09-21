from src.modules.extract.language.pwesm._pwesm import _PWESMLanguage


class PWESM2Language(_PWESMLanguage):
    name = "pwesm2"

    def __init__(self):
        super().__init__("esm2")
