from src.modules.extract.language.pwesm._pwesm import _PWESMLanguage


class PWESM1bLanguage(_PWESMLanguage):
    name = "pwesm1b"

    def __init__(self):
        super().__init__("esm1b")
