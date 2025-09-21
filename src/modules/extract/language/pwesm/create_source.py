from pathlib import Path

import h5py
from tqdm import tqdm

from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.extract.language.source import amino_acid_source_mapping


def main():
    esm2_converter = ESMConverter("esm2")
    esm1b_converter = ESMConverter("esm1b")

    path = Path(__file__).parent / "pwesm.h5"
    with h5py.File(name=path, mode="w") as f:
        for source in tqdm(amino_acid_source_mapping.values()):
            group = f.create_group(source["char"])

            attrs = group.attrs
            attrs["name"] = source["name"]
            attrs["char"] = source["char"]

            esm1b = esm1b_converter([source["char"]])[0].squeeze(dim=0)
            group.create_dataset(name="esm1b", data=esm1b, dtype="float32")

            esm2 = esm2_converter([source["char"]])[0].squeeze(dim=0)
            group.create_dataset(name="esm2", data=esm2, dtype="float32")


if __name__ == "__main__":
    main()
