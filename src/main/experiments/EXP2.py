from pathlib import Path

from src.main.analysis.aggregate_hl_prediction import aggregate_hl_prediction
from src.main.experiments.abstract.predict_hl_1 import predict_hl_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList

code = Path(__file__).stem

output_props: list[str] = ["log_halflife"]
input_props: list[str] = []


proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "schwanhausser" / "saprot" / "saprot_logarithm.h5").proteins


def main() -> None:
    for _ in range(1):
        predict_hl_1(code, input_props=input_props, output_props=output_props, proteins=proteins)

    aggregate_hl_prediction(code=code)


if __name__ == "__main__":
    main()
