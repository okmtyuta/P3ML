from pathlib import Path
import platform

from src.modules.slack_service import SlackService
from src.main.experiments.abstract.predict_hl_1 import predict_hl_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList

code = Path(__file__).stem

output_props: list[str] = ["log_halflife"]
input_props: list[str] = []


proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "zecha" / "esm2" / "logarithm.h5").proteins


def main() -> None:
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] training started: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    for _ in range(10):
        predict_hl_1(code, input_props=input_props, output_props=output_props, proteins=proteins)

    try:
        slack_service.send(f"[{server_name}] training end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}: {code}")


if __name__ == "__main__":
    main()
