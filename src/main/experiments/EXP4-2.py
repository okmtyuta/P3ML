import platform
from pathlib import Path

from src.main.experiments.abstract.predict_hl_1 import predict_hl_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList
from src.modules.slack_service import SlackService

code = Path(__file__).stem
proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "zecha" / "saprot" / "saprot_logarithm.h5").proteins


def main() -> None:
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] training started: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    predict_hl_1(code, input_props=[], output_props=["log_halflife"], proteins=proteins, random_split_seed=2236477679)

    try:
        slack_service.send(f"[{server_name}] training end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}: {code}")


if __name__ == "__main__":
    main()
