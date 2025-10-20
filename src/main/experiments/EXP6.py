import platform
from pathlib import Path

from src.main.experiments.abstract.transfer_predict_hl_1 import transfer_predict_hl_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList
from src.modules.slack_service import SlackService

code = Path(__file__).stem
proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "zecha" / "saprot" / "saprot_logarithm.h5").proteins


def main() -> None:
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] transfer prediction start: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    transfer_predict_hl_1(
        code="EXP2", version="version_0", input_props=[], output_props=["log_halflife"], proteins=proteins
    )

    try:
        slack_service.send(f"[{server_name}] transfer prediction end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")


if __name__ == "__main__":
    main()
