import platform
from pathlib import Path

from src.main.experiments.abstract.predict_onehot_ccs_1 import predict_onehot_ccs_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList
from src.modules.slack_service import SlackService

code = Path(__file__).stem
proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "ishihama" / "onehot" / "data.h5").proteins


def main() -> None:
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] training started: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    for _ in range(1):
        predict_onehot_ccs_1(code, input_props=["charge", "mass", "length"], output_props=["ccs"], proteins=proteins)

    try:
        slack_service.send(f"[{server_name}] training end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}: {code}")


if __name__ == "__main__":
    main()
