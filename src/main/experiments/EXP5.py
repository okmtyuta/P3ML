import json
import platform
from pathlib import Path

from src.main.experiments.abstract.transfer_predict_hl_1 import transfer_predict_hl_1
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList
from src.modules.slack_service import SlackService

code = Path(__file__).stem
proteins = ProteinList.from_hdf5(Helper.ROOT / "source" / "zecha" / "esm2" / "logarithm.h5").proteins


def main() -> None:
    server_name = platform.node()
    slack_service = SlackService()

    try:
        slack_service.send(f"[{server_name}] transfer prediction start: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")

    result = transfer_predict_hl_1(
        code=code, from_code="EXP1", from_version="version_3", input_props=[], output_props=["log_halflife"], proteins=proteins
    )
    result_dir = Helper.ROOT / "logs" / code
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "result.json", mode="w") as f:
        f.write(json.dumps(result[0]))

    try:
        slack_service.send(f"[{server_name}] transfer prediction end: {code}")
    except Exception as e:
        print(f"Slack notification was failed because of {e}")


if __name__ == "__main__":
    main()
