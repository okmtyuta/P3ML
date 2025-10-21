import torch

from src.modules.protein.protein_list import ProteinList
from src.modules.helper.helper import Helper
from src.modules.model.regressor import Regressor
from src.modules.extract.language.saprot.saprot_converter import SaProtConverter
import shap
from transformers import EsmTokenizer
import numpy as np
import matplotlib.pyplot as plt
import json
import polars as pl

converter = SaProtConverter()
tokenizer = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
regressor = Regressor(input_dim=1280, output_dim=1, hidden_dim=32, hidden_num=5)

seqs = [
    "MdSdDpTdApVdAdDpTpRdRdLdNdSdKdPdQdDdLpTcDrAvYpGpPdPdSqNwFdLkEaIkDaIwFdNdPwQdTwVdGdVdGdRlApRiFfTiTkYtEwViRwMmRaTiNpLdPpIlFaKpLdKrEiSdCiVeRiRdRgYpSvDlFvEvWvLlKqNvElLcEvRvDpShKpIdVpVqPdPdLaPdGdKpAqLpKvRlQsLdPpFdRdGpDdEnGrIcFpEpEpSvFnIvEvEsRnRsQvGrLvEnQvFrIlNvKvIqAcGpHdPpLrAsQsNqEwRpCsLnHnMcFsLhQhEpEhAdIdDdRpNpYdVdPtGdKtVpLpGdLpHpWvLvLpSpMdRd"  # 1726
]


def collate_fn(seqs: list[str]):
    xs = converter(seqs=seqs)
    ips = [torch.tensor([], dtype=torch.float32) for _ in xs]

    L = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax, A = int(L.max()), xs[0].shape[1]

    X = torch.zeros(len(xs), Lmax, A, dtype=torch.float32)
    for i, x in enumerate(xs):
        X[i, : x.shape[0]] = x.to(torch.float32)
    Ip = torch.stack([torch.as_tensor(ip, dtype=torch.float32) for ip in ips])

    return X, Ip, L


def create_model(regressor: Regressor):
    @torch.no_grad()
    def model(seqs: list[str]):
        X, Ip, L = collate_fn(seqs=seqs)

        y = regressor(X, Ip, L)

        return y

    return model


def explain(regressor: Regressor, seqs: list[str]):
    model = create_model(regressor=regressor)
    explainer = shap.Explainer(model=model, masker=tokenizer)

    shapv = explainer(seqs)

    tokens = np.array(shapv.data[0], dtype=object)
    values = np.array(shapv.values[0], dtype=float)
    base_value = float(np.ravel(shapv.base_values)[0])

    mask = (tokens != "") & (tokens != None)
    tokens = np.array([t.strip() for t in tokens[mask]], dtype=object)
    values = values[mask]

    result = {
        "tokens": tokens.tolist(),
        "shap_values": values.tolist(),
        "base_value": base_value,
        "predicted_value": base_value + float(values.sum()),
    }

    return result


def hl_prediction_shap(code: str, dataset_name: str, key: str):
    with open(Helper.ROOT / "logs" / code / "note.json", mode="r") as f:
        note = json.load(f)
        basic_version = note["basic_version"]

    regressor = Regressor(input_dim=1280, output_dim=1, hidden_dim=32, hidden_num=5)
    regressor.load_state_dict(torch.load(Helper.ROOT / "logs" / code / basic_version / "weight.pt"))

    result = explain(regressor=regressor, seqs=seqs)

    dir = Helper.ROOT / "output" / "shap" / "hl" / code / f"basic_version_{basic_version}"
    dir.mkdir(parents=True, exist_ok=True)

    with open(dir / f"shapv_{dataset_name}_{key}.json", mode="w") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    hl_prediction_shap(code="EXP2", dataset_name="schwanhausser", key="580")
