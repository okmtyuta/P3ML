import torch

from src.modules.helper.helper import Helper
from src.modules.model.regressor import Regressor
from src.modules.extract.language.saprot.saprot_converter import SaProtConverter
import shap
from transformers import EsmTokenizer
import numpy as np
import matplotlib.pyplot as plt
import json
import polars as pl





@torch.no_grad()
def create_predict(regressor: Regressor):
    def predict(seq: str):
        converter = SaProtConverter()
        representations = converter(seqs=[seq])[0]
        feature = representations.mean(dim=0)

        y = regressor(feature).squeeze(-1).detach().numpy()

        return y
    
    return predict

regressor = Regressor(input_dim=1280, output_dim=1, hidden_dim=32, hidden_num=5)
predict = create_predict(regressor=regressor)
predict('Lv')

# def hl_prediction_shap(code: str, version: str, seq: str):
#     regressor = Regressor(input_dim=1280, output_dim=1, hidden_dim=32, hidden_num=5)
#     regressor.load_state_dict(torch.load(Helper.ROOT / "logs" / code / version / "weight.pt"))

#     tokenizer = EsmTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
#     predict = create_predict(regressor)

#     explainer = shap.Explainer(predict, tokenizer)
#     sv = explainer([seq])

#     tokens = np.array(sv.data[0], dtype=object)
#     values = np.array(sv.values[0], dtype=float)
#     base_value = float(np.ravel(sv.base_values)[0])

#     mask = (tokens != "") & (tokens != None)
#     tokens = np.array([t.strip() for t in tokens[mask]], dtype=object)
#     values = values[mask]

#     result = {
#         "tokens": tokens.tolist(),
#         "shap_values": values.tolist(),
#         "base_value": base_value,
#         "predicted_value": base_value + float(values.sum()),
#     }

#     return result

# result = hl_prediction_shap(code='EXP2', version='version_6', seq='Lv')
# print(result)
