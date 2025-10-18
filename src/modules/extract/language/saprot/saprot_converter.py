import os
from typing import TypedDict

import torch

from src.modules.extract.language.saprot.esm_loader import load_esm_saprot


class SaProtModelResult(TypedDict):
    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor


class SaProtConverter:
    def __init__(self):
        self._model, self._alphabet = self._get_model_and_alphabet()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model.eval()

    def __call__(self, seqs: list[str]) -> list[torch.Tensor]:
        batch_tokens = self._batch_converter([(seq, seq) for seq in seqs])[2]
        batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results: SaProtModelResult = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False,
            )
        token_representations: torch.Tensor = results["representations"][33]

        sequence_representations: list[torch.Tensor] = []
        for i, tokens_len in enumerate(batch_lens):
            representation = token_representations[i, 1 : tokens_len - 1]
            sequence_representations.append(representation)  # noqa: E203
        return sequence_representations

    def _get_model_and_alphabet(self):
        model_path = os.path.join("src", "modules", "extract", "language", "saprot", "SaProt_650M_AF2.pt")
        return load_esm_saprot(model_path)
