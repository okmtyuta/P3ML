from typing import Optional, Union

import h5py
import torch

from src.modules.protein.types import ProteinProps


class Protein:
    def __init__(self, key: str, props: ProteinProps, representations: Optional[torch.Tensor] = None) -> None:
        self._key = key
        self._props = props
        self._representations = representations

    @property
    def seq(self) -> str:
        value = self._props.get("seq")
        if not isinstance(value, str):
            raise TypeError("Protein.props['seq'] must be a str")
        return value

    @property
    def representations(self) -> str:
        representations = self._representations
        if representations is None:
            raise RuntimeError("Protein representations unavailable")
        return representations

    def read_props(self, name: str) -> Union[str, int, float]:
        prop = self._props.get(name)
        if prop is None:
            raise RuntimeError(f"Prop {name} is not readable")

        return prop

    def set_representations(self, representations: torch.Tensor) -> "Protein":
        self._representations = representations
        return self

    def to_hdf5_group(self, group: h5py.Group):
        group.attrs["key"] = self._key

        props_group = group.create_group(name="props")
        for key, value in self._props.items():
            props_group.attrs[key] = value

        group.create_dataset(name="representations", data=self.representations)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group):
        key = group.attrs["key"]
        props = {key: value for key, value in group["props"].attrs.items()}
        representations = torch.from_numpy(group["representations"][:])

        return Protein(key=key, props=props, representations=representations)
