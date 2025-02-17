from abc import ABC
from abc import abstractmethod
from typing import Iterable

import torch


class StateRepr(ABC):

    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, states: Iterable["StateRepr"]) -> "StateRepr":
        """
        Creates a single StateRepresentation that represents a batch of states
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...

    def cast(self, cls, device="cpu"):
        raise NotImplementedError(f"Casting of {type(self)} not implemented")
