from abc import ABC, abstractmethod
from typing import Iterable

import torch
from lgp.abcd.action import Action


class ActionRepr(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def is_stop(self) -> bool:
        ...

    @classmethod
    def from_action(cls, action: Action, device) -> "ActionRepr":
        ...

    @classmethod
    @abstractmethod
    def collate(cls, actions: Iterable["ActionRepr"]):
        """Creates a single ActionRepresentation that represents a batch of
        actions."""
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...
