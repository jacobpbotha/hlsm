from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from hlsm.lgp.abcd.action import Action
from hlsm.lgp.abcd.observation import Observation
from hlsm.lgp.abcd.repr.action_repr import ActionRepr


class ActionReprFunction(nn.Module, ABC):
    """
    Function that builds an action representation conditioned on the corresponding observation
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, action: Action, observation: Observation) -> ActionRepr:
        ...
