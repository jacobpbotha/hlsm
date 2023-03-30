from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from hlsm.lgp.abcd.action import Action
from hlsm.lgp.abcd.observation import Observation
from hlsm.lgp.abcd.repr.action_repr import ActionRepr


class ActionInverseReprFunction(nn.Module, ABC):
    """
    Function that builds a task-conditioned state representation
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, action_repr: ActionRepr, observation: Observation) -> Action:
        ...
