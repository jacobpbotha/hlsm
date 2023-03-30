from abc import abstractmethod

import torch.nn as nn

from hlsm.lgp.abcd.repr.action_repr import ActionRepr
from hlsm.lgp.abcd.repr.state_repr import StateRepr


class DynamicsFunction(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, prev_state: StateRepr, action_repr: ActionRepr) -> StateRepr:
        ...
