from abc import ABC
from abc import abstractmethod

import torch
import torch.nn as nn

from hlsm.lgp.abcd.repr.state_repr import StateRepr
from hlsm.lgp.abcd.repr.task_repr import TaskRepr


class ValueFunction(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, state_repr: StateRepr, task_repr: TaskRepr) -> torch.tensor:
        ...
