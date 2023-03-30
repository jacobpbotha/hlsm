from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from hlsm.lgp.abcd.repr.task_repr import TaskRepr
from hlsm.lgp.abcd.task import Task


class TaskReprFunction(nn.Module, ABC):
    """
    Function that builds a task-conditioned state representation
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, task: Task) -> TaskRepr:
        ...
