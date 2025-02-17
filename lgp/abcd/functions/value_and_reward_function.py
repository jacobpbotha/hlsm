from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from hlsm.lgp.abcd.repr.action_repr import ActionRepr
from hlsm.lgp.abcd.repr.state_repr import StateRepr
from hlsm.lgp.abcd.repr.task_repr import TaskRepr


class ValueAndRewardFunction(nn.Module):
    """
    Given a current state s_t, action a_t, next state s_(t+1), predict reward r(s_t, a_t | L) and value V(s_(t+1) | L)
    conditioned on the task.
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, task_repr: TaskRepr, state: StateRepr, action: ActionRepr, next_state: StateRepr) -> Tuple[torch.tensor, torch.tensor]:
        ...
