from abc import ABC
from abc import abstractmethod

import torch.nn as nn

from hlsm.lgp.abcd.repr.action_distribution import ActionDistribution
from hlsm.lgp.abcd.repr.state_repr import StateRepr


class ActionProposalGivenState(nn.Module, ABC):
    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, state: StateRepr) -> ActionDistribution:
        ...
