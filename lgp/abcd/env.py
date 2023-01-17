from typing import Tuple, Dict
from abc import ABC, abstractmethod

from hlsm.lgp.abcd.action import Action
from hlsm.lgp.abcd.observation import Observation
from hlsm.lgp.abcd.task import Task


class Env(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def reset(self) -> (Observation, Task):
        ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        ...