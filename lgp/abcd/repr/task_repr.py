from abc import ABC
from abc import abstractmethod
from typing import Iterable

import torch

from hlsm.lgp.abcd.task import Task


class TaskRepr(ABC):

    def __init__(self):
        ...

    @classmethod
    def from_task(cls, task: Task, device) -> "TaskRepr":
        ...

    @abstractmethod
    def as_tensor(self):
        ...

    @classmethod
    @abstractmethod
    def collate(cls, task_reprs: Iterable["TaskRepr"]):
        """
        Creates a single TaskRepresentation that represents a batch of tasks (e.g natural language instructions)
        """
        ...

    @abstractmethod
    def represent_as_image(self) -> torch.tensor:
        ...
