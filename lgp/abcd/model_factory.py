from typing import Dict

from hlsm.lgp.abcd.functions.observation_function import ObservationFunction
from hlsm.lgp.abcd.functions.task_repr_function import TaskReprFunction
from hlsm.lgp.abcd.skill import Skill


class ModelFactory:
    def __init__(self):
        ...

    def get_skillset(self) -> Dict[str, Skill]:
        ...

    def get_observation_function(self) -> ObservationFunction:
        ...

    def get_task_repr_function(self) -> TaskReprFunction:
        ...
