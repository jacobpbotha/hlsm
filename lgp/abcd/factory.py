from typing import Dict
from abc import abstractmethod

from hlsm.lgp.abcd.agent import Agent
from hlsm.lgp.abcd.model_factory import ModelFactory
from hlsm.lgp.abcd.env import Env

from hlsm.lgp.parameters import Hyperparams


class Factory:

    def __init__(self):
        ...

    @abstractmethod
    def get_model_factory(self, setup: Dict, hparams : Hyperparams) -> ModelFactory:
        ...

    @abstractmethod
    def get_environment(self, setup: Dict) -> Env:
        ...

    @abstractmethod
    def get_agent(self, setup: Hyperparams, hparams: Hyperparams) -> Agent:
        ...
