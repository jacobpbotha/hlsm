import random
from typing import Dict

from hlsm.lgp.abcd.action import Action
from hlsm.lgp.abcd.agent import Agent
from hlsm.lgp.abcd.observation import Observation
from hlsm.lgp.abcd.repr.state_repr import StateRepr
from hlsm.lgp.abcd.task import Task


class DeviantAgent(Agent):

    def __init__(self,
                 oracle_agent: Agent,
                 random_agent: Agent,
                 deviance_prob: float):
        """
        :param agents: A list of agents to create a mixture policy
        :param agent_probs: For
        """
        super().__init__()
        self.oracle_agent = oracle_agent
        self.random_agent = random_agent
        self.deviance_prob = deviance_prob
        self.has_deviated = False

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        self.oracle_agent.start_new_rollout(task, state_repr)
        self.random_agent.start_new_rollout(task, state_repr)
        self.has_deviated = False

    def act(self, observation: Observation) -> Action:
        self.has_deviated = self.has_deviated or random.random() < self.deviance_prob
        #print(f"Using {'random_agent' if self.has_deviated else 'oracle_agent'}")
        agent = self.random_agent if self.has_deviated else self.oracle_agent
        action = agent.act(observation)
        return action
