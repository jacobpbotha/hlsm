import itertools
import random
from typing import Dict

import torch

import hlsm.lgp.env.blockworld.config as config
from hlsm.lgp.abcd.agent import Agent
from hlsm.lgp.abcd.repr.state_repr import StateRepr
from hlsm.lgp.env.alfred.alfred_action import ACTION_TYPES
from hlsm.lgp.env.alfred.alfred_action import AlfredAction
from hlsm.lgp.env.alfred.alfred_observation import AlfredObservation
from hlsm.lgp.env.alfred.tasks import AlfredTask
from hlsm.lgp.models.alfred.handcoded_skills.init_skill import InitSkill


class DemonstrationReplayAgent(Agent):
    def __init__(self):
        super().__init__()
        self.actions = None
        self.init_skill = InitSkill()
        self.initialized = False
        self.current_step = 0

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def start_new_rollout(self, task: AlfredTask, state_repr: StateRepr = None):
        api_ish_actions = task.traj_data.get_api_action_sequence()
        self.actions = [AlfredAction(a["action"], torch.from_numpy(a["mask"]) if a["mask"] is not None else None) for a in api_ish_actions]
        self.current_step = 0
        self.initialized = False
        self.init_skill.start_new_rollout()

    def act(self, observation: AlfredObservation) -> AlfredAction:
        # First run the init skill until it stops
        if not self.initialized:
            action = self.init_skill.act(...)
            if action.is_stop():
                self.initialized = True
            else:
                return action

        # Then execute the prerecorded action sequence
        if self.current_step < len(self.actions):
            action = self.actions[self.current_step]
        else:
            action = AlfredAction("Stop", None)

        self.current_step += 1
        return action
