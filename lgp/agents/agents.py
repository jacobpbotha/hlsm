import os

import torch

import hlsm.lgp.paths
from hlsm.lgp.parameters import Hyperparams


def build_alfred_hierarchical_agent(agent_setup, hparams, device):
    # Import agents (hierarchichal, high-level, and low-level)
    from hlsm.lgp.agents.hierarchical_agent import HierarchicalAgent
    from hlsm.lgp.agents.action_proposal_agent import ActionProposalAgent
    # Import model factory
    from hlsm.lgp.models.alfred.hlsm.hlsm_model_factory import HlsmModelFactory
    # Import classes
    from hlsm.lgp.env.alfred.alfred_action import AlfredAction
    from hlsm.lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskRepr

    model_factory = HlsmModelFactory(hparams)
    skillset = model_factory.get_skillset()
    obsfunc = model_factory.get_observation_function()
    actprop = model_factory.get_subgoal_model()

    subgoal_model_path = hlsm.lgp.paths.get_subgoal_model_path()
    if subgoal_model_path:
        sd = torch.load(subgoal_model_path)
        actprop.load_state_dict(sd, strict=False)

    actprop = actprop.to(device)

    obsfunc.eval()
    actprop.eval()

    highlevel_agent = ActionProposalAgent(actprop, obsfunc, HlsmTaskRepr, device)
    hierarchical_agent = HierarchicalAgent(highlevel_agent, skillset, obsfunc, AlfredAction)
    return hierarchical_agent


def build_alfred_deviant_agent(agent_setup, hparams, device):
    deviance_p = hparams.get("agent_setup").get("deviance")
    from hlsm.lgp.agents.deviant_agent import DeviantAgent
    from hlsm.lgp.agents.alfred.demonstration_replay_agent import DemonstrationReplayAgent
    from hlsm.lgp.agents.alfred.random_valid_agent import RandomValidAgent
    agent = DeviantAgent(oracle_agent=DemonstrationReplayAgent(),
                         random_agent=RandomValidAgent(),
                         deviance_prob=deviance_p)
    return agent


def build_demo_replay_agent(agent_setup, hparams, device):
    from hlsm.lgp.agents.alfred.demonstration_replay_agent import DemonstrationReplayAgent
    agent = DemonstrationReplayAgent()
    return agent


def build_alfred_random_agent(*args, **kwargs):
    from hlsm.lgp.agents.alfred.random_valid_agent import RandomValidAgent
    return RandomValidAgent()


AGENT_BUILDERS = {
    "alfred_random_agent": build_alfred_random_agent,
    "build_alfred_hierarchical_agent": build_alfred_hierarchical_agent,
    "alfred_deviant_agent": build_alfred_deviant_agent,
    "alfred_demo_replay_agent": build_demo_replay_agent
}


def get_agent(setup: Hyperparams, hparams: Hyperparams, device=None):
    agent_type = setup.agent_type
    agent_setup = setup.agent_setup
    if device is None:
        device = setup.device

    if agent_type not in AGENT_BUILDERS:
        raise ValueError(f"Unrecognized agent type: {agent_type}")

    return AGENT_BUILDERS[agent_type](agent_setup, hparams, device)
