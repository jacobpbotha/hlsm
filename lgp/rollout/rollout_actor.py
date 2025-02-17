import ray
import torch

from hlsm.lgp import paths
from hlsm.lgp.abcd.agent import TrainableAgent
from hlsm.lgp.env.alfred.alfred_action import AlfredAction
from hlsm.lgp.env.alfred.alfred_observation import AlfredObservation


class RolloutActorLocal:
    def __init__(
        self,
        experiment_name: str,
        agent: TrainableAgent,
        env,
        env2,
        dataset_proc,
        param_server_proc,
        max_horizon,
        dataset_device,
        index,
        collect_trace=False,
        lightweight_mode=False,
    ) -> None:
        self.dataset_process = dataset_proc
        self.param_server_proc = param_server_proc
        self.actor_index = index
        if self.actor_index == 0:
            from hlsm.lgp.utils.better_summary_writer import BetterSummaryWriter

            self.writer = BetterSummaryWriter(f"{paths.get_experiment_runs_dir(experiment_name)}-rollout", start_iter=0)
        else:
            self.writer = None

        self.agent = agent
        self.env = env
        self.env2 = env2
        self.horizon = max_horizon
        self.env.set_horizon(max_horizon)
        self.counter = 0

        self.collect_trace = collect_trace  # Whether to eval outputs of agent.get_trace in the rollout
        self.lightweight_mode = (
            lightweight_mode  # Whether to produce stripped-down rollouts with task and metadata only
        )

        self.dataset_device = dataset_device

    def _load_agent_state_from_ps(self):
        for model in self.agent.get_learnable_models():
            model.load_state_dict(ray.get(self.param_server_proc.get.remote(model.get_name())))

    def rollout_and_send_forever(self):
        while True:
            self.rollout_and_send()

    def split_rollout(self, skip_tasks=None, max_section=20, ret=None):
        rollout = []
        with torch.no_grad():
            if ret is None:
                observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)
                # Skipped:
                if task is None:
                    return None, None, True

                # print("Task: ", str(task))
                self.agent.start_new_rollout(task)
                action = self.agent.act(observation)
                start = 0
            else:
                observation = ret["observation"]
                action = ret["action"]
                task = ret["task"]
                rollout_idx = ret["rollout_idx"]
                start = ret["t"]

            total_reward = 0
            for t in range(start, self.horizon):
                next_observation, reward, done, md = self.env.step(action)
                total_reward += reward

                rollout.append(
                    {
                        "task": task,
                        "observation": None if self.lightweight_mode else (observation.to(self.dataset_device)),
                        "action": None if self.lightweight_mode else action,
                        "reward": reward,
                        "return": total_reward,
                        "agent_trace": self.agent.get_trace(device=self.dataset_device)
                        if (self.collect_trace and not self.lightweight_mode)
                        else None,
                        "done": done,
                        "md": md,
                    },
                )
                self.agent.clear_trace()

                observation = next_observation

                if done:
                    self.agent.finalize(total_reward)
                    rollout.append(
                        {
                            "task": task,
                            "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                            "action": None,
                            "agent_trace": None,
                            "reward": 0,
                            "return": total_reward,
                            "done": True,
                            "md": md,  # TODO: This gets added twice, which might be confusing
                        },
                    )
                    new_ret = None
                    break
                else:
                    action = self.agent.act(observation)

                if t - start > max_section:
                    new_ret = {
                        "t": t,
                        "task": task,
                        "rollout_idx": rollout_idx,
                        "observation": observation,
                        "action": action,
                    }
                    break
                else:
                    new_ret = None

            if new_ret is not None:
                # print(f"Pause rollout: {self.counter}, length: {len(rollout)}")
                return rollout, new_ret, False
            else:
                # print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                self.counter += 1
                return rollout, new_ret, True

    def _explore_via_teleport(self) -> AlfredObservation:
        event = self.env.thor_env.step({"action": "GetReachablePositions"})
        list_of_pos = event.metadata["actionReturn"]
        x0, y0, z0 = event.metadata["agent"]["position"].values()
        rotation = event.metadata["agent"]["rotation"]
        horizon = event.metadata["agent"]["cameraHorizon"]
        for pos in list_of_pos[::2]:
            x, y, z = pos.values()
            action = AlfredAction(action_type="Teleport", argument_mask=AlfredAction.get_empty_argument_mask())
            action.set_teleport_coords(x, z, rotation, horizon)
            observation, reward, done, md = self.env.step(action)
            self.agent.update_state(observation)
            self.agent.clear_trace()
            # print(md)
            # print(pos)
            # print(sr.get_agent_pos_m())
            # print(sr.get_pos_xyz_vx())
            # print(sr.get_origin_xyz_vx())

            action = AlfredAction(action_type="RotateLeft", argument_mask=AlfredAction.get_empty_argument_mask())
            observation, reward, done, md = self.env.step(action)
            self.agent.update_state(observation)
            self.agent.clear_trace()

            observation, reward, done, md = self.env.step(action)
            self.agent.update_state(observation)
            self.agent.clear_trace()

            observation, reward, done, md = self.env.step(action)
            self.agent.update_state(observation)
            self.agent.clear_trace()

            observation, reward, done, md = self.env.step(action)
            self.agent.update_state(observation)
            self.agent.clear_trace()

        action = AlfredAction(action_type="Teleport", argument_mask=AlfredAction.get_empty_argument_mask())
        action.set_teleport_coords(x0, z0, rotation, horizon)
        observation, reward, done, md = self.env.step(action)
        return observation

    def rollout(self, skip_tasks=None):
        rollout = []
        with torch.no_grad():
            observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)

            # Skipped:
            if task is None:
                return None
            observation = self._explore_via_teleport()
            # sg_obs = self.env2.reset(self.env.task.get_task_id()[6:])

            # print("Task: ", str(task))
            self.agent.start_new_rollout(task)
            # translate = {
            #     "PickupObject": "Pickup",
            #     "CloseObject": "Close",
            #     "OpenObject": "Open",
            #     "PutObject": "Put",
            #     "ToggleObjectOn": "ToggleOn",
            #     "ToggleObjectOff": "ToggleOff",
            #     "SliceObject": "Slice",
            #     "Stop": "Done",
            # }

            action = self.agent.act(observation)
            total_reward = 0
            # found = False
            for _t in range(self.horizon):
                next_observation, reward, done, md = self.env.step(action)

                # if self.agent.current_skill and not found and not self.agent.current_skill.found:
                #     # Find the target I need to navigate to
                #     self.agent.current_goal.arg_str()
                #     self.agent.current_goal.type_str()
                #     pass
                # if action.action_type in action.get_interact_action_list() and md["action_success"]:
                #     if translate[action.action_type] == "Put":
                #         target = md["api_action"]["receptacleObjectId"]
                #         if target not in self.env2.scene_graph.graph["Nearby"]:
                #             action_go = f"Go__{target}"
                #             sg_obs, r2, d2 = self.env2.step(sg_obs.node_keys.index(action_go))
                #     else:
                #         target = md["api_action"]["objectId"]
                #         if target not in self.env2.scene_graph.graph["Nearby"]:
                #             action_go = f"Go__{target}"
                #             sg_obs, r2, d2 = self.env2.step(sg_obs.node_keys.index(action_go))
                #     action2 = f"{translate[action.action_type]}__{target}"
                #     sg_obs, r2, d2 = self.env2.step(sg_obs.node_keys.index(action2))
                total_reward += reward

                rollout.append(
                    {
                        "task": task,
                        "observation": None if self.lightweight_mode else (observation.to(self.dataset_device)),
                        "action": None if self.lightweight_mode else action,
                        "reward": reward,
                        "return": total_reward,
                        "agent_trace": self.agent.get_trace(device=self.dataset_device)
                        if (self.collect_trace and not self.lightweight_mode)
                        else None,
                        "done": done,
                        "md": md,
                    },
                )
                self.agent.clear_trace()

                observation = next_observation

                if done:
                    self.agent.finalize(total_reward)
                    rollout.append(
                        {
                            "task": task,
                            "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                            "action": None,
                            "agent_trace": None,
                            "reward": 0,
                            "return": total_reward,
                            "done": True,
                            "md": md,  # TODO: This gets added twice, which might be confusing
                        },
                    )
                    break
                else:
                    action = self.agent.act(observation)
                    event = self.env.thor_env.last_event
                    sr = self.agent.state_repr
                    print(md)
                    print(event.metadata["cameraPosition"])
                    print(sr.get_agent_pos_m())
                    print(sr.get_pos_xyz_vx())

            # print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
            self.counter += 1
            return rollout

    def rollout_and_send(self):
        self._load_agent_state_from_ps()
        rollout = self.rollout()

        # Send to the dataset process
        self.dataset_process.add_rollout.remote(rollout)

        # Write metrics to tensorboard
        # if self.writer is not None:
        return


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutActor(RolloutActorLocal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

