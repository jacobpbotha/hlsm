from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union

from hlsm.lgp.abcd.dataset import ExtensibleDataset
from hlsm.lgp.env.alfred.alfred_observation import AlfredObservation
from hlsm.lgp.rollout.rollout_data import load_rollout_from_path


class PerceptionDataset(ExtensibleDataset):

    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths

    @abstractmethod
    def __getitem__(self, item):
        example = load_rollout_from_path(self.chunk_paths[item])
        example = self._process_example(example)
        return example

    @abstractmethod
    def __len__(self):
        return len(self.chunk_paths)

    def _process_example(self, example):
        observation = example["observation"]

        observation.data_augment()

        example_out = {
            "observation": observation
        }
        return example_out

    # Inherited from hlsm.lgp.abcd.dataset.ExtensibleDataset
    def collate_fn(self, list_of_examples: Union[List[Dict], List[List[Dict]]]) -> Dict:
        list_of_examples = [l for l in list_of_examples if l is not None]

        observations = [l["observation"] for l in list_of_examples]
        observations = AlfredObservation.collate(observations)

        out = {
            "observations": observations
        }
        return out
