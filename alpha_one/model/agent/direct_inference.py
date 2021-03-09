from typing import Union

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent.base import Agent
import pyspiel
import numpy as np


class DirectInferenceAgent(Agent):

    def __init__(self, model, n_previous_observations=1):
        super(DirectInferenceAgent, self).__init__(is_information_set_agent=n_previous_observations > 1)
        self._model = model
        self._n_previous_observations = n_previous_observations

    def next_move(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> (int, np.array):
        if self.is_information_set_agent():
            information_set_generator = state_or_information_set_generator
            observations = information_set_generator.get_padded_observation_history(self._n_previous_observations)
            legal_actions_mask = information_set_generator.get_legal_actions_mask()
        else:
            state = state_or_information_set_generator
            observations = state.observation_tensor()
            legal_actions_mask = state.legal_actions_mask()

        _, policy = self._model.inference([observations], [legal_actions_mask])
        policy = policy[0]  # NN returns batch of predictions
        policy[~np.array(legal_actions_mask, dtype=np.bool)] = 0

        action = np.random.choice(len(policy), p=policy)

        return action, policy

    def evaluate(self, state: pyspiel.State) -> float:
        value, _ = self._model.inference([state.observation_tensor()], [state.legal_actions_mask()])
        return value[0]
