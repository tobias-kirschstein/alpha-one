from typing import Union
import pyspiel

import numpy as np
from open_spiel.python.policy import TabularPolicy

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent.base import Agent


class CFRAgent(Agent):

    def __init__(self, cfr_policy_table: TabularPolicy, temperature=1.0):
        super(CFRAgent, self).__init__(is_information_set_agent=False)
        self._cfr_policy_table = cfr_policy_table
        self._temperature = temperature

    def next_move(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> (
    int, np.array):
        action_probabilities_dict = self._cfr_policy_table.action_probabilities(state_or_information_set_generator)
        policy = np.zeros(len(state_or_information_set_generator.legal_actions_mask()))
        for action, prob in action_probabilities_dict.items():
            policy[action] = prob
        if self._temperature is None or self._temperature == 0:
            new_policy = np.zeros(len(state_or_information_set_generator.legal_actions_mask()))
            new_policy[np.argmax(policy)] = 1
            policy = new_policy
        else:
            policy = policy ** (1 / self._temperature)

        action = np.random.choice(len(policy), p=policy)
        return action, policy

    def evaluate(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> float:
        raise NotImplementedError()
