from typing import Union

import numpy as np
import pyspiel

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent.base import Agent


class RandomAgent(Agent):

    def __init__(self, game):
        super(RandomAgent, self).__init__(is_information_set_agent=False)
        self.game = game

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        policy = np.zeros(self.game.num_distinct_actions())
        policy[state.legal_actions()] = 1
        policy /= policy.sum()

        return np.random.choice(len(policy), p=policy), policy

    def evaluate(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> float:
        return 0


