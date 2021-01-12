import numpy as np
import pyspiel

from alpha_one.model.agent.base import Agent


class RandomAgent(Agent):

    def __init__(self, game):
        self.game = game

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        policy = np.zeros(self.game.num_distinct_actions())
        policy[state.legal_actions()] = 1
        policy /= policy.sum()

        return np.random.choice(len(policy), p=policy), policy
