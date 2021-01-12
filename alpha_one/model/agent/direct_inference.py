from alpha_one.model.agent.base import Agent
import pyspiel
import numpy as np


class DirectInferenceAgent(Agent):

    def __init__(self, model):
        self.model = model

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        action, policy = self.model.inference([state.observation_tensor()], [state.legal_actions_mask()])
        return action[0], policy[0]
