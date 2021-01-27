from alpha_one.model.agent.base import Agent
import pyspiel
import numpy as np


class DirectInferenceAgent(Agent):

    def __init__(self, model):
        self.model = model

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        _, policy = self.model.inference([state.observation_tensor()], [state.legal_actions_mask()])
        policy = policy[0]  # NN returns batch of predictions
        action = np.random.choice(len(policy), p=policy)
        return action, policy

    def evaluate(self, state: pyspiel.State) -> float:
        value, _ = self.model.inference([state.observation_tensor()], [state.legal_actions_mask()])
        return value[0]
