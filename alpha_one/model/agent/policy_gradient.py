import numpy as np
from open_spiel.python.algorithms.policy_gradient import PolicyGradient

from alpha_one.model.agent.base import Agent
import pyspiel


class PolicyGradientAgent(Agent):

    def __init__(self, policy_gradient_model: PolicyGradient):
        super(PolicyGradientAgent, self).__init__(is_information_set_agent=False)
        self.policy_gradient_model = policy_gradient_model

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        return self.policy_gradient_model._act(state.observation_tensor(), state.legal_actions())
