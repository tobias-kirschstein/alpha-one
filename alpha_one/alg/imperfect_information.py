from abc import abstractmethod
from typing import List, Tuple

import pyspiel
from open_spiel.python.algorithms.mcts import Evaluator

from alpha_one.game.information_set import InformationSetGenerator


class ImperfectInformationMCTSEvaluator(Evaluator):
    """
    This is an extension of the regular Evaluator used for MCTS. Regular MCTS only allows to heuristically evaluate
    a state node with evaluate() and the outgoing actions with prior().
    This extension accounts for nodes in a search tree where the agent has to guess among states in his information
    set. These guess nodes are treated the same way as normal game tree nodes, i.e., the agent may also give a heuristic
    for the guess node with evaluate_observation_node() and the outgoing possible states in the information set with
    prior_observation_node()
    """

    @abstractmethod
    def prior_observation_node(self, information_set_generator: InformationSetGenerator) \
            -> List[Tuple[pyspiel.State, float]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_observation_node(self, information_set_generator: InformationSetGenerator) -> (float, float):
        raise NotImplementedError


class BasicImperfectInformationMCTSEvaluator(ImperfectInformationMCTSEvaluator):
    """
    Very stupid heuristic that just evaluates all nodes with 0 and yields a uniform distribution over possible choices.
    """

    def prior_observation_node(self, information_set_generator: InformationSetGenerator) \
            -> List[Tuple[pyspiel.State, float]]:
        information_set = information_set_generator.calculate_information_set()
        return [(state, 1.0 / len(information_set)) for state in information_set]

    def evaluate_observation_node(self, information_set_generator: InformationSetGenerator) -> (float, float):
        return [0, 0]

    def evaluate(self, state):
        return [0, 0]

    def prior(self, state):
        """Returns equal probability for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]
