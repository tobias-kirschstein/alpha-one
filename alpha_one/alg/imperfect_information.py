from abc import abstractmethod
from typing import List, Tuple

import pyspiel
import numpy as np
from open_spiel.python.algorithms.mcts import Evaluator

from alpha_one.utils.statemask import get_state_mask

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

class AlphaOneImperfectInformationMCTSEvaluator(ImperfectInformationMCTSEvaluator):
    
    def __init__(self, state_to_value, observation_model, game_model):

        self._observation_model = observation_model
    
        self._game_model = game_model
    
        self._state_to_value = state_to_value
    
    def prior_observation_node(self, information_set_generator: InformationSetGenerator) \
            -> List[Tuple[pyspiel.State, float]]:

        
        information_set = information_set_generator.calculate_information_set()
        state_mask, index_track = get_state_mask(self._state_to_value, information_set)
        
        obs = np.expand_dims(state_mask, 0)
        mask = np.expand_dims(state_mask, 0)
        
        _, policy = self._observation_model.inference(obs, mask)
        
        policy = policy[0]
        
        prior = []
        for i in range(len(information_set)):
            prior.append((information_set[i], policy[index_track[i]]))
            
        return prior
        

    def evaluate_observation_node(self, information_set_generator: InformationSetGenerator) -> (float, float):

        if information_set_generator.current_player() == -1:
    	    return [0, 0]

        
        information_set = information_set_generator.calculate_information_set()
        state_mask, _ = get_state_mask(self._state_to_value, information_set)
        
        obs = np.expand_dims(state_mask, 0)
        mask = np.expand_dims(state_mask, 0)
        
        value, _ = self._observation_model.inference(obs, mask)
        value = value[0, 0]

        return [value, -value]

    def evaluate(self, state):

        if state.is_chance_node():
            return [0, 0]
        
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        
        value, _ = self._game_model.inference(obs, mask)
        
        value = value[0, 0]

        return [value, -value]
        
        

    def prior(self, state):
        
        if state.is_chance_node():
            return state.chance_outcomes()
        
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        
        _, policy = self._game_model.inference(obs, mask)
        
        policy = policy[0]
        
        return [(action, policy[action]) for action in state.legal_actions()]


class DeterminizedMCTSEvaluator(Evaluator):
    
    def __init__(self, model):

        self._model = model


    def evaluate(self, state):

        if state.is_chance_node():
            return [0, 0]
        
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        
        value, _ = self._model.inference(obs, mask)
        
        value = value[0, 0]

        return [value, -value]
        
        

    def prior(self, state):

        if state.is_chance_node():
            return state.chance_outcomes()

        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        
        _, policy = self._model.inference(obs, mask)
        
        policy = policy[0]
        
        return [(action, policy[action]) for action in state.legal_actions()]



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
