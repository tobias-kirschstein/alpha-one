from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pyspiel
from open_spiel.python.algorithms.mcts import Evaluator
from open_spiel.python.utils import lru_cache

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.game.observer import OmniscientObserver
from alpha_one.utils.statemask import get_state_mask


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

    def __init__(self, game, state_to_value, observation_model, game_model, n_previous_observations=1):

        self._observation_model = observation_model
        self._game_model = game_model
        self._state_to_value = state_to_value
        self._observer = OmniscientObserver(game)
        self.n_previous_observations = n_previous_observations

    def prior_observation_node(self, information_set_generator: InformationSetGenerator) \
            -> List[Tuple[pyspiel.State, float]]:

        if information_set_generator.current_player() == -1:
            information_set = information_set_generator.calculate_information_set()
            return [(state, 1.0 / len(information_set)) for state in information_set]

        obs = [information_set_generator.get_padded_observation_history(self.n_previous_observations)]

        information_set = information_set_generator.calculate_information_set()
        state_mask, index_track = get_state_mask(self._state_to_value, information_set)

        # obs = np.expand_dims(obs, 0)
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

        obs = [information_set_generator.get_padded_observation_history(self.n_previous_observations)]

        information_set = information_set_generator.calculate_information_set()
        state_mask, _ = get_state_mask(self._state_to_value, information_set)

        # obs = np.expand_dims(obs, 0)
        mask = np.expand_dims(state_mask, 0)

        value, _ = self._observation_model.inference(obs, mask)
        value = value[0, 0]

        return [value, -value]

    def evaluate(self, state):

        if state is None:
            return [0, 0]

        elif state.is_chance_node():
            return [0, 0]

        # add total information of the state not just private observation after guessing state
        obs = np.expand_dims(self._observer.get_observation_tensor(state), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)

        value, _ = self._game_model.inference(obs, mask)

        value = value[0, 0]

        return [value, -value]

    def prior(self, state):

        if state is None:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

        elif state.is_chance_node():
            return state.chance_outcomes()

        obs = np.expand_dims(self._observer.get_observation_tensor(state), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)

        _, policy = self._game_model.inference(obs, mask)

        policy = policy[0]

        return [(action, policy[action]) for action in state.legal_actions()]


class DeterminizedMCTSEvaluator(Evaluator):

    def __init__(self, model, game):

        self._model = model
        self._observer = OmniscientObserver(game)

    def evaluate(self, state):

        if state.is_chance_node():
            return [0, 0]

        obs = np.expand_dims(self._observer.get_observation_tensor(state), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)

        value, _ = self._model.inference(obs, mask)

        value = value[0, 0]

        return [value, -value]

    def prior(self, state):

        if state.is_chance_node():
            return state.chance_outcomes()

        obs = np.expand_dims(self._observer.get_observation_tensor(state), 0)
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


class BasicOmniscientMCTSEvaluator(Evaluator):

    def __init__(self, game):
        self._observer = OmniscientObserver(game)

    def evaluate(self, state):
        return np.array([0, 0])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


class AlphaZeroOmniscientMCTSEvaluator(Evaluator):

    def __init__(self, game, model, cache_size=2 ** 16):
        """An AlphaZero MCTS Evaluator."""
        if game.num_players() != 2:
            raise ValueError("Game must be for two players.")
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")
        # if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        #     raise ValueError("Game must be deterministic.")

        self._model = model
        self._cache = lru_cache.LRUCache(cache_size)
        self._observer = OmniscientObserver(game)

    def _inference(self, state):
        # Make a singleton batch
        obs = np.expand_dims(self._observer.get_observation_tensor(state), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)

        # ndarray isn't hashable
        cache_key = obs.tobytes() + mask.tobytes()

        value, policy = self._cache.make(
            cache_key, lambda: self._model.inference(obs, mask))

        return value[0, 0], policy[0]  # Unpack batch

    def evaluate(self, state):
        """Returns a value for the given state."""
        if state.is_chance_node():
            return [0, 0]

        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        """Returns the probabilities for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()

        _, policy = self._inference(state)
        return [(action, policy[action]) for action in state.legal_actions()]
