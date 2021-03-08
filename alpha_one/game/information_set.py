from collections import defaultdict
from typing import List

import numpy as np
import pyspiel
from open_spiel.python.observation import make_observation

PUBLIC_OBSERVER_PLAYER_ID = 999


class InformationSetGenerator:

    def __init__(self, game: pyspiel.Game):
        self.game = game
        self.action_history = defaultdict(list)
        self.observation_history = defaultdict(list)
        self._observation_buffer = defaultdict(list)
        self.previous_information_set = dict()

        if hasattr(game, 'make_observer') or hasattr(game, 'make_py_observer'):
            self._public_observer = make_observation(
                game,
                pyspiel.IIGObservationType(
                    perfect_recall=False,
                    public_info=True,
                    private_info=pyspiel.PrivateInfoType.NONE))
        else:
            self._public_observer = None

        initial_state = game.new_initial_state()
        self._current_player = initial_state.current_player()
        try:
            # Some games don't have observation tensor implemented. In this case we use the observation string instead
            initial_state.observation_tensor(0)
            self._game_has_tensor_observation = True
            # self._get_observation = lambda state, player_id: state.observation_tensor(player_id)
        except pyspiel.SpielError:
            self._game_has_tensor_observation = True
            # self._get_observation = lambda state, player_id: state.observation_string(player_id)

    def _get_observation(self, state, player_id):
        if player_id == PUBLIC_OBSERVER_PLAYER_ID:
            assert self._public_observer is not None, "Cannot get public observation on games that don't support this"
            self._public_observer.set_from(state, 0)
            return np.copy(self._public_observer.tensor)
        else:
            if self._game_has_tensor_observation:
                return state.observation_tensor(player_id)
            else:
                return state.observation_string(player_id)

    def _observations_equal(self, observation_1, observation_2):
        if isinstance(observation_1, np.ndarray) and isinstance(observation_2, np.ndarray):
            return all(observation_1 == observation_2)
        else:
            return observation_1 == observation_2

    def clone(self):
        # TODO: test
        clone = InformationSetGenerator(self.game)
        clone.action_history = defaultdict(list)
        clone.observation_history = defaultdict(list)
        clone._observation_buffer = defaultdict(list)
        for player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}:
            if player_id in self.action_history:
                clone.action_history[player_id] = [action for action in self.action_history[player_id]]
            if player_id in self._observation_buffer:
                clone._observation_buffer[player_id] = [
                    np.array(observation) if isinstance(observation, np.ndarray) else observation
                    for observation in
                    self._observation_buffer[player_id]]

            clone.observation_history[player_id] = [
                list(observation) for observation in self.observation_history[player_id]
            ]
        clone.previous_information_set = {
            player_id: [state.clone() for state in self.previous_information_set[player_id]]
            for player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}
            if player_id in self.previous_information_set}
        clone._current_player = self._current_player

        return clone

    def register_action(self, action: int, player_id: int = None):
        if player_id is None:
            player_id = self.current_player()
        self.action_history[player_id].append(action)

    def register_observation(self, state: pyspiel.State):
        observation_player_1 = self._get_observation(state, 0)
        observation_player_2 = self._get_observation(state, 1)

        self.observation_history[0].append(observation_player_1)
        self._observation_buffer[0].append(observation_player_1)

        self.observation_history[1].append(observation_player_2)
        self._observation_buffer[1].append(observation_player_2)

        self.calculate_information_set(0)
        self.calculate_information_set(1)

        if self._public_observer is not None:
            public_observation = self._get_observation(state, PUBLIC_OBSERVER_PLAYER_ID)
            self._observation_buffer[PUBLIC_OBSERVER_PLAYER_ID].append(public_observation)
            self.observation_history[PUBLIC_OBSERVER_PLAYER_ID].append(public_observation)
            self.calculate_information_set(PUBLIC_OBSERVER_PLAYER_ID)

        self._current_player = state.current_player()

    def register_chance_player_action(self, action: int):
        """
        Models a 'what-if' scenario. This method can be used INSTEAD of the regular register() method to update the
        information sets of the corresponding players to reflect what states the players could be in if the chance
        player played the given action now. This is used in settings where no true game state is available, e.g., in
        the simulations of IIG-MCTS.
        Note that NO observation will be added to the observation history as the observation would depend on the true
        state (which is not available)

        Parameters
        ----------
        action: the action that the chance player would play
        """

        assert self.current_player() == -1, f"Can only register chance player action when it is chance player's turn (not player {self.current_player()})"
        for player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}:
            information_set = []
            for state in self.previous_information_set[player_id]:
                if action in state.legal_actions():
                    state.apply_action(action)
                    information_set.append(state)
            self.previous_information_set[player_id] = information_set
            assert all([state.current_player() == information_set[0].current_player() for state in
                        information_set]), f"All nodes in information set have to belong to same player!, {[(str(s), s.current_player()) for s in information_set]}"
            assert len(self.previous_information_set[
                           player_id]) > 0, f"Information Set is empty after registering chance node action {action}. Illegal action?"
        self._current_player = information_set[0].current_player()

    def register(self, state: pyspiel.State, action: int):
        self.register_action(action)
        self.register_observation(state)

    def current_player(self):
        return self._current_player

    def get_observation_history(self, player_id: int = None) -> List[np.ndarray]:
        player_id = self.current_player() if player_id is None else player_id
        assert player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}, f"Invalid player id {player_id}"
        return self.observation_history[player_id]

    def get_padded_observation_history(self, n_previous_observations: int, player_id: int = None, ):
        observation_history = self.get_observation_history(player_id=player_id)[-n_previous_observations:]
        observation_padding = n_previous_observations - len(observation_history)
        padded_history = [0 for _ in range(self.game.observation_tensor_shape()[0] * observation_padding)]
        for observation in observation_history:
            padded_history.extend(observation)
        return padded_history

    def get_legal_actions_mask(self, player_id: int = None):
        if player_id is None:
            player_id = self.current_player()

        if player_id in self.previous_information_set:
            return self.previous_information_set[player_id][0].legal_actions_mask(player_id)
        else:
            return self.game.new_initial_state().legal_actions_mask(player_id)

    def calculate_information_set(self, player_id: int = None) -> List[pyspiel.State]:
        if player_id is None:
            player_id = self.current_player()

        assert player_id in {0, 1, PUBLIC_OBSERVER_PLAYER_ID}, f"Invalid player id {player_id}"

        if player_id not in self.previous_information_set:
            previous_information_set = [self.game.new_initial_state()]
        else:
            previous_information_set = self.previous_information_set[player_id]
        if len(self.action_history[player_id]) == 0 and len(self._observation_buffer[player_id]) == 0:
            # We don't have any new information
            return previous_information_set
        information_set = self._calculate_information_set(previous_information_set,
                                                          player_id,
                                                          self._observation_buffer[player_id],
                                                          self.action_history[player_id])

        assert information_set[0].current_player() < 0 or \
               all([s.legal_actions_mask() == information_set[0].legal_actions_mask() for s in information_set]), \
            f"All states in information set have to have the same legal actions mask!"

        self.previous_information_set[player_id] = information_set
        self.action_history[player_id] = []
        self._observation_buffer[player_id] = []
        return information_set

    def get_public_information_set(self):
        assert self._public_observer is not None, "Cannot get public information set on games that don't support this"
        if PUBLIC_OBSERVER_PLAYER_ID not in self.previous_information_set:
            return [self.game.new_initial_state()]
        else:
            return self.previous_information_set[PUBLIC_OBSERVER_PLAYER_ID]

    def get_plausible_states(self, player_id, reference_state):
        states = []
        for s in self.get_public_information_set():
            if self._observations_equal(
                    self._get_observation(s, player_id),
                    self._get_observation(reference_state, player_id)):
                states.append(s.clone())
        return states

    def _calculate_information_set(self,
                                   information_set: List[pyspiel.State],
                                   player_id: int,
                                   observation_history: List,
                                   action_history: List[int]) -> pyspiel.State:
        states = []
        assert all([state.current_player() == information_set[0].current_player() for state in
                    information_set]), f"All nodes in information set have to belong to same player!, {[(str(s), s.current_player()) for s in information_set]}"
        if information_set[0].current_player() == player_id:
            for game_state in information_set:
                game_state.apply_action(action_history[0])
                if self._observations_equal(self._get_observation(game_state, player_id), observation_history[0]):
                    states.append(game_state)
            action_history = action_history[1:]
            observation_history = observation_history[1:]
        else:
            for game_state in information_set:
                for action in game_state.legal_actions():
                    state_clone = game_state.clone()
                    state_clone.apply_action(action)
                    if self._observations_equal(self._get_observation(state_clone, player_id), observation_history[0]):
                        states.append(state_clone)
            observation_history = observation_history[1:]

        assert len(states) > 0, "There cannot be 0 states"
        if len(observation_history) > 0 or len(action_history) > 0:
            return self._calculate_information_set(states, player_id, observation_history, action_history)
        else:
            return states

    def to_str(self):
        return str({player_id: [(str(s), s.current_player()) for s in states] for player_id, states in
                    self.previous_information_set.items()})

    def __str__(self):
        return self.to_str()
