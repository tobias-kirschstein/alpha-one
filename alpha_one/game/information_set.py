from collections import defaultdict
from typing import List

import pyspiel


class InformationSetGenerator:

    def __init__(self, game: pyspiel.Game):
        self.game = game
        self.action_history = defaultdict(list)
        self.observation_history = defaultdict(list)
        self.previous_information_set = dict()

        # Some games don't have observation tensor implemented. In this case we use the observation string instead
        try:
            game.new_initial_state().observation_tensor(0)
            self._get_observation = lambda state, player_id: state.observation_tensor(player_id)
        except pyspiel.SpielError:
            self._get_observation = lambda state, player_id: state.observation_string(player_id)

    def register_action(self, player_id: int, action: int):
        self.action_history[player_id].append(action)

    def register_observation(self, state: pyspiel.State):
        self.observation_history[0].append(self._get_observation(state, 0))
        self.observation_history[1].append(self._get_observation(state, 1))
        self.calculate_information_set(0)
        self.calculate_information_set(1)

    def calculate_information_set(self, player_id: int) -> List[pyspiel.State]:
        if player_id not in self.previous_information_set:
            previous_information_set = [self.game.new_initial_state()]
        else:
            previous_information_set = self.previous_information_set[player_id]
        if len(self.action_history[player_id]) == 0 and len(self.observation_history[player_id]) == 0:
            # We don't have any new information
            return previous_information_set
        information_set = self._calculate_information_set(previous_information_set,
                                                          player_id,
                                                          self.observation_history[player_id],
                                                          self.action_history[player_id])
        self.previous_information_set[player_id] = information_set
        self.action_history[player_id] = []
        self.observation_history[player_id] = []
        return information_set

    def _calculate_information_set(self,
                                   information_set: List[pyspiel.State],
                                   player_id: int,
                                   observation_history: List,
                                   action_history: List[int]) -> pyspiel.State:
        states = []
        assert all([state.current_player() == information_set[0].current_player() for state in
                    information_set]), "All nodes in information set have to belong to same player!"
        if information_set[0].current_player() == player_id:
            for game_state in information_set:
                game_state.apply_action(action_history[0])
                if self._get_observation(game_state, player_id) == observation_history[0]:
                    states.append(game_state)
            action_history = action_history[1:]
            observation_history = observation_history[1:]
        else:
            for game_state in information_set:
                for action in game_state.legal_actions():
                    state_clone = game_state.clone()
                    state_clone.apply_action(action)
                    if self._get_observation(state_clone, player_id) == observation_history[0]:
                        states.append(state_clone)
            observation_history = observation_history[1:]

        assert len(states) > 0, "There cannot be 0 states"
        if len(observation_history) > 0 or len(action_history) > 0:
            return self._calculate_information_set(states, player_id, observation_history, action_history)
        else:
            return states
