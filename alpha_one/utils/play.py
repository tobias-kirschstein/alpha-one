from typing import Union

import pyspiel

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.game.observer import OmniscientObserver


class InteractiveGameMachine:

    def __init__(self, game: Union[str, pyspiel.Game]):
        self.game = pyspiel.load_game(game) if isinstance(game, str) else game
        self.state = None
        self.information_set_generator = None
        self.omniscient_observer = OmniscientObserver(self.game)

    def new_game(self):
        self.state = self.game.new_initial_state()
        self.information_set_generator = InformationSetGenerator(self.game)

    def get_state(self):
        return self.state

    def list_player_actions(self):
        print(f"Current Player: {self.state.current_player()}")
        print(f"Legal actions: ")
        for action in self.state.legal_actions():
            print(f"  - {action}: {self.state.action_to_string(self.state.current_player(), action)}")

        return self.state.legal_actions()

    def get_observations(self):
        player_observation = None
        if self.state.current_player() >= 0:
            print(f"Player {self.state.current_player()} observation: "
                  f"{self.state.observation_string(self.state.current_player())}")
            player_observation = self.state.observation_tensor(self.state.current_player())
        print(f"Omniscient Observation: {self.omniscient_observer.get_observation_string(self.state)}")
        omniscient_observation = self.omniscient_observer.get_observation_tensor(self.state)
        return player_observation, omniscient_observation

    def await_action(self):
        human_input = input()
        action = int(human_input)
        self.state.apply_action(action)
        self.information_set_generator.register(self.state, action)

    def finish_game(self):
        if self.state.is_terminal():
            print(f"Returns: {self.state.returns()}")
            return True
        else:
            return False