from typing import Union

import pyspiel
import numpy as np

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.game.observer import OmniscientObserver
from alpha_one.game.trajectory import GameTrajectory


class GameMachine:

    def __init__(self, game: Union[str, pyspiel.Game]):
        self.game = pyspiel.load_game(game) if isinstance(game, str) else game
        self.state = None
        self.information_set_generator = None
        self._trajectory = None
        self.omniscient_observer = OmniscientObserver(self.game)

    def new_game(self):
        self._trajectory = GameTrajectory(self.game)
        self.state = self.game.new_initial_state()
        self.information_set_generator = InformationSetGenerator(self.game)
        self._handle_chance_player()

    def _handle_chance_player(self):
        while self.state.current_player() == -1:
            actions, probs = zip(*self.state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            self.state.apply_action(action)
            self.information_set_generator.register(self.state, action)

    def get_state(self):
        return self.state

    def get_information_set_generator(self):
        return self.information_set_generator

    def get_rewards(self):
        assert self.is_finished(), f"Game has to be finished!"
        return self.state.returns()

    def get_trajectory(self):
        return self._trajectory

    def current_player(self):
        return self.state.current_player()

    def list_player_actions(self):
        return self.state.legal_actions()

    def get_player_observation(self):
        player_observation = None
        if self.state.current_player() >= 0:
            player_observation = self.state.observation_tensor(self.state.current_player())
        return player_observation

    def get_omniscient_observation(self):
        omniscient_observation = self.omniscient_observer.get_observation_tensor(self.state)
        return omniscient_observation

    def get_observations(self):
        player_observation = None
        if self.state.current_player() >= 0:
            player_observation = self.state.observation_tensor(self.state.current_player())
        omniscient_observation = self.omniscient_observer.get_observation_tensor(self.state)
        return player_observation, omniscient_observation

    def play_action(self, action: int):
        self._trajectory.append(self.state, action, None)
        self.state.apply_action(action)
        self.information_set_generator.register(self.state, action)

        self._handle_chance_player()
        if self.is_finished():
            self._trajectory.set_final_rewards(self.get_rewards())

    def is_finished(self):
        return self.state.is_terminal()


class VerboseGameMachine:

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

    def get_information_set_generator(self):
        return self.information_set_generator

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

    def play_action(self, action: int):
        self.state.apply_action(action)
        self.information_set_generator.register(self.state, action)

    def finish_game(self):
        if self.state.is_terminal():
            print(f"Returns: {self.state.returns()}")
            return True
        else:
            return False


class InteractiveGameMachine(VerboseGameMachine):

    def __init__(self, game: Union[str, pyspiel.Game]):
        super(InteractiveGameMachine, self).__init__(game)

    def await_action(self):
        human_input = input()
        action = int(human_input)
        self.play_action(action)
