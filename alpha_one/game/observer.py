import pyspiel
from open_spiel.python.observation import make_observation
import numpy as np


class OmniscientObserver:

    def __init__(self, game: pyspiel.Game, perfect_recall=False, public_info=True):
        self.game = game
        self.is_game_without_all_players_implementation = game.get_type().short_name in {'kuhn_poker'}
        if self.is_game_without_all_players_implementation:
            self.observer = make_observation(
                game,
                pyspiel.IIGObservationType(
                    perfect_recall=perfect_recall,
                    public_info=public_info,
                    private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
        else:
            self.observer = make_observation(
                game,
                pyspiel.IIGObservationType(
                    perfect_recall=perfect_recall,
                    public_info=public_info,
                    private_info=pyspiel.PrivateInfoType.ALL_PLAYERS))

    def get_observation_tensor(self, state):
        if self.is_game_without_all_players_implementation:
            observation = []
            for player_id in range(self.game.num_players()):
                self.observer.set_from(state, player_id)
                observation.append(np.array(self.observer.tensor))
            return np.concatenate(observation)
        else:
            self.observer.set_from(state, 0)
            return self.observer.tensor
