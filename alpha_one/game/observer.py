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

    def get_observation_string(self, state):
        if self.is_game_without_all_players_implementation:
            observation_strings = [self.observer.string_from(state, player_id) for player_id in
                                   range(self.game.num_players())]
            return '\n'.join(observation_strings)
        else:
            return self.observer.string_from(state, 0)


def get_observation_tensor_shape(game: pyspiel.Game, omniscient_observer=False):
    """
    Calculates the shape of the tensor that will be fed into a NN.
    Usually, this is just the shape of the player's observation tensor, but if omniscient_observer is specified,
    it will be the shape of this larger tensor instead.

    Parameters
    ----------
    game
    omniscient_observer:
        whether to use the shape of the larger omniscient observation tensor instead of the single player's private
        observation tensor.

    Returns
    -------
        the shape of the respective observation tensor
    """

    if omniscient_observer:
        observation_tensor_shape = OmniscientObserver(game).get_observation_tensor(
            game.new_initial_state()).shape
    else:
        observer = make_observation(
            game,
            pyspiel.IIGObservationType(
                perfect_recall=False,
                public_info=True,
                private_info=pyspiel.PrivateInfoType.ALL_PLAYERS))
        observation_tensor_shape = observer.tensor.shape[0]
    return observation_tensor_shape
