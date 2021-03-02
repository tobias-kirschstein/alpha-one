from open_spiel.python.observation import make_observation
import pyspiel

from alpha_one.game.observer import OmniscientObserver


class TrajectoryState(object):

    # A particular point along a trajectory.
    def __init__(self, observation, current_player, legals_mask, action, policy):
        self.observation = observation
        self.current_player = current_player
        self.legals_mask = legals_mask
        self.action = action
        self.policy = policy


class GameTrajectory(object):
    # A sequence of observations, actions and policies, and the outcomes.
    def __init__(self, game, omniscient_observer=False):
        """
        Parameters
        ----------
        omniscient_observer:
            If set, the game trajectory will contain both the current player's single observation as well as the
            omniscient observation.
        """

        self.states = []
        self.returns = None

        if omniscient_observer:
            self.omniscient_observer = OmniscientObserver(game)
        else:
            self.omniscient_observer = None

        # TODO: let the observer have perfect recall? I.e., contain information about all past decisions of player?
        self.observer = make_observation(
            game,
            pyspiel.IIGObservationType(
                perfect_recall=False,
                public_info=True,
                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))

    def append(self, game_state, action, policy):
        self.observer.set_from(game_state, game_state.current_player())
        player_observation_tensor = self.observer.tensor

        if self.omniscient_observer is not None:
            omniscient_observation_tensor = self.omniscient_observer.get_observation_tensor(game_state)
            observation_tensor = {'player_observation': player_observation_tensor,
                                  'omniscient_observation': omniscient_observation_tensor}
        else:
            observation_tensor = player_observation_tensor

        self.states.append(TrajectoryState(
            observation_tensor,
            game_state.current_player(),
            game_state.legal_actions_mask(),
            action,
            policy))

    def set_final_rewards(self, final_returns):
        self.returns = final_returns

    def get_final_rewards(self):
        return self.returns

    def get_final_reward(self, player_id):
        return self.returns[player_id]

    def get_player_states(self, player_id):
        return [s for s in self.states if s.current_player == player_id]

    def __len__(self):
        return len(self.states)
