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
    def __init__(self):
        self.states = []
        self.returns = None

    def append(self, game_state, action, policy):
        self.states.append(TrajectoryState(
            game_state.observation_tensor(),
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
