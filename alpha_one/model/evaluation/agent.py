from alpha_one.game.trajectory import GameTrajectory
from alpha_one.metrics import MatchOutcome
from alpha_one.model.agent.base import Agent
from alpha_one.utils.mcts import MCTSConfig, initialize_bot


class MCTSAgent:

    def __init__(self, game, model, mcts_config: MCTSConfig):
        self.mcts_bot = initialize_bot(game, model, mcts_config.uct_c,
                                       mcts_config.max_mcts_simulations,
                                       mcts_config.policy_epsilon, mcts_config.policy_alpha)

    def next_move(self):
        pass


class AgentEvaluator:

    def __init__(self, game):
        self.game = game

    def evaluate(self, agent1: Agent, agent2: Agent):
        state = self.game.new_initial_state()
        trajectory = GameTrajectory()
        while not state.is_terminal():
            current_player = state.current_player()
            current_agent = agent1 if current_player == 0 else agent2
            action, policy = current_agent.next_move(state)
            trajectory.append(state, action, policy)
            state.apply_action(action)

        trajectory.set_final_rewards(state.returns())
        match_outcome = MatchOutcome.win(0, 1) if trajectory.get_final_reward(0) == 1 else MatchOutcome.defeat(0, 1)
        return match_outcome, trajectory
