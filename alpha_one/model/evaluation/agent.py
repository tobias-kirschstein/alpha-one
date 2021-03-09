from alpha_one.game.trajectory import GameTrajectory
from alpha_one.metrics import MatchOutcome
from alpha_one.model.agent.base import Agent
from alpha_one.utils.mcts import MCTSConfig, initialize_bot
from alpha_one.utils.play import GameMachine


class MCTSAgent:

    def __init__(self, game, model, mcts_config: MCTSConfig):
        self.mcts_bot = initialize_bot(game, model, mcts_config.uct_c,
                                       mcts_config.max_mcts_simulations,
                                       mcts_config.policy_epsilon, mcts_config.policy_alpha)

    def next_move(self):
        pass


class AgentEvaluator:

    def __init__(self, game):
        self._game = game
        self._game_machine = GameMachine(game)

    def evaluate(self, agent1: Agent, agent2: Agent):
        self._game_machine.new_game()
        trajectory = GameTrajectory(self._game)

        while not self._game_machine.is_finished():
            current_player = self._game_machine.current_player()
            current_agent = agent1 if current_player == 0 else agent2

            if current_agent.is_information_set_agent():
                action, policy = current_agent.next_move(self._game_machine.get_information_set_generator())
            else:
                action, policy = current_agent.next_move(self._game_machine.get_state())

            trajectory.append(self._game_machine.get_state(), action, policy)
            self._game_machine.play_action(action)

        rewards = self._game_machine.get_rewards()
        trajectory.set_final_rewards(rewards)
        match_outcome = MatchOutcome.win(0, 1, rewards) \
            if rewards[0] == 1 \
            else MatchOutcome.defeat(0, 1, rewards)

        return match_outcome, trajectory
