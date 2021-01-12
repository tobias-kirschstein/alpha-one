import numpy as np
import pyspiel
from open_spiel.python.algorithms.mcts import MCTSBot

from alpha_one.model.agent.base import Agent
from alpha_one.utils.mcts import compute_mcts_policy, MCTSConfig, initialize_bot


class MCTSAgent(Agent):

    def __init__(self, game, bot: MCTSBot, temperature, temperature_drop=None):
        self.game = game
        self.bot = bot
        self.temperature = temperature
        self.temperature_drop = temperature_drop

    @staticmethod
    def from_config(game, model, config: MCTSConfig):
        bot = initialize_bot(game,
                             model,
                             uct_c=config.uct_c,
                             max_simulations=config.max_mcts_simulations,
                             policy_epsilon=config.policy_epsilon,
                             policy_alpha=config.policy_alpha)
        return MCTSAgent(game, bot, config.temperature, config.temperature_drop)

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        current_turn = len(state.history()) + 1
        root = self.bot.mcts_search(state)
        if not self.temperature_drop or current_turn < self.temperature_drop:
            policy = compute_mcts_policy(self.game, root, self.temperature)
        else:
            policy = compute_mcts_policy(self.game, root, 0)

        action = np.random.choice(len(policy), p=policy)
        return action, policy
