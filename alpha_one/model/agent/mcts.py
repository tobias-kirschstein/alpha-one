import numpy as np
import pyspiel
from open_spiel.python.algorithms.mcts import MCTSBot

from alpha_one.model.agent.base import Agent
from alpha_one.utils.mcts import MCTSConfig, initialize_bot, compute_mcts_policy_reward, \
    compute_mcts_policy_new


class MCTSAgent(Agent):

    def __init__(self, game, bot: MCTSBot, temperature, temperature_drop=None, use_reward_policy=False,
                 normalize_policy=True):
        super(MCTSAgent, self).__init__(is_information_set_agent=False)
        self._game = game
        self._bot = bot
        self._temperature = temperature
        self._temperature_drop = temperature_drop
        self._use_reward_policy = use_reward_policy
        self._normalize_policy = normalize_policy

    @staticmethod
    def from_config(game, model, config: MCTSConfig, normalize_policy=True):
        bot = initialize_bot(game,
                             model,
                             uct_c=config.uct_c,
                             max_simulations=config.max_mcts_simulations,
                             policy_epsilon=config.policy_epsilon,
                             policy_alpha=config.policy_alpha)
        return MCTSAgent(game, bot, config.temperature, config.temperature_drop, normalize_policy=normalize_policy)

    def next_move(self, state: pyspiel.State) -> (int, np.array):
        current_turn = len(state.history()) + 1
        root = self._bot.mcts_search(state)

        if self._use_reward_policy:
            policy_fn = compute_mcts_policy_reward
        else:
            policy_fn = compute_mcts_policy_new

        if not self._temperature_drop or current_turn < self._temperature_drop:
            policy = policy_fn(root,
                               state.legal_actions_mask(),
                               temperature=self._temperature,
                               normalize=self._normalize_policy)
        else:
            policy = policy_fn(root, state.legal_actions_mask(), temperature=0, normalize=self._normalize_policy)

        action = np.random.choice(len(policy), p=policy)
        return action, policy

    def evaluate(self, state: pyspiel.State) -> float:
        root = self._bot.mcts_search(state)
        return root.total_reward
