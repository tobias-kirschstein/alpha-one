import numpy as np
import pyspiel

from alpha_one.alg.mcts import ImperfectInformationMCTSBot
from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent.base import Agent
from alpha_one.utils.mcts import compute_mcts_policy_reward, compute_mcts_policy_new
from alpha_one.utils.mcts_II import initialize_bot_alphaone, IIGMCTSConfig, get_policy_value_obs_node
from alpha_one.utils.statemask import get_state_mask


class IIGMCTSAgent(Agent):

    def __init__(self, bot: ImperfectInformationMCTSBot, temperature: float, state_to_value_dict: dict,
                 temperature_drop: int = None,
                 use_reward_policy: bool = False):
        super(IIGMCTSAgent, self).__init__(is_information_set_agent=True)
        self._bot = bot
        self._temperature = temperature
        self._temperature_drop = temperature_drop
        self._use_reward_policy = use_reward_policy
        self._state_to_value_dict = state_to_value_dict
        self._last_guessed_state = None
        self._last_state_policy = None

    @staticmethod
    def from_config(game, observation_model, game_model, mcts_config: IIGMCTSConfig):
        bot = initialize_bot_alphaone(game, [observation_model, game_model], mcts_config)
        return IIGMCTSAgent(bot, mcts_config.temperature, mcts_config.state_to_value,
                            temperature_drop=mcts_config.temperature_drop,
                            use_reward_policy=mcts_config.use_reward_policy)

    def next_move(self, information_set_generator: InformationSetGenerator) -> (int, np.array):
        current_turn = len(information_set_generator.get_observation_history()) + 1
        root, _ = self._bot.mcts_search(information_set_generator)

        if self._use_reward_policy:
            policy_fn = compute_mcts_policy_reward
        else:
            policy_fn = compute_mcts_policy_new

        information_set = information_set_generator.calculate_information_set()

        state_mask, index_track = get_state_mask(self._state_to_value_dict, information_set)
        state_masked_policy, state_policy = get_policy_value_obs_node(root, state_mask, index_track, self._temperature,
                                                                      self._use_reward_policy)

        guessed_state_id = np.argmax(state_policy)
        guessed_node = [c for c in root.children if c.action == guessed_state_id][0]
        self._last_guessed_state = information_set[guessed_state_id]
        self._last_state_policy = state_policy

        if not self._temperature_drop or current_turn < self._temperature_drop:
            policy = policy_fn(guessed_node, information_set_generator.get_legal_actions_mask(),
                               temperature=self._temperature)
        else:
            policy = policy_fn(guessed_node, information_set_generator.get_legal_actions_mask(),
                               temperature=0)

        action = np.random.choice(len(policy), p=policy)
        return action, policy

    def predict_direct_state_policy(self, information_set_generator):

        obs = [information_set_generator.get_padded_observation_history(self._bot.evaluator.n_previous_observations)]

        information_set = information_set_generator.calculate_information_set()
        state_mask, _ = get_state_mask(self._bot.state_to_value_dict, information_set)
        value, policy = self._bot.evaluator._observation_model.inference(obs, [state_mask])

        return value, policy

    def get_last_guessed_state(self):
        return self._last_guessed_state

    def get_last_state_policy(self):
        return self._last_state_policy

    def evaluate(self, state: pyspiel.State) -> float:
        root, _ = self._bot.mcts_search(state)
        return root.total_reward
