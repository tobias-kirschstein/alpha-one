import numpy as np

from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.model.agent import MCTSAgent
from alpha_one.model.agent.base import Agent
from alpha_one.utils.determinized_mcts import initialize_bot, initialize_rollout_dmcts_bot, compute_mcts_policy_reward, compute_mcts_policy_new
from alpha_one.utils.mcts import MCTSConfig


class DMCTSAgent(Agent):

    def __init__(self, model, mcts_config: MCTSConfig, n_rollouts=None, information_set_weights_fn=None):
        super(DMCTSAgent, self).__init__(is_information_set_agent=True)
        self._model = model
        self._mcts_config = mcts_config
        self._n_rollouts = n_rollouts
        self._information_set_weights_fn = information_set_weights_fn

    def next_move(self, information_set_generator: InformationSetGenerator) -> (int, np.array):
        current_player = information_set_generator.current_player()
        information_set = information_set_generator.calculate_information_set(current_player)
        policy = np.zeros(information_set_generator.game.num_distinct_actions())
        legal_actions_mask = np.array(information_set_generator.get_legal_actions_mask(), dtype=np.bool)

        if self._information_set_weights_fn is None:
            information_set_weights = [1.0 / len(information_set) for _ in range(len(information_set))]
        else:
            information_set_weights = self._information_set_weights_fn(information_set_generator)

        # Evaluate each state in the information set by MCTS independently.
        # After the searches are completed, the numbers of visits for each action from the root
        # are summed across all trees,
        # and an action is chosen that maximises the total number of visits.
        for w, s in zip(information_set_weights, information_set):
            if self._model is None and self._n_rollouts is not None:
                bot = initialize_rollout_dmcts_bot(information_set_generator.game, self._n_rollouts, self._mcts_config)
            else:
                bot = initialize_bot(information_set_generator.game, self._model, self._mcts_config)
        
            root = bot.mcts_search(s)

            if self._mcts_config.use_reward_policy:
                policy_temp = compute_mcts_policy_reward(information_set_generator.game, root)
            else:
                policy_temp = compute_mcts_policy_new(information_set_generator.game, root)

            policy = np.add(policy, w * policy_temp)
    
        if self._mcts_config.temperature == 0 or self._mcts_config.temperature is None:
            # Return single-peaked policy with argmax
            new_policy = np.zeros(len(legal_actions_mask))
            # Explicitly set illegal actions to -inf as it can happen that policy is all 0
            # (if no child was explored or all rewards are 0)
            policy[~legal_actions_mask] = float('-inf')
            new_policy[policy.argmax(-1)] = 1
            policy = new_policy
        else:
            policy = policy ** (1 / self._mcts_config.temperature)
            policy_exp = np.exp(policy, where=legal_actions_mask)
            policy = policy_exp / np.sum(policy_exp)
        #for w, s in zip(information_set_weights, information_set):
        #    if self._model is None and self._n_rollouts is not None:
        #        bot = initialize_rollout_dmcts_bot(information_set_generator.game, self._n_rollouts, self._mcts_config)
        #    else:
        #        bot = initialize_bot(information_set_generator.game, self._model, self._mcts_config)
        #    
        #    mcts_agent = MCTSAgent(information_set_generator.game, bot, self._mcts_config.temperature,
        #                           temperature_drop=self._mcts_config.temperature_drop,
        #                           use_reward_policy=self._mcts_config.use_reward_policy)
        #    _, state_policy = mcts_agent.next_move(s)
        #    policy += w * state_policy

        #policy_exp = np.exp(policy, where=information_set_generator.get_legal_actions_mask())
        #policy = policy_exp / np.sum(policy_exp)

        action = np.random.choice(len(policy), p=policy)
        return action, policy

    def evaluate(self, information_set_generator: InformationSetGenerator) -> float:
        current_player = information_set_generator.current_player()
        information_set = information_set_generator.calculate_information_set(current_player)

        value = 0
        for s in information_set:
            mcts_agent = MCTSAgent.from_config(information_set_generator.game,
                                               self._model,
                                               self._mcts_config,
                                               normalize_policy=False)
            state_value = mcts_agent.evaluate(s)
            value += state_value
        return value
