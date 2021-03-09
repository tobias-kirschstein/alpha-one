import numpy as np
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

from alpha_one.alg.imperfect_information import AlphaZeroOmniscientMCTSEvaluator
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.model.config.base import ModelConfig


class MCTSConfig(ModelConfig):

    def __init__(self,
                 uct_c: float,
                 max_mcts_simulations: int,
                 temperature: float,
                 temperature_drop: int = None,
                 policy_epsilon: float = None,
                 policy_alpha: float = None,
                 omniscient_observer: bool = False,
                 use_reward_policy: bool = False,
                 determinized_MCTS: bool = False,
                 **kwargs):
        super(MCTSConfig, self).__init__(
            uct_c=uct_c,
            max_mcts_simulations=max_mcts_simulations,
            temperature=temperature,
            temperature_drop=temperature_drop,
            policy_epsilon=policy_epsilon,
            policy_alpha=policy_alpha,
            omniscient_observer=omniscient_observer,
            use_reward_policy=use_reward_policy,
            determinized_MCTS=determinized_MCTS,
            **kwargs
        )


def initialize_bot(game, model, uct_c, max_simulations, policy_epsilon=None, policy_alpha=None,
                   omniscient_observer=False):
    if policy_epsilon is None or policy_alpha is None:
        noise = None
    else:
        noise = (policy_epsilon, policy_alpha)

    if omniscient_observer:
        az_evaluator = AlphaZeroOmniscientMCTSEvaluator(game, model)
    else:
        az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

    bot = mcts.MCTSBot(
        game,
        uct_c,
        max_simulations,
        az_evaluator,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False)

    return bot


def initialize_rollout_mcts_bot(game, n_rollouts, uct_c, max_simulations, policy_epsilon=None, policy_alpha=None):
    if policy_epsilon is None or policy_alpha is None:
        noise = None
    else:
        noise = (policy_epsilon, policy_alpha)

    evaluator = RandomRolloutEvaluator(n_rollouts)

    bot = mcts.MCTSBot(
        game,
        uct_c,
        max_simulations,
        evaluator,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False)

    return bot


def compute_mcts_policy(game, root, temperature):
    policy = np.zeros(game.num_distinct_actions())

    for c in root.children:
        policy[c.action] = c.explore_count
    if temperature == 0 or temperature is None:
        # Create probability distribution with peak at most likely action
        new_policy = np.zeros(game.num_distinct_actions())
        new_policy[policy.argmax(-1)] = 1
        policy = new_policy
    else:
        policy = policy ** (1 / temperature)
        policy /= policy.sum()
    return policy


def compute_mcts_policy_new(root, legal_actions_mask, temperature: float = 1.0, normalize=True):
    legal_actions_mask = np.array(legal_actions_mask, dtype=np.bool)
    policy = np.zeros(len(legal_actions_mask))

    for c in root.children:
        policy[c.action] = c.explore_count

    assert np.sum(
        policy[~np.array(legal_actions_mask, dtype=np.bool)]) == 0, f"MCTS gave non-zero weight to illegal actions!"

    if temperature == 0 or temperature is None:
        # Return single-peaked policy with argmax
        new_policy = np.zeros(len(legal_actions_mask))
        # Explicitly set illegal actions to -inf as it can happen that policy is all 0
        # (if no child was explored or all rewards are 0)
        policy[~legal_actions_mask] = float('-inf')
        new_policy[policy.argmax(-1)] = 1
        policy = new_policy
    else:
        policy = policy ** (1 / temperature)

        if normalize:
            policy /= policy.sum()

    return policy


def compute_mcts_policy_reward(root, legal_actions_mask, temperature: float = 1.0, normalize=True):
    """
    Computes a policy after an MCTS search has been conducted by leveraging both explore counts and total rewards
    of the child nodes.

    Parameters
    ----------
    root:
        node where MCTS was started
    legal_actions_mask:
        binary mask to indicate which actions are legal at current node
    temperature: (optional)
        Temperature exponent for the final policy
    normalize:
        default = True
        Whether or not the returned policy should be normalized via softmax to yield a proper probability

    Returns
    -------
        A policy with the same size as `legal_actions_mask`
    """

    legal_actions_mask = np.array(legal_actions_mask, dtype=np.bool)
    policy = np.zeros(len(legal_actions_mask))

    for c in root.children:
        if c.explore_count > 0:
            if c.outcome is not None or c.explore_count == 1:
                policy[c.action] = c.total_reward / c.explore_count
            else:
                # If node is not a leaf, one explore count is used to unfold it. To get a proper average,
                # we have to subtract that here
                policy[c.action] = c.total_reward / (c.explore_count - 1)

    assert np.sum(
        policy[~np.array(legal_actions_mask, dtype=np.bool)]) == 0, f"MCTS gave non-zero weight to illegal actions!"

    if temperature == 0 or temperature is None:
        # Return single-peaked policy with argmax
        new_policy = np.zeros(len(legal_actions_mask))
        # Explicitly set illegal actions to -inf as it can happen that policy is all 0
        # (if no child was explored or all rewards are 0)
        policy[~legal_actions_mask] = float('-inf')
        new_policy[policy.argmax(-1)] = 1
        policy = new_policy
    else:
        assert temperature == int(temperature) or np.min(policy[legal_actions_mask]) >= 0, \
            f"negative policy values {policy[legal_actions_mask]} cannot be used with non-integer temperatures {temperature}!"
        policy = policy ** (1 / temperature)

        if normalize:
            # softmax
            policy_exp = np.exp(policy, where=legal_actions_mask)
            policy = policy_exp / np.sum(policy_exp)

    return policy


def play_one_game(game, bots, temperature, temperature_drop, omniscient_observer=False, use_reward_policy=False):
    trajectory = GameTrajectory(game, omniscient_observer=omniscient_observer)
    state = game.new_initial_state()
    current_turn = 0
    while not state.is_terminal():
        if state.is_chance_node():
            policy = np.zeros(game.max_chance_outcomes())
            for action, prob in state.chance_outcomes():
                policy[action] = prob
        else:
            root = bots[state.current_player()].mcts_search(state)

            if not temperature_drop or current_turn < temperature_drop:
                current_temperature = temperature
            else:
                current_temperature = 0

            if use_reward_policy:
                policy = compute_mcts_policy_reward(root, state.legal_actions_mask(), current_temperature)
            else:
                policy = compute_mcts_policy(game, root, temperature)

        action = np.random.choice(len(policy), p=policy)

        if not state.is_chance_node():
            # TODO: consider whether chance player actions should be recorded as well and filtered out later
            trajectory.append(state, action, policy)
        state.apply_action(action)
        current_turn += 1

    trajectory.set_final_rewards(state.returns())
    return trajectory


def mcts_inference(game, model, state, uct_c, max_simulations, temperature, policy_epsilon=None, policy_alpha=None):
    bot = initialize_bot(game,
                         model,
                         uct_c=uct_c,
                         max_simulations=max_simulations,
                         policy_epsilon=policy_epsilon,
                         policy_alpha=policy_alpha)
    root = bot.mcts_search(state)
    return compute_mcts_policy(game, root, temperature)


def investigate_node(node, level=0, uct_c=2):
    for s1 in node.children:
        print(''.join(['  '] * (level + 1)),
              f"p{s1.player}, {s1.action}, explore: {s1.explore_count}, reward: {s1.total_reward: 0.2f}"
              f", puct {s1.puct_value(node.explore_count, uct_c):0.3f}, outcome: {s1.outcome}")
        investigate_node(s1, level=level + 2)
