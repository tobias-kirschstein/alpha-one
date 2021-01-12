import numpy as np
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

from alpha_one.game.trajectory import GameTrajectory
from alpha_one.model.config.base import ModelConfig


class MCTSConfig(ModelConfig):

    def __init__(self,
                 uct_c: float,
                 max_mcts_simulations: int,
                 temperature: float,
                 temperature_drop: int = None,
                 policy_epsilon: float = None,
                 policy_alpha: float = None):
        super(MCTSConfig, self).__init__(
            uct_c=uct_c,
            max_mcts_simulations=max_mcts_simulations,
            temperature=temperature,
            temperature_drop=temperature_drop,
            policy_epsilon=policy_epsilon,
            policy_alpha=policy_alpha
        )


def initialize_bot(game, model, uct_c, max_simulations, policy_epsilon=None, policy_alpha=None):
    if policy_epsilon is None or policy_alpha is None:
        noise = None
    else:
        noise = (policy_epsilon, policy_alpha)

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


def play_one_game(game, bots, temperature, temperature_drop):
    trajectory = GameTrajectory()
    state = game.new_initial_state()
    current_turn = 0
    while not state.is_terminal():
        root = bots[state.current_player()].mcts_search(state)

        if not temperature_drop or current_turn < temperature_drop:
            policy = compute_mcts_policy(game, root, temperature)
        else:
            policy = compute_mcts_policy(game, root, 0)

        action = np.random.choice(len(policy), p=policy)

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
