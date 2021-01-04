import numpy as np

from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms import mcts


def initialize_bot(game, model, uct_c, max_simulations, policy_epsilon=None, policy_alpha=None):
    if policy_epsilon == None or policy_alpha == None:
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