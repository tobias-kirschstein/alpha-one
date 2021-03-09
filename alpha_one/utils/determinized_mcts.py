from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import numpy as np
from open_spiel.python.algorithms import mcts
from alpha_one.utils.mcts import MCTSConfig
from alpha_one.alg.imperfect_information import DeterminizedMCTSEvaluator
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.game.information_set import InformationSetGenerator
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator


def initialize_bot(game, model, mcts_config: MCTSConfig):
    
    if mcts_config.policy_epsilon == None or mcts_config.policy_alpha == None:
        noise = None
    else:
        noise = (mcts_config.policy_epsilon, mcts_config.policy_alpha)
    
    evaluator = DeterminizedMCTSEvaluator(model, game)

    bot = mcts.MCTSBot(
          game,
          mcts_config.uct_c,
          mcts_config.max_mcts_simulations,
          evaluator,
          solve=False,
          dirichlet_noise=noise,
          child_selection_fn=mcts.SearchNode.puct_value,
          verbose=False)
    
    return bot

def initialize_rollout_dmcts_bot(game, n_rollouts, mcts_config: MCTSConfig):
    if mcts_config.policy_epsilon == None or mcts_config.policy_alpha == None:
        noise = None
    else:
        noise = (mcts_config.policy_epsilon, mcts_config.policy_alpha)

    evaluator = RandomRolloutEvaluator(n_rollouts)

    bot = mcts.MCTSBot(
        game,
        mcts_config.uct_c,
        mcts_config.max_mcts_simulations,
        evaluator,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False)

    return bot

def compute_mcts_policy_reward(game, root):
    policy = np.zeros(game.num_distinct_actions())
    for c in root.children:
        if c.explore_count > 0:
            if c.outcome is not None or c.explore_count == 1:
                policy[c.action] = c.total_reward / c.explore_count
            else:
                # If node is not a leaf, one explore count is used to unfold it. To get a proper average,
                # we have to subtract that here
                policy[c.action] = c.total_reward / (c.explore_count - 1)
    return policy

def compute_mcts_policy_new(game, root):
    policy = np.zeros(game.num_distinct_actions())
    for c in root.children:
        policy[c.action] = c.explore_count
    return policy

def compute_mcts_policy(game, model, state, information_set_generator, mcts_config: MCTSConfig, use_NN=True, n_rollouts=None):

    current_player = information_set_generator.current_player()
    information_set = information_set_generator.calculate_information_set(current_player)
    policy = np.zeros(game.num_distinct_actions())
    legal_actions_mask = np.array(information_set_generator.get_legal_actions_mask(), dtype=np.bool)

    for s in information_set:
        if use_NN:
            bot = initialize_bot(game, model, mcts_config)
        else:
            bot = initialize_rollout_dmcts_bot(game, n_rollouts, mcts_config)
        
        root = bot.mcts_search(s)

        if mcts_config.use_reward_policy:
            policy_temp = compute_mcts_policy_reward(game, root)
        else:
            policy_temp = compute_mcts_policy_new(game, root)

        policy = np.add(policy, policy_temp)
    
    if mcts_config.temperature == 0 or mcts_config.temperature is None:
        # Return single-peaked policy with argmax
        new_policy = np.zeros(len(legal_actions_mask))
        # Explicitly set illegal actions to -inf as it can happen that policy is all 0
        # (if no child was explored or all rewards are 0)
        policy[~legal_actions_mask] = float('-inf')
        new_policy[policy.argmax(-1)] = 1
        policy = new_policy
    else:
        policy = policy ** (1 / mcts_config.temperature)
        policy_exp = np.exp(policy, where=legal_actions_mask)
        policy = policy_exp / np.sum(policy_exp)
    return policy

def play_one_game_d(game, models, mcts_config: MCTSConfig):
    trajectory = GameTrajectory(game, omniscient_observer=mcts_config.omniscient_observer)
    
    state = game.new_initial_state()
    information_set_generator = InformationSetGenerator(game)
    
    current_turn = 0

    while not state.is_terminal():
        if state.current_player() < 0:
            action = np.random.choice(state.legal_actions())
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)

        elif state.current_player() == 0:

            policy = compute_mcts_policy(game, models[0], state, information_set_generator, mcts_config)
            if mcts_config.temperature_drop == None:
                action = np.argmax(policy)
            elif current_turn < mcts_config.temperature_drop:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)

            trajectory.append(state, action, policy)
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)
            current_turn += 1

        else:

            policy = compute_mcts_policy(game, models[1], state, information_set_generator, mcts_config)
            if mcts_config.temperature_drop == None:
                action = np.argmax(policy)
            elif current_turn < mcts_config.temperature_drop:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)

            trajectory.append(state, action, policy)
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)
            current_turn += 1



    trajectory.set_final_rewards(state.returns())
    return trajectory

