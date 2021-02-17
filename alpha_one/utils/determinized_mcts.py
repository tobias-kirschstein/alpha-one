from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import numpy as np
from open_spiel.python.algorithms import mcts

from alpha_one.utils.mcts import MCTSConfig
from alpha_one.alg.imperfect_information import DeterminizedMCTSEvaluator
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.game.information_set import InformationSetGenerator


def initialize_bot(game, model, mcts_config: MCTSConfig):
    
    if mcts_config.policy_epsilon == None or mcts_config.policy_alpha == None:
        noise = None
    else:
        noise = (mcts_config.policy_epsilon, mcts_config.policy_alpha)
    
    evaluator = DeterminizedMCTSEvaluator(model)

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

def compute_mcts_policy(game, model, state, information_set_generator, mcts_config: MCTSConfig):

    current_player = state.current_player()
    information_set = information_set_generator.calculate_information_set(current_player)
    policy = np.zeros(game.num_distinct_actions())

    for s in information_set:
        bot = initialize_bot(game, model, mcts_config)

        root = bot.mcts_search(s)

        for c in root.children:
            if c.explore_count == 0:
        	    policy[c.action] += c.total_reward
            else:
                policy[c.action] += c.total_reward / c.explore_count
    
    if mcts_config.temperature != 0:
        policy = policy ** (1 / mcts_config.temperature)
    policy = (policy - np.min(policy)) / (np.max(policy) - np.min(policy))
    policy[np.where(state.legal_actions_mask() == 0)] = 0
    policy /= policy.sum()
    return policy

def play_one_game_d(game, models, mcts_config: MCTSConfig):
    trajectory = GameTrajectory()

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

