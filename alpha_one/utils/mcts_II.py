import numpy as np

from alpha_one.alg.mcts import ImperfectInformationMCTSBot
from alpha_one.alg.imperfect_information import AlphaOneImperfectInformationMCTSEvaluator
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.utils.mcts import MCTSConfig
from alpha_one.game.trajectory import TrajectoryState
from open_spiel.python.algorithms.mcts import SearchNode
from alpha_one.utils.statemask import get_state_mask


# Initialize the bot here
def initialize_bot_alphaone(game, model, mcts_config: MCTSConfig):
    
    if mcts_config.policy_epsilon == None or mcts_config.policy_alpha == None:
        noise = None
    else:
        noise = (mcts_config.policy_epsilon, mcts_config.policy_alpha)
    
    evaluator = AlphaOneImperfectInformationMCTSEvaluator(game, mcts_config.state_to_value, model[0], model[1])

    bot = ImperfectInformationMCTSBot(game,
                                      mcts_config.uct_c,
                                      mcts_config.max_mcts_simulations,
                                      evaluator,
                                      solve=False,
                                      dirichlet_noise=noise,
                                      child_selection_fn=SearchNode.puct_value,
                                      verbose=False)
    
    return bot

# get policy and value at the observation node
def get_policy_value_obs_node(root, state_mask, index_track, mcts_config: MCTSConfig):
    
    # state_mask and state_masked_policy are used while training NN

    policy = np.zeros(len(root.children))
    for c in root.children:
        if c.explore_count == 0:
            policy[c.action] += c.total_reward
        else:
            policy[c.action] += c.total_reward / c.explore_count
    
    if mcts_config.temperature != 0:
        policy = policy ** (1 / mcts_config.temperature)
    
    # make policy positive
    policy = np.exp(policy)
    policy /= policy.sum()

    state_masked_policy = np.zeros(len(state_mask))
    
    for i in range(len(index_track)):
        state_masked_policy[index_track] = policy[i]
        
    return state_masked_policy, policy

# get policy and value at the game node after guessing the state
def get_policy_value_game_node(root, guess_state, mcts_config: MCTSConfig, game):
    
    policy = np.zeros(game.num_distinct_actions())
    
    for c in root.children[guess_state].children:
        #policy[c.action] = c.explore_count
        if c.explore_count == 0:
            policy[c.action] += c.total_reward
        else:
            policy[c.action] += c.total_reward / c.explore_count
    
    
    if mcts_config.temperature != 0:
        policy = policy ** (1 / mcts_config.temperature)
    policy = np.exp(policy)
    policy /= policy.sum()
    return policy

def ii_mcts_agent(information_set_generator, mcts_config: MCTSConfig, bot, game):
    root, _ = bot.mcts_search(information_set_generator)
    
    information_set = information_set_generator.calculate_information_set()
    
    state_mask, index_track = get_state_mask(mcts_config.state_to_value, information_set)
    
    state_masked_policy, state_policy = get_policy_value_obs_node(root, state_mask, index_track, mcts_config)
    
    guess_state = np.argmax(state_policy)
    
    game_node_policy = get_policy_value_game_node(root, guess_state, mcts_config, game)

    return state_masked_policy, game_node_policy, information_set[guess_state], state_mask



def play_one_game_alphaone(game, bots, mcts_config: MCTSConfig):

    trajectory_observation = GameTrajectory(game)
    trajectory_game = GameTrajectory(game, omniscient_observer=True)
    state = game.new_initial_state()
    information_set_generator = InformationSetGenerator(game)
  
    current_turn = 0

    while not state.is_terminal():
    
        if state.current_player() < 0:
            action = np.random.choice(state.legal_actions())
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)

        else:
            bot = bots[state.current_player()]
            state_masked_policy, game_node_policy, guess_state, state_mask = ii_mcts_agent(
                                                                           information_set_generator, 
                                                                           mcts_config, 
                                                                           bot,
                                                                           game)

            if mcts_config.temperature_drop == None:
                action = np.argmax(game_node_policy)
            elif current_turn < mcts_config.temperature_drop:
                action = np.random.choice(len(game_node_policy), p=game_node_policy)
            else:
                action = np.argmax(game_node_policy)

            trajectory_observation.states.append(TrajectoryState(state.observation_tensor(), 
                                                                 state.current_player(), 
                                                                 state_mask, 
                                                                 action,
                                                                 state_masked_policy))


            trajectory_game.append(guess_state, action, game_node_policy)
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)

    trajectory_observation.returns = state.returns()
    trajectory_game.returns = state.returns()
    return trajectory_observation, trajectory_game



    