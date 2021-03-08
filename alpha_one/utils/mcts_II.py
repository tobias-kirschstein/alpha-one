import numpy as np
from open_spiel.python.algorithms.mcts import SearchNode

from alpha_one.alg.imperfect_information import AlphaOneImperfectInformationMCTSEvaluator
from alpha_one.alg.mcts import ImperfectInformationMCTSBot
from alpha_one.game.information_set import InformationSetGenerator
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.game.trajectory import TrajectoryState
from alpha_one.utils.mcts import MCTSConfig, compute_mcts_policy_reward, compute_mcts_policy_new
from alpha_one.utils.statemask import get_state_mask


class IIGMCTSConfig(MCTSConfig):

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
                 alpha_one: bool = False,
                 state_to_value=None,
                 n_previous_observations=1,
                 optimism=1,
                 **kwargs):
        super(IIGMCTSConfig, self).__init__(
            uct_c=uct_c,
            max_mcts_simulations=max_mcts_simulations,
            temperature=temperature,
            temperature_drop=temperature_drop,
            policy_epsilon=policy_epsilon,
            policy_alpha=policy_alpha,
            omniscient_observer=omniscient_observer,
            use_reward_policy=use_reward_policy,
            determinized_MCTS=determinized_MCTS,
            alpha_one=alpha_one,
            state_to_value=state_to_value,
            n_previous_observations=n_previous_observations,
            optimism=optimism,
            **kwargs)


# Initialize the bot here
def initialize_bot_alphaone(game, model, mcts_config: IIGMCTSConfig):
    if mcts_config.policy_epsilon == None or mcts_config.policy_alpha == None:
        noise = None
    else:
        noise = (mcts_config.policy_epsilon, mcts_config.policy_alpha)

    evaluator = AlphaOneImperfectInformationMCTSEvaluator(game, mcts_config.state_to_value, model[0], model[1],
                                                          n_previous_observations=mcts_config.n_previous_observations)

    bot = ImperfectInformationMCTSBot(game,
                                      mcts_config.uct_c,
                                      mcts_config.max_mcts_simulations,
                                      evaluator,
                                      solve=False,
                                      dirichlet_noise=noise,
                                      child_selection_fn=SearchNode.puct_value,
                                      verbose=False,
                                      optimism=mcts_config.optimism)

    return bot


# get policy and value at the observation node
def get_policy_value_obs_node(root, state_mask, index_track, temperature, use_reward_policy):
    # state_mask and state_masked_policy are used while training NN

    if use_reward_policy:
        policy = compute_mcts_policy_reward(root, np.ones(len(root.children)), temperature=temperature)
    else:
        policy = compute_mcts_policy_new(root, np.ones(len(root.children)), temperature=temperature)

    state_masked_policy = np.zeros(len(state_mask))

    for i in range(len(index_track)):
        state_masked_policy[index_track[i]] = policy[i]

    return state_masked_policy, policy


def ii_mcts_agent(information_set_generator, mcts_config: IIGMCTSConfig, bot):
    root, _ = bot.mcts_search(information_set_generator)

    information_set = information_set_generator.calculate_information_set()

    state_mask, index_track = get_state_mask(mcts_config.state_to_value, information_set)

    state_masked_policy, state_policy = get_policy_value_obs_node(root, state_mask, index_track,
                                                                  mcts_config.temperature,
                                                                  mcts_config.use_reward_policy)

    guessed_state_id = np.argmax(state_policy)
    guessed_state = information_set[guessed_state_id]

    guessed_node = [c for c in root.children if c.action == guessed_state_id][0]

    if mcts_config.use_reward_policy:
        game_node_policy = compute_mcts_policy_reward(guessed_node,
                                                      guessed_state.legal_actions_mask(),
                                                      temperature=mcts_config.temperature)
    else:
        game_node_policy = compute_mcts_policy_new(guessed_node,
                                                   guessed_state.legal_actions_mask(),
                                                   temperature=mcts_config.temperature)

    return state_masked_policy, game_node_policy, guessed_state, state_mask


def play_one_game_alphaone(game, bots, mcts_config: IIGMCTSConfig, use_teacher_forcing=False):
    trajectory_observation = GameTrajectory(game)
    trajectory_game = GameTrajectory(game, omniscient_observer=True)
    state = game.new_initial_state()
    information_set_generator = InformationSetGenerator(game)

    current_turn = 0

    while not state.is_terminal():

        if state.current_player() < 0:
            actions, probs = list(zip(*state.chance_outcomes()))
            action = np.random.choice(actions, p=probs)
            information_set_generator.register_action(action)
            state.apply_action(action)
            information_set_generator.register_observation(state)
        else:
            bot = bots[state.current_player()]
            state_masked_policy, game_node_policy, guess_state, state_mask = ii_mcts_agent(
                information_set_generator,
                mcts_config,
                bot)

            if mcts_config.temperature_drop == None:
                action = np.argmax(game_node_policy)
            elif current_turn < mcts_config.temperature_drop:
                action = np.random.choice(len(game_node_policy), p=game_node_policy)
            else:
                action = np.argmax(game_node_policy)
            assert state.legal_actions_mask()[
                       action] == 1, f"Illegal action played: {action}, policy: {game_node_policy}, legal actions mask: {guess_state.legal_actions_mask()}, observation history: {[(s.current_player, s.action) for s in trajectory_game.states]}"

            assert np.sum(state_mask) == len(
                information_set_generator.calculate_information_set()), f"State mask should reflect information set"

            if use_teacher_forcing:
                state_masked_policy = np.zeros(len(mcts_config.state_to_value))
                state_masked_policy[mcts_config.state_to_value[state.__str__()]] = 1

            padded_history = information_set_generator.get_padded_observation_history(
                mcts_config.n_previous_observations)

            trajectory_observation.states.append(TrajectoryState(padded_history,
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
