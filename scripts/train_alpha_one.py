from statistics import mean

import pyspiel

from alpha_one.data.replay import ReplayDataManager
from alpha_one.game.observer import get_observation_tensor_shape
from alpha_one.metrics import EloRatingSystem, TrueSkillRatingSystem
from alpha_one.model.config import OpenSpielModelConfig
from alpha_one.model.evaluation import EvaluationManager
from alpha_one.model.model_manager import OpenSpielCheckpointManager
from alpha_one.train import AlphaOneTrainManager
from alpha_one.utils.logging import TensorboardLogger, generate_run_name
from alpha_one.utils.mcts_II import IIGMCTSConfig
from alpha_one.utils.state_to_value import state_to_value
from env import LOGS_DIR

# =========================================================================
# Settings
# =========================================================================

game_name = 'leduc_poker'
game_prefix = 'LP-local'

continue_training = 'LP-local-43'

n_iterations = 1000  # How often the whole procedure is repeated. Also corresponds to the number of evaluations

# Train samples generation
n_games_train = 300  # How many new states will be generated by the best model via self-play for training (Training set size delta). Has to be larger than batch_size
n_games_valid = 10
store_replays_every = 10

# Model update
n_most_recent_train_samples = 5000  # Among which training samples to choose to train current model
n_most_recent_valid_samples = 5000
n_train_steps_obs = 800  # Gradient updates for observation model
n_train_steps_game = 40  # Gradient updates for game model
n_valid_steps = 10
batch_size = 32

# Evaluation
n_evaluations = 50  # How many games should be played to measure which model is better
evaluation_strategy = 'mcts'  # 'best_response'
win_ratio_needed = None #0.55                # Minimum win ratio that the challenger model needs in order to supersede the current best model
average_reward_needed = 0.2            # Minimum average reward over current best model that the challenger model needs in order to supersede the current best model. Mutually exclusive with win_ratio_needed

# MCTS config
UCT_C = 5  # Exploration constant. Should be higher if absolute rewards are higher in a game
max_mcts_simulations = 100
optimism = 0.1  # Whether guessing states is biased towards good outcomes

policy_epsilon = None  # 0.25            # What noise epsilon to use
policy_alpha = None  # 1                 # What dirichlet noise alpha to use

temperature = 1
temperature_drop = 10

alpha_one = True
omniscient_observer = True  # Whether the game model should have total information of the state it guessed
use_reward_policy = True  # Whether the total rewards of nodes should be taken into account when constructing policies, or only the explore_counts
use_teacher_forcing = False  # Whether the true game states should be used as label for the observation model, or the guessing policy of the IIG-MCTS
n_previous_observations = 3            # How many previous observations the observation model should use

# Model Hyperparameters
model_type_obs = 'mlp'
nn_width_obs = 128
nn_depth_obs = 4
weight_decay_obs = 1e-5
learning_rate_obs = 1e-3

model_type_game = 'mlp'
nn_width_game = 64
nn_depth_game = 2
weight_decay_game = 1e-5
learning_rate_game = 1e-5

# =========================================================================
# END Settings
# =========================================================================

assert win_ratio_needed is None and average_reward_needed is not None \
       or win_ratio_needed is not None and average_reward_needed is None, \
    f"win_ratio_needed and average_reward_needed are mutually exclusive"

# this is state to id
state_to_value_dict = state_to_value(game_name)

mcts_config = IIGMCTSConfig(
    UCT_C,
    max_mcts_simulations,
    temperature,
    temperature_drop,
    policy_epsilon,
    policy_alpha,
    alpha_one=alpha_one,
    state_to_value=state_to_value_dict,
    use_reward_policy=use_reward_policy,
    optimism=optimism,
    n_previous_observations=n_previous_observations)

evaluation_mcts_config = IIGMCTSConfig(
    UCT_C,
    max_mcts_simulations,
    0,
    temperature_drop=None,
    policy_epsilon=None,
    policy_alpha=None,
    alpha_one=alpha_one,
    state_to_value=state_to_value_dict,
    use_reward_policy=use_reward_policy,
    optimism=optimism,
    n_previous_observations=n_previous_observations)

hyperparameters = dict(
    game_name=game_name,
    UCT_C=UCT_C,
    max_mcts_simulations=max_mcts_simulations,
    n_iterations=n_iterations,

    n_games_train=n_games_train,
    n_games_valid=n_games_valid,
    store_replays_every=store_replays_every,

    n_most_recent_train_samples=n_most_recent_train_samples,
    n_most_recent_valid_samples=n_most_recent_valid_samples,
    n_train_steps_obs=n_train_steps_obs,
    n_train_steps_game=n_train_steps_game,
    n_valid_steps=n_valid_steps,
    batch_size=batch_size,

    n_evaluations=n_evaluations,
    win_ratio_needed=win_ratio_needed,

    policy_epsilon=policy_epsilon,
    policy_alpha=policy_alpha,

    temperature=temperature,
    temperature_drop=temperature_drop,

    model_type_obs=model_type_obs,
    nn_width_obs=nn_width_obs,
    nn_depth_obs=nn_depth_obs,
    weight_decay_obs=weight_decay_obs,
    learning_rate_obs=learning_rate_obs,

    model_type_game=model_type_game,
    nn_width_game=nn_width_game,
    nn_depth_game=nn_depth_game,
    weight_decay_game=weight_decay_game,
    learning_rate_game=learning_rate_game,

    omniscient_observer=omniscient_observer,
    use_reward_policy=use_reward_policy,
    optimism=optimism,
    use_teacher_forcing=use_teacher_forcing,
    n_previous_observations=n_previous_observations
)


def mean_total_loss(losses):
    return mean([loss.total for loss in losses])


if __name__ == '__main__':
    # Setup model and game
    run_name = generate_run_name(f'{LOGS_DIR}/{game_name}', game_prefix, match_arbitrary_suffixes=True)
    print(f"Starting run: {run_name}")

    game = pyspiel.load_game(game_name)

    # Setup Model Manager
    observation_model_config = OpenSpielModelConfig(
        game,
        model_type_obs,
        [game.observation_tensor_shape()[0] * n_previous_observations],
        nn_width_obs,
        nn_depth_obs,
        weight_decay_obs,
        learning_rate_obs,
        omniscient_observer=False, output_shape=len(state_to_value_dict))
    observation_model_manager = OpenSpielCheckpointManager(game_name, f"{run_name}-observation_model")
    observation_model_manager.store_config(observation_model_config)

    game_model_config = OpenSpielModelConfig(
        game,
        model_type_game,
        get_observation_tensor_shape(game, omniscient_observer),
        nn_width_game,
        nn_depth_game,
        weight_decay_game,
        learning_rate_game,
        omniscient_observer=omniscient_observer)
    game_model_manager = OpenSpielCheckpointManager(game_name, f"{run_name}-game_model")
    game_model_manager.store_config(game_model_config)

    model_manager = {"game_model_manager": game_model_manager, "observation_model_manager": observation_model_manager}

    # Setup Replay Data Manager
    observation_data_manager = observation_model_manager.get_replay_data_manager()
    game_data_manager = game_model_manager.get_replay_data_manager()

    # Setup Evaluation Manager
    evaluation_manager = EvaluationManager(game, n_evaluations, evaluation_mcts_config)

    # Setup rating systems for evaluation
    elo_rating_system = EloRatingSystem(40)
    true_skill_rating_system = TrueSkillRatingSystem()
    rating_systems = [elo_rating_system, true_skill_rating_system]

    # Setup final training manager
    train_manager = AlphaOneTrainManager(game, model_manager, evaluation_manager, n_most_recent_train_samples,
                                         n_most_recent_valid_samples, rating_systems)

    if continue_training is not None:
        print(f"Continue training model {continue_training}")
        # Load models and replay data of run to continue
        observation_model_manager_continue = OpenSpielCheckpointManager(game_name, f"{continue_training}-observation_model")
        game_model_manager_continue = OpenSpielCheckpointManager(game_name, f"{continue_training}-game_model")

        observation_model_continue = observation_model_manager_continue.load_checkpoint(-1)
        observation_model_continue._path = train_manager.observation_model_current_best._path
        game_model_continue = game_model_manager_continue.load_checkpoint(-1)
        game_model_continue._path = train_manager.game_model_challenger._path

        replay_data_manager_observation_continue = observation_model_manager_continue.get_replay_data_manager()
        replay_data_manager_game_continue = game_model_manager_continue.get_replay_data_manager()

        replay_buffer_observation_continue = replay_data_manager_observation_continue.load_replays(-1)
        replay_buffer_game_continue = replay_data_manager_game_continue.load_replays(-1)

        # Overwrite train manager attributes with models and data of previous run
        train_manager.replay_buffer_observation = replay_buffer_observation_continue
        train_manager.replay_buffer_model = replay_buffer_game_continue
        train_manager.observation_model_challenger = observation_model_continue
        train_manager.observation_model_current_best = observation_model_continue
        train_manager.game_model_challenger = game_model_continue
        train_manager.game_model_current_best = game_model_continue

        train_manager.model_current_best = [
            train_manager.observation_model_current_best,
            train_manager.game_model_current_best]
        train_manager.model_challenger = [
            train_manager.observation_model_challenger,
            train_manager.game_model_challenger]


    print("Observation Model: Num variables:", train_manager.observation_model_challenger.num_trainable_variables)
    train_manager.observation_model_challenger.print_trainable_variables()
    print("")
    print("Game Model: Num variables:", train_manager.game_model_challenger.num_trainable_variables)
    train_manager.game_model_challenger.print_trainable_variables()

    observation_tensorboard = TensorboardLogger(f"{LOGS_DIR}/{game_name}/{run_name}-observation_model")
    observation_tensorboard.log_hyperparameters(hyperparameters)

    game_tensorboard = TensorboardLogger(f"{LOGS_DIR}/{game_name}/{run_name}-game_model")
    game_tensorboard.log_hyperparameters(hyperparameters)

    iteration = 0
    for iteration in range(1, n_iterations + 1):
        print(f"Iteration {iteration}")

        # 1 Generate training data with current best model
        new_train_observation_samples, \
        new_valid_observation_samples, \
        new_train_game_samples, \
        new_valid_game_samples = train_manager.generate_training_data(n_games_train, n_games_valid, mcts_config,
                                                                      use_teacher_forcing=use_teacher_forcing)
        print(
            f'  - Generated {len(new_train_observation_samples)} additional training observation samples and {len(new_valid_observation_samples)} additional validation observation samples')
        print(
            f'  - Generated {len(new_train_game_samples)} additional training game samples and {len(new_valid_game_samples)} additional validation game samples')
        observation_tensorboard.log_scalar("n_training_observation_samples",
                                           train_manager.replay_buffer_observation.get_total_samples(), iteration)
        game_tensorboard.log_scalar("n_training_game_samples", train_manager.replay_buffer_model.get_total_samples(),
                                    iteration)

        # 2 Repeatedly sample from training set and update weights on current model
        train_observation_losses, \
        valid_observation_losses, \
        train_game_losses, \
        valid_game_losses = train_manager.train_model(n_train_steps_obs, n_train_steps_game, n_valid_steps, batch_size,
                                                      weight_decay_obs, weight_decay_game)
        print(
            f'  - Training Observation Model: {mean_total_loss(train_observation_losses[:int(len(train_observation_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_observation_losses[int(len(train_observation_losses) / 4):int(2 * len(train_observation_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_observation_losses[int(2 * len(train_observation_losses) / 4):int(3 * len(train_observation_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_observation_losses[int(3 * len(train_observation_losses) / 4):]):.2f}')

        print(f'  - Training Game Model: {mean_total_loss(train_game_losses[:int(len(train_game_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_game_losses[int(len(train_game_losses) / 4):int(2 * len(train_game_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_game_losses[int(2 * len(train_game_losses) / 4):int(3 * len(train_game_losses) / 4)]):.2f} \
                -> {mean_total_loss(train_game_losses[int(3 * len(train_game_losses) / 4):]):.2f}')

        observation_tensorboard.log_scalars("Loss", {
            "total/train": mean([loss.total for loss in train_observation_losses]),
            "policy/train": mean([loss.policy for loss in train_observation_losses]),
            "value/train": mean([loss.value for loss in train_observation_losses]),
            "total/valid": mean([loss.total for loss in valid_observation_losses]),
            "policy/valid": mean([loss.policy for loss in valid_observation_losses]),
            "value/valid": mean([loss.value for loss in valid_observation_losses])
        }, iteration)

        game_tensorboard.log_scalars("Loss", {
            "total/train": mean([loss.total for loss in train_game_losses]),
            "policy/train": mean([loss.policy for loss in train_game_losses]),
            "value/train": mean([loss.value for loss in train_game_losses]),
            "total/valid": mean([loss.total for loss in valid_game_losses]),
            "policy/valid": mean([loss.policy for loss in valid_game_losses]),
            "value/valid": mean([loss.value for loss in valid_game_losses])
        }, iteration)

        challenger_win_rate, challenger_policies, match_outcomes, challenger_average_reward = train_manager.evaluate_challenger_model()

        player_name_current_best = train_manager.get_player_name_current_best()
        player_name_challenger = train_manager.get_player_name_challenger()

        true_skill_rating_system.update_ratings(match_outcomes)
        elo_rating_system.update_ratings(match_outcomes)
        print(
            f"  - Ratings current best: {true_skill_rating_system.get_rating(player_name_current_best)}, {elo_rating_system.get_rating(player_name_current_best):0.3f}")
        print(
            f"  - Ratings challenger: {true_skill_rating_system.get_rating(player_name_challenger)}, {elo_rating_system.get_rating(player_name_challenger):0.3f}")

        game_tensorboard.log_scalars("elo_rating", {
            "current_best": elo_rating_system.get_rating(player_name_current_best),
            "challenger": elo_rating_system.get_rating(player_name_challenger)
        }, iteration)
        observation_tensorboard.log_scalars("elo_rating", {
            "current_best": elo_rating_system.get_rating(player_name_current_best),
            "challenger": elo_rating_system.get_rating(player_name_challenger)
        }, iteration)

        game_tensorboard.log_scalars("true_skill_rating", {
            "current_best": true_skill_rating_system.get_rating(player_name_current_best).mu,
            "challenger": true_skill_rating_system.get_rating(player_name_challenger).mu
        }, iteration)

        observation_tensorboard.log_scalars("true_skill_rating", {
            "current_best": true_skill_rating_system.get_rating(player_name_current_best).mu,
            "challenger": true_skill_rating_system.get_rating(player_name_challenger).mu
        }, iteration)

        print(
            f'  - Challenger won {int(round(challenger_win_rate * n_evaluations))}/{n_evaluations} games ({challenger_win_rate:.2%} win rate)')
        game_tensorboard.log_scalar("challenger_win_rate", challenger_win_rate, iteration)

        observation_tensorboard.log_scalar("challenger_win_rate", challenger_win_rate, iteration)
        observation_tensorboard.log_scalar("challenger_average_reward", challenger_average_reward, iteration)
        game_tensorboard.log_scalar("challenger_win_rate", challenger_win_rate, iteration)
        game_tensorboard.log_scalar("challenger_average_reward", challenger_average_reward, iteration)

        # 3 Evaluate trained model against current best model
        train_manager.replace_model_with_challenger(challenger_win_rate, win_ratio_needed, challenger_average_reward,
                                                    average_reward_needed)
        if win_ratio_needed is not None:
            if challenger_win_rate > win_ratio_needed:
                print(
                    f"  - Model at iteration {iteration} supersedes previous model ({challenger_win_rate:.2%} win rate)")
        elif average_reward_needed is not None:
            if challenger_average_reward > average_reward_needed:
                print(
                    f"  - Model at iteration {iteration} supersedes previous model ({challenger_average_reward:.2f} average reward)")

        observation_tensorboard.log_scalar("best_model_generation", player_name_current_best, iteration)
        game_tensorboard.log_scalar("best_model_generation", player_name_current_best, iteration)

        if iteration % store_replays_every == 0:
            print("Replay buffer stored")
            observation_data_manager.store_replays(train_manager.replay_buffer_observation, iteration)
            game_data_manager.store_replays(train_manager.replay_buffer_model, iteration)

        game_tensorboard.flush()
        observation_tensorboard.flush()

    observation_data_manager.store_replays(train_manager.replay_buffer_observation, iteration)
    game_data_manager.store_replays(train_manager.replay_buffer_model, iteration)