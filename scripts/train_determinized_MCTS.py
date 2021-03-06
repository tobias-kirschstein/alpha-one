#importing libraries
import numpy as np
import pyspiel
import math
import ray
from statistics import mean


from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms import mcts

from alpha_one.metrics import MatchOutcome, EloRatingSystem, TrueSkillRatingSystem, calculate_entropy
from alpha_one.game.trajectory import GameTrajectory
from alpha_one.game.buffer import ReplayBuffer
from alpha_one.game.observer import get_observation_tensor_shape
from alpha_one.utils.mcts import initialize_bot, compute_mcts_policy, play_one_game, mcts_inference
from alpha_one.utils.logging import TensorboardLogger, generate_run_name
from alpha_one.model.model_manager import OpenSpielCheckpointManager, OpenSpielModelManager
from alpha_one.model.evaluation import EvaluationManager, ParallelEvaluationManager
from alpha_one.model.config import OpenSpielModelConfig
from alpha_one.train import AlphaZeroTrainManager, MCTSConfig
from alpha_one.data.replay import ReplayDataManager
from env import MODEL_SAVES_DIR, LOGS_DIR


# The training is done in a similar way as AlphaZero.
# First the NN is trained in a cheating manner same as alphazero
# While evaluating, the trained cheating NN is used to guide the MCTS tree in D-MCTS algorithm when evaluating the states in information set
# See AlphaZeroTrainManager and utilis/determinized_mcts for debugging


game_name = 'leduc_poker'
game_prefix = 'LP-local'


n_iterations = 50                     # How often the whole procedure is repeated. Also corresponds to the number of evaluations

# Train samples generation
n_games_train = 100             # How many new states will be generated by the best model via self-play for training (Training set size delta). Has to be larger than batch_size
n_games_valid = 10
store_replays_every = 10

# Model update
n_most_recent_train_samples = 50000    # Among which training samples to choose to train current model
n_most_recent_valid_samples = 50000
n_train_steps = 100                     # After how many gradient updates the new model tries to beat the current best
n_valid_steps = 10
batch_size = 8

# Evaluation
n_evaluations = 100                    # How many games should be played to measure which model is better
evaluation_strategy = 'mcts'           # 'best_response'
win_ratio_needed = None #0.55                # Minimum win ratio that the challenger model needs in order to supersede the current best model
average_reward_needed = 0.2            # Minimum average reward over current best model that the challenger model needs in order to supersede the current best model. Mutually exclusive with win_ratio_needed

# MCTS config
UCT_C = 5
max_mcts_simulations = 10

policy_epsilon = None #0.25            # What noise epsilon to use
policy_alpha = None #1                 # What dirichlet noise alpha to use

temperature = 1
temperature_drop = 10

omniscient_observer = True
use_reward_policy = True
determinized_MCTS = True

assert win_ratio_needed is None and average_reward_needed is not None \
       or win_ratio_needed is not None and average_reward_needed is None, \
    f"win_ratio_needed and average_reward_needed are mutually exclusive"

mcts_config = MCTSConfig(UCT_C, max_mcts_simulations, temperature, temperature_drop, policy_epsilon, policy_alpha, omniscient_observer=omniscient_observer, use_reward_policy=use_reward_policy)
evaluation_mcts_config = MCTSConfig(UCT_C, max_mcts_simulations, 0, None, None, None, determinized_MCTS=determinized_MCTS, omniscient_observer=omniscient_observer, use_reward_policy=use_reward_policy)


# Model Hyperparameters
model_type = 'mlp'
nn_width = 128
nn_depth = 4
weight_decay = 1e-5
learning_rate = 1e-3


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
    n_train_steps=n_train_steps,
    n_valid_steps=n_valid_steps,
    batch_size=batch_size,
    
    n_evaluations=n_evaluations,
    win_ratio_needed=win_ratio_needed,
    
    policy_epsilon=policy_epsilon,
    policy_alpha=policy_alpha,
    
    temperature=temperature,
    temperature_drop=temperature_drop,
    
    model_type=model_type,
    nn_width=nn_width,
    nn_depth=nn_depth,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    determinized_MCTS = determinized_MCTS,
    omniscient_observer=omniscient_observer,
    use_reward_policy=use_reward_policy
)


def mean_total_loss(losses):
    return mean([loss.total for loss in losses])


# Setup model and game
run_name = generate_run_name(f'{LOGS_DIR}/{game_name}', game_prefix)
print(f"Starting run: {run_name}")

game = pyspiel.load_game(game_name)

# Setup Model Manager
model_config = OpenSpielModelConfig(
    game, 
    model_type, 
    get_observation_tensor_shape(game, omniscient_observer), 
    nn_width, 
    nn_depth, 
    weight_decay, 
    learning_rate,
    omniscient_observer=omniscient_observer)
model_manager = OpenSpielCheckpointManager(game_name, run_name)
model_manager.store_config(model_config)

# Setup Evaluation Manager
if ray.is_initialized():
    evaluation_manager = ParallelEvaluationManager(game, model_manager, n_evaluations, evaluation_mcts_config)
else:
    evaluation_manager = EvaluationManager(game, n_evaluations, evaluation_mcts_config)
    
    
# Setup rating systems for evaluation
elo_rating_system = EloRatingSystem(40)
true_skill_rating_system = TrueSkillRatingSystem()
rating_systems = [elo_rating_system, true_skill_rating_system]

# Setup final training manager
train_manager = AlphaZeroTrainManager(game, model_manager, evaluation_manager, n_most_recent_train_samples, n_most_recent_valid_samples, rating_systems)

print("Num variables:", train_manager.model_challenger.num_trainable_variables)
train_manager.model_challenger.print_trainable_variables()


tensorboard = TensorboardLogger(f"{LOGS_DIR}/{game_name}/{run_name}")
tensorboard.log_hyperparameters(hyperparameters)


# Training loop
for iteration in range(1, n_iterations + 1):
    print(f"Iteration {iteration}")
    
    # 1 Generate training data with current best model
    new_train_samples, new_valid_samples = train_manager.generate_training_data(n_games_train, n_games_valid, mcts_config)
    print(f'  - Generated {len(new_train_samples)} additional training samples and {len(new_valid_samples)} additional validation samples')
    tensorboard.log_scalar("n_training_samples", train_manager.replay_buffer.get_total_samples(), iteration)
    
    # 2 Repeatedly sample from training set and update weights on current model
    train_losses, valid_losses = train_manager.train_model(n_train_steps, n_valid_steps, batch_size, weight_decay)
    print(f'  - Training: {mean_total_loss(train_losses[:int(len(train_losses)/4)]):.2f}             -> {mean_total_loss(train_losses[int(len(train_losses)/4):int(2 * len(train_losses)/4)]):.2f}             -> {mean_total_loss(train_losses[int(2 * len(train_losses)/4):int(3 * len(train_losses)/4)]):.2f}             -> {mean_total_loss(train_losses[int(3 * len(train_losses)/4):]):.2f}')
    tensorboard.log_scalars("Loss", {
        "total/train": mean([loss.total for loss in train_losses]),
        "policy/train": mean([loss.policy for loss in train_losses]),
        "value/train": mean([loss.value for loss in train_losses]),
        "total/valid": mean([loss.total for loss in valid_losses]),
        "policy/valid": mean([loss.policy for loss in valid_losses]),
        "value/valid": mean([loss.value for loss in valid_losses])
    }, iteration)
    
    # 3 Evaluate trained model against current best model
    challenger_win_rate, challenger_policies, match_outcomes, challenger_average_reward = train_manager.evaluate_challenger_model()
    
    player_name_current_best = train_manager.get_player_name_current_best()
    player_name_challenger = train_manager.get_player_name_challenger()
    
    true_skill_rating_system.update_ratings(match_outcomes)
    elo_rating_system.update_ratings(match_outcomes)
    print(f"  - Ratings current best: {true_skill_rating_system.get_rating(player_name_current_best)}, {elo_rating_system.get_rating(player_name_current_best):0.3f}")
    print(f"  - Ratings challenger: {true_skill_rating_system.get_rating(player_name_challenger)}, {elo_rating_system.get_rating(player_name_challenger):0.3f}")
    tensorboard.log_scalars("elo_rating", {
        "current_best": elo_rating_system.get_rating(player_name_current_best),
        "challenger": elo_rating_system.get_rating(player_name_challenger)
    }, iteration)
    tensorboard.log_scalars("true_skill_rating", {
        "current_best": true_skill_rating_system.get_rating(player_name_current_best).mu,
        "challenger": true_skill_rating_system.get_rating(player_name_challenger).mu
    }, iteration)
    
    print(f'  - Challenger won {int(round(challenger_win_rate * n_evaluations))}/{n_evaluations} games ({challenger_win_rate:.2%} win rate)')
    tensorboard.log_scalar("challenger_win_rate", challenger_win_rate, iteration)
    tensorboard.log_scalar("challenger_average_reward", challenger_average_reward, iteration)
    
    # 4 Replace current best model with challenger model if it is better
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
        
    challenger_entropy = calculate_entropy(challenger_policies)
    print(f"  - Challenger entropy: {challenger_entropy:0.3f}")
    label_entropy = calculate_entropy([sample.policy for sample in new_train_samples])
    print(f"  - Label entropy: {label_entropy:0.3f}")
    
    tensorboard.log_scalars("entropy", {
        "current_best": label_entropy,
        "challenger": challenger_entropy}, iteration)
    tensorboard.log_scalar("best_model_generation", player_name_current_best, iteration)
    
    
    tensorboard.flush()




