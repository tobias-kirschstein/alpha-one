import math
from statistics import mean

import pyspiel
import ray

from alpha_one.metrics import EloRatingSystem, TrueSkillRatingSystem, calculate_entropy
from alpha_one.model.config import OpenSpielModelConfig
from alpha_one.model.evaluation import ParallelEvaluationManager, EvaluationManager
from alpha_one.model.model_manager import OpenSpielModelManager
from alpha_one.train import AlphaZeroTrainManager
from alpha_one.utils.logging import TensorboardLogger
from alpha_one.utils.mcts import MCTSConfig
from env import MODEL_SAVES_DIR, LOGS_DIR



NUM_CPUS = 1

# Training hyperparameters
game_name = 'leduc_poker'
game_prefix = 'LP'

model_saves_path = f'{MODEL_SAVES_DIR}/{game_name}'
tensorboard_log_dir = f'{LOGS_DIR}/{game_name}'

n_iterations = 10  # How often the whole procedure is repeated. Also corresponds to the number of evaluations

# Train samples generation
n_games_train = 300  # How many new states will be generated by the best model via self-play for training (Training set size delta). Has to be larger than batch_size
n_games_valid = 300

# Model update
n_most_recent_train_samples = 50000  # Among which training samples to choose to train current model
n_most_recent_valid_samples = 10000
n_train_steps = 100  # After how many gradient updates the new model tries to beat the current best
n_valid_steps = 100
batch_size = 512

# Evaluation
n_evaluations = 100  # How many games should be played to measure which model is better
evaluation_strategy = 'mcts'  # 'best_response'
win_ratio_needed = 0.55  # Minimum win ratio that the challenger model needs in order to supersede the current best model

# MCTS config
UCT_C = math.sqrt(2)
max_mcts_simulations = 25

policy_epsilon = None  # 0.25            # What noise epsilon to use
policy_alpha = None  # 1                 # What dirichlet noise alpha to use

temperature = 0.5
temperature_drop = 5

imperfect_info = True

mcts_config = MCTSConfig(UCT_C, max_mcts_simulations, temperature, temperature_drop, policy_epsilon, policy_alpha, 
                        imperfect_info)
evaluation_mcts_config = MCTSConfig(UCT_C, max_mcts_simulations, 0, None, None, None,
                                   imperfect_info)

# Model Hyperparameters
model_type = 'mlp'
nn_width = 64
nn_depth = 2
weight_decay = 1e-5
learning_rate = 0.001

hyperparameters = dict(
    game_name=game_name,
    n_iterations=n_iterations,

    n_games_train=n_games_train,
    n_games_valid=n_games_valid,

    n_most_recent_train_samples=n_most_recent_train_samples,
    n_most_recent_valid_samples=n_most_recent_valid_samples,
    n_train_steps=n_train_steps,
    n_valid_steps=n_valid_steps,
    batch_size=batch_size,

    n_evaluations=n_evaluations,
    win_ratio_needed=win_ratio_needed,

    model_type=model_type,
    nn_width=nn_width,
    nn_depth=nn_depth,
    weight_decay=weight_decay,
    learning_rate=learning_rate,

    **mcts_config
)



def mean_total_loss(losses):
    return mean([loss.total for loss in losses])



if __name__ == '__main__':    
    if not ray.is_initialized() and NUM_CPUS > 1:
        ray.init(num_cpus=NUM_CPUS)

    # Setup model and game
    model_manager = OpenSpielModelManager(game_name, game_prefix)
    # Setup model and game
    model_manager = OpenSpielModelManager(game_name, game_prefix)
    checkpoint_manager = model_manager.new_run()
    run_name = checkpoint_manager.get_run_name()
    print("#===============================")
    print(f"# Starting run: {run_name}")
    print("#===============================")

    game = pyspiel.load_game(game_name)

    # Setup Model Manager
    model_config = OpenSpielModelConfig(game, model_type, nn_width, nn_depth, weight_decay, learning_rate)
    checkpoint_manager.store_config(model_config)

    if ray.is_initialized() and NUM_CPUS > 1:
        evaluation_manager = ParallelEvaluationManager(game, checkpoint_manager, n_evaluations, evaluation_mcts_config)
    else:
        evaluation_manager = EvaluationManager(game, n_evaluations, evaluation_mcts_config)

    # Setup rating systems for evaluation
    elo_rating_system = EloRatingSystem(40)
    true_skill_rating_system = TrueSkillRatingSystem()
    rating_systems = [elo_rating_system, true_skill_rating_system]

    train_manager = AlphaZeroTrainManager(game, checkpoint_manager, evaluation_manager, n_most_recent_train_samples,
                                              n_most_recent_valid_samples, rating_systems)

    print("Num variables:", train_manager.model_challenger.num_trainable_variables)
    train_manager.model_challenger.print_trainable_variables()

    # Setup logging
    tensorboard = TensorboardLogger(f"{tensorboard_log_dir}/{run_name}")
    tensorboard.log_hyperparameters(hyperparameters)
    for iteration in range(1, n_iterations + 1):
        print(f"Iteration {iteration}")

        # 1 Generate training data with current best model
        new_train_samples, new_valid_samples = train_manager.generate_training_data(n_games_train, n_games_valid,
                                                                                        mcts_config)


        print(f'  - Generated {len(new_train_samples)} additional training samples and {len(new_valid_samples)} additional validation samples')
        tensorboard.log_scalar("n_training_samples", train_manager.replay_buffer.get_total_samples(), iteration)
        train_losses, valid_losses = train_manager.train_model(n_train_steps, n_valid_steps, batch_size, weight_decay)

        print(f'  - Training: {mean_total_loss(train_losses[:int(len(train_losses) / 4)]):.2f}                 -> {mean_total_loss(train_losses[int(len(train_losses) / 4):int(2 * len(train_losses) / 4)]):.2f}                 -> {mean_total_loss(train_losses[int(2 * len(train_losses) / 4):int(3 * len(train_losses) / 4)]):.2f}                 -> {mean_total_loss(train_losses[int(3 * len(train_losses) / 4):]):.2f}')
        tensorboard.log_scalars("Loss", {
            "total/train": mean([loss.total for loss in train_losses]),
            "policy/train": mean([loss.policy for loss in train_losses]),
            "value/train": mean([loss.value for loss in train_losses]),
            "total/valid": mean([loss.total for loss in valid_losses]),
            "policy/valid": mean([loss.policy for loss in valid_losses]),
            "value/valid": mean([loss.value for loss in valid_losses])
        }, iteration)

        # 3 Evaluate trained model against current best model
        challenger_win_rate, challenger_policies, match_outcomes = train_manager.evaluate_challenger_model()
        player_name_current_best = train_manager.get_player_name_current_best()
        player_name_challenger = train_manager.get_player_name_challenger()

        true_skill_rating_system.update_ratings(match_outcomes)
        elo_rating_system.update_ratings(match_outcomes)
        print(
            f"  - Ratings current best: {true_skill_rating_system.get_rating(player_name_current_best)}, {elo_rating_system.get_rating(player_name_current_best):0.3f}")
        print(
            f"  - Ratings challenger: {true_skill_rating_system.get_rating(player_name_challenger)}, {elo_rating_system.get_rating(player_name_challenger):0.3f}")
        tensorboard.log_scalars("elo_rating", {
            "current_best": elo_rating_system.get_rating(player_name_current_best),
            "challenger": elo_rating_system.get_rating(player_name_challenger)
        }, iteration)
        tensorboard.log_scalars("true_skill_rating", {
            "current_best": true_skill_rating_system.get_rating(player_name_current_best).mu,
            "challenger": true_skill_rating_system.get_rating(player_name_challenger).mu
        }, iteration)

        print(
            f'  - Challenger won {int(round(challenger_win_rate * n_evaluations))}/{n_evaluations} games ({challenger_win_rate:.2%} win rate)')
        tensorboard.log_scalar("challenger_win_rate", challenger_win_rate, iteration)


        # 4 Replace current best model with challenger model if it is better
        train_manager.replace_model_with_challenger(challenger_win_rate, win_ratio_needed)
        if challenger_win_rate > win_ratio_needed:
            print(f"  - Model at iteration {iteration} supersedes previous model ({challenger_win_rate:.2%} win rate)")

        challenger_entropy = calculate_entropy(challenger_policies)
        print(f"  - Challenger entropy: {challenger_entropy:0.3f}")
        label_entropy = calculate_entropy([sample.policy for sample in new_train_samples])
        print(f"  - Label entropy: {label_entropy:0.3f}")

        tensorboard.log_scalars("entropy", {
            "current_best": label_entropy,
            "challenger": challenger_entropy}, iteration)
        tensorboard.log_scalar("best_model_generation", player_name_current_best, iteration)

        tensorboard.flush()