import math
from statistics import mean

import numpy as np
import pyspiel
from open_spiel.python.algorithms.alpha_zero import model as model_lib

from alpha_one.game.buffer import ReplayBuffer
from alpha_one.metrics import MatchOutcome, EloRatingSystem, TrueSkillRatingSystem, calculate_entropy
from alpha_one.model.model_manager import OpenSpielModelManager
from alpha_one.model.config import OpenSpielModelConfig
from alpha_one.utils.logging import TensorboardLogger, generate_run_name
from alpha_one.utils.mcts import initialize_bot, play_one_game
from env import MODEL_SAVES_DIR, LOGS_DIR

# Training hyperparameters
game_name = 'connect_four'
game_prefix = 'C4'

model_saves_path = f'{MODEL_SAVES_DIR}/{game_name}'
tensorboard_log_dir = f'{LOGS_DIR}/{game_name}'

UCT_C = math.sqrt(2)
max_mcts_simulations = 100

n_iterations = 100  # How often the whole procedure is repeated. Also corresponds to the number of evaluations

# Train samples generation
n_new_train_samples = 1000  # How many new states will be generated by the best model via self-play for training (Training set size delta). Has to be larger than batch_size

# Model update
n_most_recent_train_samples = 50000  # Among which training samples to choose to train current model
n_train_steps = 50  # After how many gradient updates the new model tries to beat the current best
batch_size = 256

# Evaluation
n_evaluations = 50  # How many games should be played to measure which model is better
evaluation_strategy = 'mcts'  # 'best_response'
win_ratio_needed = 0.55  # Minimum win ratio that the challenger model needs in order to supersede the current best model

policy_epsilon = None  # 0.25            # What noise epsilon to use
policy_alpha = None  # 1                 # What dirichlet noise alpha to use

temperature = 1
temperature_drop = 10

# Model Hyperparameters
model_type = 'mlp'
nn_width = 64
nn_depth = 4
weight_decay = 1e-5
learning_rate = 5e-4

hyperparameters = dict(
    game_name=game_name,
    UCT_C=UCT_C,
    max_mcts_simulations=max_mcts_simulations,
    n_iterations=n_iterations,

    n_new_train_samples=n_new_train_samples,

    n_most_recent_train_samples=n_most_recent_train_samples,
    n_train_steps=n_train_steps,
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
    learning_rate=learning_rate
)


def generate_training_data():
    n_new_states = 0
    train_samples = []
    while True:
        bot = initialize_bot(game, model_current_best, UCT_C, max_mcts_simulations, policy_epsilon, policy_alpha)
        trajectory = play_one_game(game, [bot, bot], temperature, temperature_drop)
        p1_outcome = trajectory.get_final_reward(0)
        new_train_states = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, value=p1_outcome)
                            for s in trajectory.states]
        replay_buffer.extend(new_train_states)
        train_samples.extend(new_train_states)
        n_new_states += len(trajectory)
        if n_new_states > n_new_train_samples:
            break
    return train_samples


def train_model():
    losses = []
    for _ in range(n_train_steps):
        loss = model.update(replay_buffer.sample(batch_size))
        losses.append(loss)
    return losses


def evaluate_challenger_model(model_challenger, model_current_best):
    challenger_results = []
    challenger_policies = []
    match_outcomes = []
    for _ in range(n_evaluations):
        if evaluation_strategy == 'mcts':
            mcts_bot_best_model = initialize_bot(game, model_current_best, UCT_C, max_mcts_simulations, policy_epsilon,
                                                 policy_alpha)
            mcts_bot_challenger = initialize_bot(game, model_challenger, UCT_C, max_mcts_simulations, policy_epsilon,
                                                 policy_alpha)

        player_id_challenger = np.random.choice([0, 1])  # ensure that each model will play as each player
        bots = [mcts_bot_challenger, mcts_bot_best_model] if player_id_challenger == 0 else [mcts_bot_best_model,
                                                                                             mcts_bot_challenger]

        trajectory = play_one_game(game, bots, temperature, temperature_drop)
        challenger_policies.extend([s.policy for s in trajectory.get_player_states(player_id_challenger)])

        challenger_reward = trajectory.get_final_reward(player_id_challenger)
        challenger_results.append(challenger_reward)
        match_outcomes.append(
            MatchOutcome.win(player_name_challenger, player_name_current_best)
            if challenger_reward == 1 else
            MatchOutcome.defeat(player_name_challenger, player_name_current_best))

    n_challenger_wins = (np.array(challenger_results) == 1).sum()
    challenger_win_rate = n_challenger_wins / n_evaluations
    return challenger_win_rate, challenger_policies, match_outcomes


def mean_total_loss(losses):
    return mean([loss.total for loss in losses])


def copy_and_create_checkpoint(iteration):
    model_manager.store(model, iteration)
    return model_manager.load(iteration)


if __name__ == '__main__':
    run_name = generate_run_name(f'{LOGS_DIR}/{game_name}', game_prefix)
    print(f"Starting run: {run_name}")

    # Setup model and game
    game = pyspiel.load_game(game_name)

    model_config = OpenSpielModelConfig(game, model_type, nn_width, nn_depth, weight_decay, learning_rate)
    model_manager = OpenSpielModelManager(f"{game_name}/{run_name}")
    model_manager.store_config(model_config)
    model = model_manager.build_model(model_config)
    
    print("Num variables:", model.num_trainable_variables)
    model.print_trainable_variables()
    model_current_best = copy_and_create_checkpoint(0)

    # Setup rating systems for evaluation
    elo_rating_system = EloRatingSystem(40)
    true_skill_rating_system = TrueSkillRatingSystem()

    player_name_current_best = 0
    player_name_challenger = 1

    # Setup logging
    tensorboard = TensorboardLogger(f"{tensorboard_log_dir}/{run_name}")
    tensorboard.log_hyperparameters(hyperparameters)

    # Training loop
    replay_buffer = ReplayBuffer(n_most_recent_train_samples)
    for iteration in range(1, n_iterations + 1):
        print(f"Iteration {iteration}")

        # 1 Generate training data with current best model
        new_train_samples = generate_training_data()
        print(f'  - Generated {len(new_train_samples)} additional training samples')
        tensorboard.log_scalar("n_training_samples", replay_buffer.get_total_samples(), iteration)

        # 2 Repeatedly sample from training set and update weights on current model
        losses = train_model()
        print(f'  - Training: {mean_total_loss(losses[:int(len(losses) / 4)]):.2f} \
                -> {mean_total_loss(losses[int(len(losses) / 4):int(2 * len(losses) / 4)]):.2f} \
                -> {mean_total_loss(losses[int(2 * len(losses) / 4):int(3 * len(losses) / 4)]):.2f} \
                -> {mean_total_loss(losses[int(3 * len(losses) / 4):]):.2f}')
        tensorboard.log_scalar("Loss", mean_total_loss(losses), iteration)

        # 3 Evaluate trained model against current best model
        challenger_win_rate, challenger_policies, match_outcomes = evaluate_challenger_model(model, model_current_best)

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
        if challenger_win_rate > win_ratio_needed:
            print(f"  - Model at iteration {iteration} supersedes previous model ({challenger_win_rate:.2%} win rate)")
            model_current_best = copy_and_create_checkpoint(iteration)
            player_name_current_best = player_name_challenger

        challenger_entropy = calculate_entropy(challenger_policies)
        print(f"  - Challenger entropy: {challenger_entropy:0.3f}")
        label_entropy = calculate_entropy([sample.policy for sample in new_train_samples])
        print(f"  - Label entropy: {label_entropy:0.3f}")

        tensorboard.log_scalars("entropy", {
            "current_best": label_entropy,
            "challenger": challenger_entropy}, iteration)
        tensorboard.log_scalar("best_model_generation", player_name_current_best, iteration)

        tensorboard.flush()

        player_name_challenger += 1
