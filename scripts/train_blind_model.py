"""
Trains a so-called blind model that tries to imitate the predictions of an AlphaZero/AlphaOne model by simply predicting
the MCTS policy directly at the current observation. Takes as input the observation (history) of the current player
and directly outputs value and policy estimates.
"""

from statistics import mean

from alpha_one.data.replay import ReplayDataManager
from alpha_one.model.config import OpenSpielModelConfig
from alpha_one.model.model_manager import OpenSpielModelManager, OpenSpielCheckpointManager
import pyspiel

from alpha_one.utils.logging import TensorboardLogger
from env import LOGS_DIR

# =========================================================================
# Settings
# =========================================================================

game_name = 'kuhn_poker'
run_name = 'KP-local-18'  # On which run this blind model should be based on
replay_iteration = -1  # What version of generated replay data should be used (usually this will be the most recent)

# Training settings
n_iterations = 100
n_train_steps = 100
batch_size = 256
n_most_recent = 500  # Only use the most recent generated train data

# Model settings
model_type = 'mlp'
nn_width = 128
nn_depth = 4
weight_decay = 1e-5
learning_rate = 1e-5

# =========================================================================
# END Settings
# =========================================================================

game = pyspiel.load_game(game_name)

replay_data_manager = ReplayDataManager(OpenSpielCheckpointManager(game_name, run_name).get_model_store_path())
replay_buffer = replay_data_manager.load_replays()

blind_model_manager = OpenSpielModelManager(game_name, f"{run_name}-blind").new_run()

blind_model_config = OpenSpielModelConfig(game,
                                          model_type=model_type,
                                          input_shape=replay_buffer.data[0].observation['player_observation'].shape,
                                          nn_width=nn_width,
                                          nn_depth=nn_depth,
                                          weight_decay=weight_decay,
                                          learning_rate=learning_rate)

blind_model_manager.store_config(blind_model_config)
blind_model = blind_model_manager.build_model(blind_model_config)

tensorboard_blind = TensorboardLogger(f"{LOGS_DIR}/{game_name}/{blind_model_manager.get_run_name()}")

for iteration in range(n_iterations):
    blind_losses = []
    for _ in range(n_train_steps):
        sampled_train_inputs = replay_buffer.sample(batch_size, 'player_observation', n_most_recent=500)
        loss = blind_model.update(sampled_train_inputs)
        blind_losses.append(loss)
    blind_model_manager.store_checkpoint(blind_model, iteration)

    tensorboard_blind.log_scalars("Loss", {
        "total/train": mean([loss.total for loss in blind_losses]),
        "policy/train": mean([loss.policy for loss in blind_losses]),
        "value/train": mean([loss.value for loss in blind_losses])
    }, iteration)
    tensorboard_blind.flush()
