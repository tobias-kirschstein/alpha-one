import collections
import copy
from typing import Union, List, Dict

import numpy as np
import ray
import tensorflow as tf
from open_spiel.python.algorithms.alpha_zero import model as model_lib

from alpha_one.game.buffer import ReplayBuffer
from alpha_one.metrics import cross_entropy, RatingSystem
from alpha_one.model.evaluation import EvaluationManager, ParallelEvaluationManager
from alpha_one.model.model_manager import CheckpointManager
from alpha_one.utils.mcts_II import initialize_bot_alphaone, play_one_game_alphaone, IIGMCTSConfig


class Losses(collections.namedtuple("Losses", "policy value l2")):
    """Losses from a training step."""

    @property
    def total(self):
        return self.policy + self.value + self.l2

    def __str__(self):
        return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
                "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

    def __add__(self, other):
        return Losses(self.policy + other.policy,
                      self.value + other.value,
                      self.l2 + other.l2)

    def __truediv__(self, n):
        return Losses(self.policy / n, self.value / n, self.l2 / n)


@ray.remote(num_returns=1)
def _generate_one_game_parallel(game, checkpoint_manager: Dict[str, CheckpointManager], mcts_config: IIGMCTSConfig,
                                use_teacher_forcing=False):
    # Tensorflow models are not pickeable. Hence, the worker has to load the model from the disk

    model_current_best = [checkpoint_manager["observation_model_manager"].load_checkpoint(-1),
                          checkpoint_manager["game_model_manager"].load_checkpoint(-1)]
    return _generate_one_game(game, model_current_best, mcts_config, use_teacher_forcing=use_teacher_forcing)


def _generate_one_game(game, model_current_best, mcts_config: IIGMCTSConfig, use_teacher_forcing=False):
    bot = initialize_bot_alphaone(game, model_current_best, mcts_config)

    trajectory_observation, trajectory_game = play_one_game_alphaone(game, [bot, bot], mcts_config,
                                                                     use_teacher_forcing=use_teacher_forcing)

    p1_outcome = trajectory_game.get_final_reward(0)
    p1_outcome = p1_outcome / game.max_utility() if p1_outcome > 0 else p1_outcome / -game.min_utility()

    new_observations = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, value=p1_outcome)
                        for s in trajectory_observation.states]

    new_states = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, value=p1_outcome)
                  for s in trajectory_game.states]

    return new_observations, new_states


class AlphaOneTrainManager:

    def __init__(self,
                 game,
                 checkpoint_manager: Dict[str, CheckpointManager],
                 evaluation_manager: Union[EvaluationManager, ParallelEvaluationManager],
                 replay_buffer_size,
                 replay_buffer_size_valid,
                 rating_systems: List[RatingSystem]):
        self.game = game
        self.checkpoint_manager = checkpoint_manager
        self.evaluation_manager = evaluation_manager
        self.replay_buffer_observation = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_valid_observation = ReplayBuffer(replay_buffer_size_valid)
        self.replay_buffer_model = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_valid_model = ReplayBuffer(replay_buffer_size_valid)
        self.rating_systems = rating_systems
        self.iteration = 1

        # self.player_name_current_best = 0
        # self.player_name_challenger = 1
        self.current_generation = 0

        self.observation_model_current_best = checkpoint_manager["observation_model_manager"].build_model()
        checkpoint_manager["observation_model_manager"].store_checkpoint(self.observation_model_current_best, 0)
        self.observation_model_challenger = checkpoint_manager["observation_model_manager"].load_checkpoint(0)

        self.game_model_current_best = checkpoint_manager["game_model_manager"].build_model()
        checkpoint_manager["game_model_manager"].store_checkpoint(self.game_model_current_best, 0)
        self.game_model_challenger = checkpoint_manager["game_model_manager"].load_checkpoint(0)

        self.model_current_best = [self.observation_model_current_best, self.game_model_current_best]
        self.model_challenger = [self.observation_model_challenger, self.game_model_challenger]

        self.use_parallelism = ray.is_initialized()
        if self.use_parallelism:
            print("AlphaZero Train manager will use parallelism")
        # self.omniscient_observer = checkpoint_manager.load_config().omniscient_observer

    def generate_training_data(self, n_games_train: int, n_games_valid: int, mcts_config: IIGMCTSConfig,
                               use_teacher_forcing=False):
        train_game_samples = []
        train_observation_samples = []

        valid_game_samples = []
        valid_observation_samples = []

        for _ in range(n_games_train):
            if self.use_parallelism:

                observation_samples, game_samples = _generate_one_game_parallel.remote(game=self.game,
                                                                                       checkpoint_manager=self.checkpoint_manager,
                                                                                       mcts_config=mcts_config,
                                                                                       use_teacher_forcing=use_teacher_forcing)
                train_observation_samples.append(observation_samples)
                train_game_samples.append(game_samples)
            else:
                observation_samples, game_samples = _generate_one_game(game=self.game,
                                                                       model_current_best=self.model_current_best,
                                                                       mcts_config=mcts_config,
                                                                       use_teacher_forcing=use_teacher_forcing)

                train_observation_samples.append(observation_samples)
                train_game_samples.append(game_samples)

        for _ in range(n_games_valid):
            if self.use_parallelism:
                observation_samples, game_samples = _generate_one_game_parallel.remote(game=self.game,
                                                                                       checkpoint_manager=self.checkpoint_manager,
                                                                                       mcts_config=mcts_config,
                                                                                       use_teacher_forcing=use_teacher_forcing)

                valid_observation_samples.append(observation_samples)
                valid_game_samples.append(game_samples)
            else:
                observation_samples, game_samples = _generate_one_game(game=self.game,
                                                                       model_current_best=self.model_current_best,
                                                                       mcts_config=mcts_config,
                                                                       use_teacher_forcing=use_teacher_forcing)

                valid_observation_samples.append(observation_samples)
                valid_game_samples.append(game_samples)

        if self.use_parallelism:
            train_observation_samples_ = ray.get(train_observation_samples)
            valid_observation_samples_ = ray.get(valid_observation_samples)

            train_game_samples_ = ray.get(train_game_samples)
            valid_game_samples_ = ray.get(valid_game_samples)

            train_game_samples = copy.deepcopy(train_game_samples_)
            valid_game_samples = copy.deepcopy(valid_game_samples_)

            train_observation_samples = copy.deepcopy(train_observation_samples_)
            valid_observation_samples = copy.deepcopy(valid_observation_samples_)

            del train_observation_samples_
            del valid_observation_samples_
            del train_game_samples_
            del valid_game_samples_

        train_observation_samples = [sample for batch in train_observation_samples for sample in batch]
        valid_observation_samples = [sample for batch in valid_observation_samples for sample in batch]

        train_game_samples = [sample for batch in train_game_samples for sample in batch]
        valid_game_samples = [sample for batch in valid_game_samples for sample in batch]

        self.replay_buffer_observation.extend(train_observation_samples)
        self.replay_buffer_valid_observation.extend(valid_observation_samples)

        self.replay_buffer_model.extend(train_game_samples)
        self.replay_buffer_valid_model.extend(valid_game_samples)

        return train_observation_samples, valid_observation_samples, train_game_samples, valid_game_samples

    def train_model(self, n_train_steps_obs, n_train_steps_game, n_valid_steps, batch_size, weight_decay_obs,
                    weight_decay_game):
        train_observation_losses = []
        valid_observation_losses = []

        train_game_losses = []
        valid_game_losses = []

        for _ in range(n_train_steps_obs):
            sampled_train_observation_inputs = self.replay_buffer_observation.sample(batch_size, None)
            loss = self.model_challenger[0].update(sampled_train_observation_inputs)
            train_observation_losses.append(loss)

        for _ in range(n_train_steps_game):
            sampled_train_game_inputs = self.replay_buffer_model.sample(batch_size, 'omniscient_observation')
            loss = self.model_challenger[1].update(sampled_train_game_inputs)
            train_game_losses.append(loss)

        for _ in range(n_valid_steps):
            valid_observation_samples = self.replay_buffer_valid_observation.sample(batch_size, None)

            values, policies = self.model_challenger[0].inference(
                [valid_sample.observation for valid_sample in valid_observation_samples],
                [valid_sample.legals_mask for valid_sample in valid_observation_samples])

            value_loss = np.mean((values - np.array([[sample.value] for sample in valid_observation_samples])) ** 2)
            policy_loss = np.mean(
                cross_entropy(np.array([sample.policy for sample in valid_observation_samples]), policies))
            reg_loss = sum([weight_decay_obs * np.linalg.norm(var.eval())
                            for var in tf.compat.v1.trainable_variables()
                            if "/bias:" not in var.name])
            valid_observation_losses.append(Losses(policy_loss, value_loss, reg_loss))

            valid_game_samples = self.replay_buffer_valid_model.sample(batch_size, 'omniscient_observation')

            values, policies = self.model_challenger[1].inference(
                [valid_sample.observation for valid_sample in valid_game_samples],
                [valid_sample.legals_mask for valid_sample in valid_game_samples])

            value_loss = np.mean((values - np.array([[sample.value] for sample in valid_game_samples])) ** 2)
            policy_loss = np.mean(cross_entropy(np.array([sample.policy for sample in valid_game_samples]), policies))
            reg_loss = sum([weight_decay_game * np.linalg.norm(var.eval())
                            for var in tf.compat.v1.trainable_variables()
                            if "/bias:" not in var.name])
            valid_game_losses.append(Losses(policy_loss, value_loss, reg_loss))

        return train_observation_losses, valid_observation_losses, train_game_losses, valid_game_losses

    def evaluate_challenger_model(self):
        if isinstance(self.evaluation_manager, ParallelEvaluationManager):
            # Hack: store challenger model as checkpoint '-2' as Tensorflow models cannot be pickled (and thus
            # cannot be sent to the Ray workers). Instead, the ray workers have to load the models from disk
            self.checkpoint_manager.store_checkpoint(self.model_challenger, -2)
            challenger_win_rate, trajectories, match_outcomes, challenger_average_reward = self.evaluation_manager.compare_models(
                -2, -1)
        else:
            challenger_win_rate, trajectories, match_outcomes, challenger_average_reward = self.evaluation_manager.compare_models(
                self.model_challenger, self.model_current_best)

        challenger_policies = [state.policy
                               for trajectory, player_id_challenger in trajectories
                               for state in trajectory.states
                               if state.current_player == player_id_challenger]
        player_name_mapping = {1: self.get_player_name_challenger(),
                               2: self.get_player_name_current_best()}
        match_outcomes = [match_outcome.with_renamed_players(player_name_mapping) for match_outcome in match_outcomes]

        return challenger_win_rate, challenger_policies, match_outcomes, challenger_average_reward

    def replace_model_with_challenger(self, challenger_win_rate: float, win_ratio_needed: float,
                                      challenger_average_reward: float, average_reward_needed: float):
        if win_ratio_needed is not None:
            supersedes = challenger_win_rate > win_ratio_needed
        elif average_reward_needed is not None:
            supersedes = challenger_average_reward > average_reward_needed

        if supersedes:
            self.checkpoint_manager["observation_model_manager"].store_checkpoint(self.model_challenger[0],
                                                                                  self.get_player_name_challenger())
            self.checkpoint_manager["game_model_manager"].store_checkpoint(self.model_challenger[1],
                                                                           self.get_player_name_challenger())

            self.model_current_best = [
                self.checkpoint_manager["observation_model_manager"].load_checkpoint(self.get_player_name_challenger()),
                self.checkpoint_manager["game_model_manager"].load_checkpoint(self.get_player_name_challenger())]

            ratings_challenger = [rating_system.get_rating(self.get_player_name_challenger())
                                  for rating_system
                                  in self.rating_systems]

            # self.player_name_current_best = self.player_name_challenger
            # self.player_name_challenger = iteration + 1
            self.current_generation += 1

            # Copy ratings of (previous) challenger such that new challenger will be identical
            for rating_system, rating_challenger in zip(self.rating_systems, ratings_challenger):
                rating_system.add_player(self.get_player_name_challenger(), rating_challenger)

    def get_player_name_current_best(self):
        return self.current_generation

    def get_player_name_challenger(self):
        return self.current_generation + 1
