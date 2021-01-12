import collections
import copy
from typing import Union, List

import numpy as np
import ray
import tensorflow as tf
from open_spiel.python.algorithms.alpha_zero import model as model_lib

from alpha_one.game.buffer import ReplayBuffer
from alpha_one.metrics import cross_entropy, RatingSystem
from alpha_one.model.evaluation import EvaluationManager, ParallelEvaluationManager
from alpha_one.model.model_manager import ModelManager
from alpha_one.utils.mcts import initialize_bot, play_one_game, MCTSConfig


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
def _generate_one_game_parallel(game, model_manager: ModelManager, mcts_config: MCTSConfig):
    # Tensorflow models are not pickeable. Hence, the worker has to load the model from the disk
    model_current_best = model_manager.load_model(-1)
    return _generate_one_game(game, model_current_best, mcts_config)


def _generate_one_game(game, model_current_best, mcts_config: MCTSConfig):
    bot = initialize_bot(game, model_current_best, mcts_config.uct_c,
                         mcts_config.max_mcts_simulations, mcts_config.policy_epsilon,
                         mcts_config.policy_alpha)
    trajectory = play_one_game(game, [bot, bot], mcts_config.temperature, mcts_config.temperature_drop)
    p1_outcome = trajectory.get_final_reward(0)
    new_states = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, value=p1_outcome)
                  for s in trajectory.states]
    return new_states


class AlphaZeroTrainManager:

    def __init__(self,
                 game,
                 model_manager: ModelManager,
                 evaluation_manager: Union[EvaluationManager, ParallelEvaluationManager],
                 replay_buffer_size,
                 replay_buffer_size_valid,
                 rating_systems: List[RatingSystem]):
        self.game = game
        self.model_manager = model_manager
        self.evaluation_manager = evaluation_manager
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_valid = ReplayBuffer(replay_buffer_size_valid)
        self.rating_systems = rating_systems
        self.iteration = 1

        # self.player_name_current_best = 0
        # self.player_name_challenger = 1
        self.current_generation = 0

        self.model_current_best = model_manager.build_model()
        model_manager.store_model(self.model_current_best, 0)
        self.model_challenger = model_manager.load_model(0)
        self.use_parallelism = ray.is_initialized()
        if self.use_parallelism:
            print("AlphaZero Train manager will use parallelism")

    def generate_training_data(self, n_games_train: int, n_games_valid: int, mcts_config: MCTSConfig):
        train_samples = []
        valid_samples = []

        for _ in range(n_games_train):
            if self.use_parallelism:
                train_samples.append(
                    _generate_one_game_parallel.remote(game=self.game, model_manager=self.model_manager,
                                                       mcts_config=mcts_config))
            else:
                train_samples.append(
                    _generate_one_game(game=self.game, model_current_best=self.model_current_best,
                                       mcts_config=mcts_config))

        for _ in range(n_games_valid):
            if self.use_parallelism:
                valid_samples.append(
                    _generate_one_game_parallel.remote(game=self.game, model_manager=self.model_manager,
                                                       mcts_config=mcts_config))
            else:
                valid_samples.append(
                    _generate_one_game(game=self.game, model_current_best=self.model_current_best,
                                       mcts_config=mcts_config))

        if self.use_parallelism:
            train_samples_ = ray.get(train_samples)
            valid_samples_ = ray.get(valid_samples)
            train_samples = copy.deepcopy(train_samples_)
            valid_samples = copy.deepcopy(valid_samples_)
            del train_samples_
            del valid_samples_

        train_samples = [sample for batch in train_samples for sample in batch]
        valid_samples = [sample for batch in valid_samples for sample in batch]

        self.replay_buffer.extend(train_samples)
        self.replay_buffer_valid.extend(valid_samples)

        return train_samples, valid_samples

        # while True:
        #     bot = initialize_bot(self.game, self.model_current_best, mcts_config.uct_c,
        #                          mcts_config.max_mcts_simulations, mcts_config.policy_epsilon,
        #                          mcts_config.policy_alpha)
        #     trajectory = play_one_game(self.game, [bot, bot], mcts_config.temperature, mcts_config.temperature_drop)
        #     p1_outcome = trajectory.get_final_reward(0)
        #     new_train_states = [model_lib.TrainInput(s.observation, s.legals_mask, s.policy, value=p1_outcome)
        #                         for s in trajectory.states]
        #     if len(train_samples) < n_new_train_samples:
        #         self.replay_buffer.extend(new_train_states)
        #         train_samples.extend(new_train_states)
        #     else:
        #         self.replay_buffer_valid.extend(new_train_states)
        #         valid_samples.extend(new_train_states)
        #         if len(valid_samples) > n_new_valid_samples:
        #             break
        # return train_samples, valid_samples

    def train_model(self, n_train_steps, n_valid_steps, batch_size, weight_decay):
        train_losses = []
        valid_losses = []

        for _ in range(n_train_steps):
            loss = self.model_challenger.update(self.replay_buffer.sample(batch_size))
            train_losses.append(loss)

        for _ in range(n_valid_steps):
            valid_samples = self.replay_buffer_valid.sample(batch_size)

            values, policies = self.model_challenger.inference(
                [valid_sample.observation for valid_sample in valid_samples],
                [valid_sample.legals_mask for valid_sample in valid_samples])

            value_loss = np.mean((values - np.array([[sample.value] for sample in valid_samples])) ** 2)
            policy_loss = np.mean(cross_entropy(np.array([sample.policy for sample in valid_samples]), policies))
            reg_loss = sum([weight_decay * np.linalg.norm(var.eval())
                            for var in tf.compat.v1.trainable_variables()
                            if "/bias:" not in var.name])
            valid_losses.append(Losses(policy_loss, value_loss, reg_loss))

        return train_losses, valid_losses

    def evaluate_challenger_model(self):
        if isinstance(self.evaluation_manager, ParallelEvaluationManager):
            # Hack: store challenger model as checkpoint '-2' as Tensorflow models cannot be pickled (and thus
            # cannot be sent to the Ray workers). Instead, the ray workers have to load the models from disk
            self.model_manager.store_model(self.model_challenger, -2)
            challenger_win_rate, trajectories, match_outcomes = self.evaluation_manager.compare_models(-2, -1)
        else:
            challenger_win_rate, trajectories, match_outcomes = self.evaluation_manager.compare_models(
                self.model_challenger, self.model_current_best)

        challenger_policies = [state.policy
                               for trajectory, player_id_challenger in trajectories
                               for state in trajectory.states
                               if state.current_player == player_id_challenger]
        player_name_mapping = {1: self.get_player_name_challenger(),
                               2: self.get_player_name_current_best()}
        match_outcomes = [match_outcome.with_renamed_players(player_name_mapping) for match_outcome in match_outcomes]

        return challenger_win_rate, challenger_policies, match_outcomes

        # challenger_results = []
        # challenger_policies = []
        # match_outcomes = []
        # for _ in range(n_evaluations):
        #     mcts_bot_best_model = initialize_bot(self.game, self.model_current_best, mcts_config.uct_c,
        #                                          mcts_config.max_mcts_simulations,
        #                                          mcts_config.policy_epsilon, mcts_config.policy_alpha)
        #     mcts_bot_challenger = initialize_bot(self.game, self.model_challenger, mcts_config.uct_c,
        #                                          mcts_config.max_mcts_simulations,
        #                                          mcts_config.policy_epsilon, mcts_config.policy_alpha)
        #
        #     player_id_challenger = np.random.choice([0, 1])  # ensure that each model will play as each player
        #     bots = [mcts_bot_challenger, mcts_bot_best_model] if player_id_challenger == 0 else [mcts_bot_best_model,
        #                                                                                          mcts_bot_challenger]
        #
        #     trajectory = play_one_game(self.game, bots, mcts_config.temperature, mcts_config.temperature_drop)
        #     challenger_policies.extend([s.policy for s in trajectory.get_player_states(player_id_challenger)])
        #
        #     challenger_reward = trajectory.get_final_reward(player_id_challenger)
        #     challenger_results.append(challenger_reward)
        #     match_outcomes.append(
        #         MatchOutcome.win(self.current_generation + 1, self.current_generation)
        #         if challenger_reward == 1 else
        #         MatchOutcome.defeat(self.current_generation + 1, self.current_generation))
        #
        # n_challenger_wins = (np.array(challenger_results) == 1).sum()
        # challenger_win_rate = n_challenger_wins / n_evaluations
        # return challenger_win_rate, challenger_policies, match_outcomes

    def replace_model_with_challenger(self, challenger_win_rate: float, win_ratio_needed: float, iteration: int):
        if challenger_win_rate > win_ratio_needed:
            self.model_manager.store_model(self.model_challenger, iteration)
            self.model_current_best = self.model_manager.load_model(iteration)
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
