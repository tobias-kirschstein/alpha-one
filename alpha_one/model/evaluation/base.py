import copy
from statistics import mean

from alpha_one.metrics import MatchOutcome
from alpha_one.model.model_manager import CheckpointManager
from alpha_one.utils.mcts import initialize_bot, play_one_game, MCTSConfig
from alpha_one.utils.determinized_mcts import play_one_game_d
from alpha_one.utils.mcts_II import initialize_bot_alphaone, play_one_game_alphaone, IIGMCTSConfig
import numpy as np
import ray


def _compare_models(game, model_1, model_2, mcts_config: MCTSConfig, player_id_model_1):
    if mcts_config.determinized_MCTS:
        models = [model_1, model_2] if player_id_model_1 == 0 else [model_2, model_1]
        return play_one_game_d(game, models, mcts_config)

    elif isinstance(mcts_config, IIGMCTSConfig) and mcts_config.alpha_one:
        mcts_bot_model_1 = initialize_bot_alphaone(game, model_1, mcts_config)

        mcts_bot_model_2 = initialize_bot_alphaone(game, model_2, mcts_config)

        bots = [mcts_bot_model_1, mcts_bot_model_2] \
        if player_id_model_1 == 0 \
        else [mcts_bot_model_2, mcts_bot_model_1]

        trajectory_observation, trajectory_game = play_one_game_alphaone(game, bots, mcts_config)

        return trajectory_game
    else:



        mcts_bot_model_1 = initialize_bot(game, model_1, mcts_config.uct_c,
                                          mcts_config.max_mcts_simulations,
                                          mcts_config.policy_epsilon, mcts_config.policy_alpha,
                                          omniscient_observer=mcts_config.omniscient_observer)

        mcts_bot_model_2 = initialize_bot(game, model_2, mcts_config.uct_c,
                                          mcts_config.max_mcts_simulations,
                                          mcts_config.policy_epsilon, mcts_config.policy_alpha,
                                          omniscient_observer=mcts_config.omniscient_observer)

        bots = [mcts_bot_model_1, mcts_bot_model_2] \
            if player_id_model_1 == 0 \
            else [mcts_bot_model_2, mcts_bot_model_1]

        return play_one_game(game, bots, mcts_config.temperature, mcts_config.temperature_drop,
                             omniscient_observer=mcts_config.omniscient_observer)


@ray.remote(num_returns=1)
def _compare_models_parallel(game, checkpoint_manager: CheckpointManager, mcts_config: MCTSConfig, player_id_model_1,
                             model_checkpoint_1, model_checkpoint_2):
    model_1 = checkpoint_manager.load_checkpoint(model_checkpoint_1)
    model_2 = checkpoint_manager.load_checkpoint(model_checkpoint_2)
    return _compare_models(game, model_1, model_2, mcts_config, player_id_model_1)


class EvaluationManager:

    def __init__(self, game, n_evaluations, mcts_config: MCTSConfig):
        self.game = game
        self.n_evaluations = n_evaluations
        self.mcts_config = mcts_config

    def compare_models(self, model_1, model_2):
        model_1_results = []
        match_outcomes = []
        trajectories = []
        for i in range(self.n_evaluations):
            player_id_model_1 = np.random.choice([0, 1])  # ensure that each model will play as each player

            trajectory = _compare_models(self.game, model_1, model_2, self.mcts_config, player_id_model_1)

            trajectories.append((trajectory, player_id_model_1))

            model_1_reward = trajectory.get_final_reward(player_id_model_1)
            model_1_results.append(model_1_reward)
            match_outcomes.append(
                MatchOutcome.win(1, 2)
                if model_1_reward > 0 else
                MatchOutcome.defeat(1, 2))

        n_model_1_wins = (np.array(model_1_results) > 0).sum()
        model_1_win_rate = n_model_1_wins / self.n_evaluations
        model_1_average_reward = mean(model_1_results)
        return model_1_win_rate, trajectories, match_outcomes, model_1_average_reward


class ParallelEvaluationManager:

    def __init__(self, game, checkpoint_manager: CheckpointManager, n_evaluations, mcts_config: MCTSConfig):
        self.game = game
        self.checkpoint_manager = checkpoint_manager
        self.n_evaluations = n_evaluations
        self.mcts_config = mcts_config

    def compare_models(self, model_checkpoint_1, model_checkpoint_2):
        model_1_results = []
        match_outcomes = []
        trajectories = []
        for i in range(self.n_evaluations):
            player_id_model_1 = np.random.choice([0, 1])  # ensure that each model will play as each player

            trajectory_ = _compare_models_parallel.remote(game=self.game, checkpoint_manager=self.checkpoint_manager,
                                                          mcts_config=self.mcts_config,
                                                          player_id_model_1=player_id_model_1,
                                                          model_checkpoint_1=model_checkpoint_1,
                                                          model_checkpoint_2=model_checkpoint_2)
            trajectory = copy.deepcopy(trajectory_)
            del trajectory_

            trajectories.append((trajectory, player_id_model_1))

        trajectories = [(ray.get(trajectory), player_id_model_1) for trajectory, player_id_model_1 in trajectories]
        for trajectory, player_id_model_1 in trajectories:
            model_1_reward = trajectory.get_final_reward(player_id_model_1)
            model_1_results.append(model_1_reward)
            match_outcomes.append(
                MatchOutcome.win(1, 2)
                if model_1_reward > 0 else
                MatchOutcome.defeat(1, 2))

        n_model_1_wins = (np.array(model_1_results) > 0).sum()
        model_1_win_rate = n_model_1_wins / self.n_evaluations
        model_1_average_reward = mean(model_1_results)
        return model_1_win_rate, trajectories, match_outcomes, model_1_average_reward
