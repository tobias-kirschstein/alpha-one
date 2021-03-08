from typing import Tuple

from alpha_one.model.model_manager import ModelManager, CheckpointManager, OpenSpielCheckpointManager


class AlphaOneModelManager(ModelManager):

    def __init__(self, game_name, prefix):
        super(AlphaOneModelManager, self).__init__(game_name, prefix, AlphaOneCheckpointManager)


class AlphaOneCheckpointManager(CheckpointManager):

    def __init__(self, game_name, run_name):
        self.observation_model_manager = OpenSpielCheckpointManager(game_name, f"{run_name}-observation_model")
        self.game_model_manager = OpenSpielCheckpointManager(game_name, f"{run_name}-game_model")
        super(AlphaOneCheckpointManager, self).__init__(game_name, run_name)

    def store_checkpoint(self, model: Tuple, iteration):
        self.observation_model_manager.store_checkpoint(model[0], iteration)
        self.game_model_manager.store_checkpoint(model[1], iteration)

    def _load_checkpoint(self, iteration, **kwargs):
        observation_model = self.observation_model_manager.load_checkpoint(iteration, **kwargs)
        game_model = self.game_model_manager.load_checkpoint(iteration, **kwargs)
        return observation_model, game_model

    def list_checkpoints(self):
        return self.observation_model_manager.list_checkpoints()

    def load_config(self, **kwargs):
        observation_model_config = self.observation_model_manager.load_config()
        game_model_config = self.game_model_manager.load_config()
        return observation_model_config, game_model_config

    def load_observation_model_config(self):
        return self.observation_model_manager.load_config()

    def load_game_model_config(self):
        return self.game_model_manager.load_config()

    def load_observation_model_checkpoint(self, iteration):
        return self.observation_model_manager.load_checkpoint(iteration)

    def load_game_model_checkpoint(self, iteration):
        return self.game_model_manager.load_checkpoint(iteration)

    def _build_model(self, config: Tuple[dict, dict]):
        observation_model_config, game_model_config = config
        observation_model = self.observation_model_manager.build_model(observation_model_config)
        game_model = self.game_model_manager.build_model(game_model_config)
        return observation_model, game_model
