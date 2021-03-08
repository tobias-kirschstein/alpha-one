from pathlib import Path

from open_spiel.python.algorithms.alpha_zero import model as model_lib

from .base import CheckpointManager, ModelManager
from ..config import OpenSpielModelConfig
from ...data.replay import ReplayDataManager
from ...utils.io import list_file_numbering


class OpenSpielModelManager(ModelManager):

    def __init__(self, game_name, prefix):
        super(OpenSpielModelManager, self).__init__(game_name, prefix, OpenSpielCheckpointManager)


class OpenSpielCheckpointManager(CheckpointManager):

    def __init__(self, game_name, run_name):
        super(OpenSpielCheckpointManager, self).__init__(game_name, run_name)

    def get_replay_data_manager(self):
        return ReplayDataManager(self.model_store_path)

    def store_checkpoint(self, model, iteration):
        Path(self.model_store_path).mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(iteration)

    def _load_checkpoint(self, iteration, **kwargs):
        new_model = self.build_model(self.load_config())
        new_model.load_checkpoint(f"{self.model_store_path}/checkpoint-{iteration}")
        return new_model

    def list_checkpoints(self):
        return list_file_numbering(self.model_store_path, "checkpoint", ".index")

    def _build_model(self, config: OpenSpielModelConfig):
        if config.output_shape is not None:
            return model_lib.Model.build_model(
                config.model_type,
                config.input_shape,
                config.output_shape,
                nn_width=config.nn_width,
                nn_depth=config.nn_depth,
                weight_decay=config.weight_decay,
                learning_rate=config.learning_rate,
                path=self.model_store_path)
        return model_lib.Model.build_model(
            config.model_type,
            config.input_shape,
            config.game.num_distinct_actions(),
            nn_width=config.nn_width,
            nn_depth=config.nn_depth,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            path=self.model_store_path)
