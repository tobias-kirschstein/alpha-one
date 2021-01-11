from glob import glob
from pathlib import Path

from open_spiel.python.algorithms.alpha_zero import model as model_lib

from env import MODEL_SAVES_DIR
from .base import ModelManager
from ..config import OpenSpielModelConfig
import re

from ...utils.io import list_file_numbering


class OpenSpielModelManager(ModelManager):

    def __init__(self, model_dir):
        super(OpenSpielModelManager, self).__init__(f"{MODEL_SAVES_DIR}/{model_dir}")

    def store_model(self, model, iteration):
        Path(self.model_store_path).mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(iteration)

    def load_model(self, iteration):
        config = self.load_config()
        new_model = self.build_model(config)
        if iteration == -1 or iteration == 'latest' or iteration == 'last':
            stored_models = self.list_models()
            iteration = stored_models[-1]
        new_model.load_checkpoint(f"{self.model_store_path}/checkpoint-{iteration}")
        return new_model

    def list_models(self):
        return list_file_numbering(self.model_store_path, "checkpoint", ".index")

    def _build_model(self, config: OpenSpielModelConfig):
        return model_lib.Model.build_model(
            config.model_type,
            config.game.observation_tensor_shape(),
            config.game.num_distinct_actions(),
            nn_width=config.nn_width,
            nn_depth=config.nn_depth,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            path=self.model_store_path)