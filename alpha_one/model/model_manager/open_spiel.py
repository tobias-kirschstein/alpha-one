from .base import ModelManager
from pathlib import Path
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from env import MODEL_SAVES_DIR
from alpha_one.model.config import OpenSpielModelConfig


class OpenSpielModelManager(ModelManager):

    def __init__(self, model_dir, config: OpenSpielModelConfig):
        self.model_saves_path = f"{MODEL_SAVES_DIR}/{model_dir}"
        self.config = config

    def store(self, model, iteration):
        Path(self.model_saves_path).mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(iteration)

    def load(self, iteration):
        new_model = self.build_model()
        new_model.load_checkpoint(f"{self.model_saves_path}/checkpoint-{iteration}")
        return new_model

    def build_model(self):
        return model_lib.Model.build_model(
            self.config.model_type,
            self.config.game.observation_tensor_shape(),
            self.config.game.num_distinct_actions(),
            nn_width=self.config.nn_width,
            nn_depth=self.config.nn_depth,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            path=self.model_saves_path)
