from abc import ABC, abstractmethod
from typing import Type

from alpha_one.utils.io import save_pickled, load_pickled, list_file_numbering
from alpha_one.utils.logging import generate_run_name
from env import MODEL_SAVES_DIR


class CheckpointManager(ABC):

    def __init__(self, model_store_path: str, run_name: str):
        self.model_store_path = f"{MODEL_SAVES_DIR}/{model_store_path}/{run_name}"
        self.run_name = run_name

    @abstractmethod
    def store_checkpoint(self, model, iteration):
        pass

    @abstractmethod
    def _load_checkpoint(self, iteration, **kwargs):
        pass

    def load_checkpoint(self, checkpoint, **kwargs):
        if checkpoint in {'latest', 'last'}:
            checkpoint = -1
        if checkpoint < 0:
            checkpoints = self.list_checkpoints()
            checkpoint = checkpoints[checkpoint]
        return self._load_checkpoint(checkpoint, **kwargs)

    @abstractmethod
    def list_checkpoints(self):
        pass

    @abstractmethod
    def _build_model(self, config: dict):
        pass

    def build_model(self, config: dict = None):
        if config is None:
            # Assume that config has already been stored
            return self._build_model(self.load_config())
        else:
            return self._build_model(config)

    def store_config(self, config: dict):
        save_pickled(config, f"{self.model_store_path}/config")

    def load_config(self, **kwargs):
        return load_pickled(f"{self.model_store_path}/config")

    def get_run_name(self):
        return self.run_name

    def get_model_store_path(self):
        return self.model_store_path


class ModelManager:

    def __init__(self, model_dir: str, prefix, cls_checkpoint_manager: Type[CheckpointManager]):
        self.model_store_path = f"{MODEL_SAVES_DIR}/{model_dir}"
        self.model_dir = model_dir
        self.prefix = prefix
        self.cls_checkpoint_manager = cls_checkpoint_manager

    def list_runs(self):
        run_ids = list_file_numbering(self.model_store_path, self.prefix)
        return [f"{self.prefix}-{run_id}" for run_id in run_ids]

    def generate_run_name(self):
        return generate_run_name(self.model_store_path, self.prefix, match_arbitrary_suffixes=True)

    def new_run(self):
        return self.cls_checkpoint_manager(self.model_dir, self.generate_run_name())

    def get_checkpoint_manager(self, run_name):
        return self.cls_checkpoint_manager(self.model_dir, run_name)
