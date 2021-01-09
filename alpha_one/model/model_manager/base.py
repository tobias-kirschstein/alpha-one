from abc import ABC, abstractmethod

from alpha_one.utils.io import save_pickled, load_pickled


class ModelManager(ABC):

    def __init__(self, model_store_path: str):
        self.model_store_path = model_store_path

    @abstractmethod
    def store_model(self, model, iteration):
        pass

    @abstractmethod
    def load_model(self, iteration):
        pass

    @abstractmethod
    def list_models(self):
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

    def load_config(self):
        return load_pickled(f"{self.model_store_path}/config")
