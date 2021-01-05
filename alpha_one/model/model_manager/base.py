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

    def store_config(self, config: dict):
        save_pickled(config, f"{self.model_store_path}/config")

    def load_config(self):
        return load_pickled(f"{self.model_store_path}/config")
