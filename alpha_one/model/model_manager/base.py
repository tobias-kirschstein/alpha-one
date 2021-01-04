from abc import ABC, abstractmethod

class ModelManager(ABC):

    @abstractmethod
    def store(self, model, iteration):
        pass

    def load(self, iteration):
        pass

