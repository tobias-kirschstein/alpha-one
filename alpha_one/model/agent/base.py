from abc import ABC, abstractmethod

import numpy as np
import pyspiel


class Agent(ABC):
    @abstractmethod
    def next_move(self, state: pyspiel.State) -> (int, np.array):
        pass

    @abstractmethod
    def evaluate(self, state: pyspiel.State) -> float:
        pass
