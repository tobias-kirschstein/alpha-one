from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pyspiel

from alpha_one.game.information_set import InformationSetGenerator


class Agent(ABC):

    def __init__(self, is_information_set_agent: bool = False):
        self._is_information_set_agent = is_information_set_agent

    @abstractmethod
    def next_move(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> (int, np.array):
        pass

    @abstractmethod
    def evaluate(self, state_or_information_set_generator: Union[pyspiel.State, InformationSetGenerator]) -> float:
        pass

    def is_information_set_agent(self) -> bool:
        return self._is_information_set_agent
