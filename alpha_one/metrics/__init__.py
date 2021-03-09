from .rating import *
from .elo import *
from .true_skill import *
from .average_reward import AverageRewardRatingSystem
from .entropy import cross_entropy

from statistics import mean
from scipy.stats import entropy


def calculate_entropy(policies):
    return mean([entropy(policy) for policy in policies])
