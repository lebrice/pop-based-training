import copy
import random
from dataclasses import dataclass
from typing import *

from hyperparameters import HyperParameters

from .candidate import Candidate


class Population(List[Candidate]):
    def select_fittest(self, n: int, t: int=2) -> "Population":
        """Performs tournament selection, as described in the EPBT paper. """
        winners = Population()
        while len(winners) < n:
            fighters: List[Candidate] = random.sample(self, t)
            fighters.sort()
            best_fighter = fighters.pop()
            winners.append(best_fighter)
        return winners

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Population(super().__getitem__(index))
        return super().__getitem__(index)
    
    def __add__(self, other: Union[List, "Population"]) -> "Population":
        return Population(super().__add__(other))
