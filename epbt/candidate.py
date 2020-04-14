import copy
import random
from dataclasses import dataclass, field
from typing import *

from .hyperparameters import HyperParameters


@dataclass
class Candidate:
    """ Simple class holding the model, the hyperparameters, and the fitness.
    
    TODO: Might want to add some serialization helper methods?
    """
    model: Any = field(repr=False)
    hparams: HyperParameters
    fitness: float

    def __lt__(self, other: "Candidate") -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.fitness < other.fitness
