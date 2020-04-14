import copy
import dataclasses
import logging
import random
from functools import singledispatchmethod
from typing import Any

from hyperparameters import HyperParameters
from candidate import Candidate
from utils import T

logger = logging.getLogger(__file__)


class MutationOperator:
    def __init__(self, mu: float=1., sigma: float=1.0, reset_prob: float=0.05):
        self.mu = mu
        self.sigma = sigma
        self.reset_prob = reset_prob

    def __call__(self, obj: T, inplace: bool=False) -> T:
        """ Mutates either a `HyperParameters`, `Candidate` or `Population`. """ 
        obj = obj if inplace else copy.deepcopy(obj)
        self.mutate(obj)  # type: ignore
        return obj

    @singledispatchmethod
    def mutate(self, obj):
        """ Most general case: mutate a dataclass. """
        print(f"Cannot mutate {obj} of type {type(obj)}")
        return

    @mutate.register
    def mutate_hparam(self, obj: HyperParameters) -> None:
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name, field.default)
            noise = random.gauss(self.mu, self.sigma)

            new_value: Any = value
            if isinstance(new_value, HyperParameters):
                self.mutate(new_value)  # type: ignore
            elif isinstance(new_value, (int, float)):
                new_value = new_value * noise

            # Make sure we don't exceed the bounds on that field, if there are any.
            min_value = field.metadata.get("min")
            max_value = field.metadata.get("max")

            if min_value is not None:
                new_value = min(new_value, min_value)
            if max_value is not None:
                new_value = max(new_value, max_value)
            
            if field.type is int:
                new_value = round(new_value)
            
            # Randomly reset to the default value with probability reset_prob.
            if random.random() < self.reset_prob:
                if field.default is not dataclasses.MISSING:
                    new_value = field.default
                else:
                    new_value = field.default_factory()  # type: ignore
            
            # Set the attribute back on obj.
            setattr(obj, field.name, new_value)


    @mutate.register
    def mutate_pop(self, population: list) -> None:
        """ Mutates a Population instance in-place. """
        for candidate in population:
            self.mutate(candidate)  # type: ignore

    @mutate.register
    def mutate_candidate(self, candidate: Candidate) -> None:
        """ Mutates a Candidate instance in-place. """
        self.mutate(candidate.hparams)  # type: ignore
