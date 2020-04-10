import copy
import dataclasses
import logging
import random
from functools import singledispatchmethod

from hyperparameters import HyperParameters
from utils import T

from model import Candidate, Population

logger = logging.getLogger(__file__)


class MutationOperator:
    def __init__(self, mu: float=1., sigma: float=1.0, reset_prob: float=0.05):
        self.mu = mu
        self.sigma = sigma
        self.reset_prob = reset_prob

    def __call__(self, obj: T, inplace: bool=False) -> T:
        """ Mutates either a `HyperParameters`, `Candidate` or `Population`. """ 
        obj = obj if inplace else copy.deepcopy(obj)
        self.mutate(obj)
        return obj

    @singledispatchmethod
    def mutate(self, obj) -> None:
        pass

    @mutate.register
    def mutate_pop(self, population: Population) -> None:
        """ Mutates a Population instance in-place. """
        for candidate in population:
            self.mutate(candidate)

    @mutate.register
    def mutate_candidate(self, candidate: Candidate) -> None:
        """ Mutates a Candidate instance in-place. """
        self.mutate(candidate.hparams)

    @mutate.register
    def mutate_hparam(self, obj: HyperParameters) -> None:
        """ Mutates a HyperParameters instance in-place. """
        obj = obj
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name, field.default)
            noise = random.gauss(self.mu, self.sigma)

            new_value = value
            if isinstance(value, HyperParameters):
                new_value = self.mutate(value)
            elif isinstance(value, (int, float)):
                new_value = value * noise
            
            # Make sure we don't exceed the bounds on that field, if there are any.
            min_value = field.metadata.get("min_value")
            max_value = field.metadata.get("max_value")

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
