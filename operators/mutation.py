import copy
import dataclasses
import logging
import random

from utils import T

from model.candidate import Candidate
from model.hyperparameters import HyperParameters
from model.population import Population

logger = logging.getLogger(__file__)


class MutationOperator:
    def __init__(self, mu: float=0., sigma: float=1.0, reset_prob: float=0.05):
        self.mu = mu
        self.sigma = sigma
        self.reset_prob = reset_prob

    def __call__(self, obj: T, inplace: bool=False) -> T:
        """ Mutates either a `HyperParameters`, `Candidate` or `Population`. """ 
        obj = obj if inplace else copy.deepcopy(obj)
        if isinstance(obj, HyperParameters):
            self.mutate_hparam(obj)
        elif isinstance(obj, Candidate):
            self.mutate_candidate(obj)
        elif isinstance(obj, Population):
            self.mutate_population(obj)
        else:
            # TODO: Add a case for nn.Module instance?
            try:
                self.mutate_hparam(obj)
            except Exception as e:
                logger.error(f"Couldn't mutate the object {obj}")
        return obj

    def mutate_population(self, population: Population) -> None:
        """ Mutates a Population instance in-place. """
        for candidate in population:
            self.mutate_candidate(candidate)

    def mutate_candidate(self, candidate: Candidate) -> None:
        """ Mutates a Candidate instance in-place. """
        self.mutate_hparam(candidate.hparams)

    def mutate_hparam(self, obj: HyperParameters) -> None:
        """ Mutates a HyperParameters instance in-place. """
        obj = obj
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name, field.default)
            noise = random.gauss(self.mu, self.sigma)

            new_value = value
            if isinstance(value, HyperParameters):
                new_value = self.mutate_candidate(value)
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
                    new_value = field.default_factory()
            
            # Set the attribute back on obj.
            setattr(obj, field.name, new_value)
