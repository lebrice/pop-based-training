import copy
import dataclasses
import logging
import random
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, NamedTuple, Union

from hyperparameters import HyperParameters
from candidate import Candidate
from utils import T

logger = logging.getLogger(__file__)

@dataclass
class MutationOptions:
    mu: float=1.
    sigma: float=1.0
    reset_prob: float=0.05

    @property
    def gauss_noise(self) -> float:
        return random.gauss(self.mu, self.sigma)
    @property
    def reset(self) -> bool:
        return random.random() <= self.reset_prob

class MutationOperator:
    def __init__(self, mu: float=1., sigma: float=1.0, reset_prob: float=0.05):
        self.mu = mu
        self.sigma = sigma
        self.reset_prob = reset_prob
        self.options = MutationOptions(
            mu=self.mu,
            sigma=self.sigma,
            reset_prob=self.reset_prob,
        )

    def __call__(self, obj: T, inplace: bool=False) -> T:
        """ Mutates either a `HyperParameters`, `Candidate` or `Population`. """ 
        obj = obj if inplace else copy.deepcopy(obj)
        obj_ = mutate(obj, self.options)
        if obj_ is not None:
            return obj_
        return obj


@singledispatch
def mutate(obj, options: MutationOptions):
    logger.debug(f"Cannot mutate {obj} of type {type(obj)}")
    return obj


@mutate.register
def mutate_float(value: float, options: MutationOptions) -> float:
    return value * options.gauss_noise


@mutate.register
def mutate_int(value: int, options: MutationOptions) -> float:
    return round(value * options.gauss_noise)


@mutate.register
def mutate_hparams(obj: HyperParameters, options: MutationOptions) -> None:
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name, field.default)
        # TODO: Might be cool to pass the mutation options directly to the field
        options = field.metadata.get("mutation_options") or options

        new_value: Any = mutate(value, options)

        if field.type in {int, float}:
            # Make sure we don't exceed the bounds on that field, if any
            min_value = field.metadata.get("min")
            if min_value is not None:
                new_value = min(new_value, min_value)

            max_value = field.metadata.get("max")
            if max_value is not None:
                new_value = max(new_value, max_value)
            if field.type is int:
                new_value = round(new_value)
        
        # Randomly reset to the default value with probability reset_prob.
        if options.reset:
            if field.default is not dataclasses.MISSING:
                new_value = field.default
            elif field.default_factory is not dataclasses.MISSING:  # type: ignore
                new_value = field.default_factory()  # type: ignore
            else:
                # can't reset to the default value. Keep the original value.
                new_value = value
        # Set the attribute back on obj.
        setattr(obj, field.name, new_value)


@mutate.register
def mutate_population(population: list, options: MutationOptions) -> None:
    """ Mutates a Population instance in-place. """
    for candidate in population:
        mutate(candidate, options)

@mutate.register
def mutate_candidate(candidate: Candidate, options: MutationOptions) -> None:
    """ Mutates a Candidate instance in-place. """
    mutate(candidate.hparams, options)
