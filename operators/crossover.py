import copy
import dataclasses
import logging
import random
from collections.abc import MutableMapping, MutableSequence
from dataclasses import Field
from functools import singledispatchmethod
from typing import Any, Dict, List, Tuple, Union

from model import Candidate, Population, HyperParameters
from utils import Dataclass, T, field_dict

logger = logging.getLogger(__file__)


class CrossoverOperator:
    def __init__(self, swap_p: float=0.5):
        self.swap_p = swap_p

    def __call__(self, obj1: T, obj2: T=None, inplace: bool=False) -> Tuple[T, T]:
        obj1 = obj1 if inplace else copy.deepcopy(obj1)
        obj2 = obj2 if inplace else copy.deepcopy(obj2)
        if obj2 is not None:
            assert type(obj1) == type(obj2), "Can only crossover between objects of the same type (for now)"

        self.crossover(obj1, obj2)
        if obj2 is None:
            return obj1
        return obj1, obj2

    @singledispatchmethod
    def crossover(self, obj1, obj2) -> None:
        logger.error(f"Unable to apply crossover to objects {obj1} and {obj2} of type {type(obj1)}.")

    @crossover.register
    def _(obj1: object, obj2: object) -> None:
        """ Most General case: Randomly swap the attributes on two objects """
        crossover(vars(obj1), vars(obj2))

    @crossover.register
    def _(self, obj1: HyperParameters, obj2: HyperParameters) -> None:
        """ Performs crossover between two dataclass instances in-place. """
        obj1_fields: Dict[str, dataclasses.Field] = field_dict(obj1)
        obj2_fields: Dict[str, dataclasses.Field] = field_dict(obj2)

        for field in dataclasses.fields(obj1):
            v1 = getattr(obj1, field.name)
            v2 = getattr(obj2, field.name, v1)

            if random.random() <= self.swap_p:
                setattr(obj1, field.name, v2)
                setattr(obj2, field.name, v1)
    
    @crossover.register
    def _(self, pop1: Population, pop2: Population=None) -> None:
        """ Performs crossover either within one or between two `Population` instances in-place. """
        if not pop2:
            pop2 = pop1[1::2]
            pop1 = pop1[0::2]
        for c1, c2 in zip(pop1, pop2):
            self.crossover(c1, c2)
    
    @crossover.register
    def _(self, candidate1: Candidate, candidate2: Candidate) -> None:
        """ Performs crossover between two `Candidate` instances in-place. """
        self.crossover(candidate1.hparams, candidate2.hparams)
    
    @crossover.register
    def _(self, obj1: dict, obj2: dict) -> None:
        """ Performs crossover between two `dict` instances in-place. """
        for key, v1 in obj1.items():
            if key in obj2:
                # TODO: also crossover the nested dicts?
                if isinstance(v1, dict):
                    self.crossover(obj1[key], obj2[key])
                else:  
                    v2 = obj2.get(key)
                    v1_n, v2_n = self.crossover(v1, v2)
                      
                    if random.random() <= self.swap_p:
                        obj1[key] = v1_n
                        obj2[key] = v2_n
