import copy
import dataclasses
import logging
import random
from collections.abc import MutableMapping, MutableSequence
from dataclasses import Field
from functools import singledispatch
from typing import Any, Dict, List, Tuple, TypeVar, Union, overload

from hyperparameters import HyperParameters
from candidate import Candidate
from utils import Dataclass, T, field_dict

logger = logging.getLogger(__file__)
P = TypeVar("P", bound=List[Candidate])

class CrossoverOperator:
    def __init__(self, swap_p: float=0.5):
        self.swap_p = swap_p

    @overload
    def __call__(self, obj1: P, inplace:bool=False) -> P:
        pass
    
    @overload
    def __call__(self, obj1: T, obj2: T, inplace:bool=False) -> Tuple[T, T]:
        pass

    def __call__(self, obj1, obj2=None, inplace=False):
        obj1 = obj1 if inplace else copy.deepcopy(obj1)
        obj2 = obj2 if inplace else copy.deepcopy(obj2)
        if obj2 is not None:
            assert type(obj1) == type(obj2), "Can only crossover between objects of the same type (for now)"

        crossover(obj1, obj2)
        if obj2 is None:
            return obj1
        return obj1, obj2


@singledispatch
def crossover(obj1: object, obj2: object, swap_p: float=0.5) -> Tuple[object, object]:
    """ Most General case: Randomly swap the attributes on two objects """
    raise RuntimeError(f"Cannot perform crossover between objects {obj1} and {obj2}.")


@crossover.register
def crossover_hparam(obj1: HyperParameters, obj2: HyperParameters, swap_p: float=0.5) -> None:
    """ Performs crossover between two dataclass instances in-place. """
    obj1_fields: Dict[str, dataclasses.Field] = field_dict(obj1)
    obj2_fields: Dict[str, dataclasses.Field] = field_dict(obj2)

    for field in dataclasses.fields(obj1):
        v1 = getattr(obj1, field.name)
        v2 = getattr(obj2, field.name, v1)

        if random.random() <= swap_p:
            setattr(obj1, field.name, v2)
            setattr(obj2, field.name, v1)


@crossover.register
def crossover_pop(pop1: list, pop2: list=None, swap_p: float=0.5) -> None:
    """ Performs crossover either within one or between two `Population` instances in-place. """
    if not pop2:
        pop2 = pop1[1::2]
        pop1 = pop1[0::2]
    assert pop2, f"pop2 should not be empty or None: {pop1}, {pop2}"
    for c1, c2 in zip(pop1, pop2):
        crossover(c1, c2, swap_p)


@crossover.register
def crossover_candidate(candidate1: Candidate, candidate2: Candidate, swap_p: float=0.5) -> None:
    """ Performs crossover between two `Candidate` instances in-place. """
    crossover(candidate1.hparams, candidate2.hparams, swap_p)


@crossover.register
def crossover_dicts(obj1: dict, obj2: dict, swap_p: float=0.5) -> None:
    """ Performs crossover between two `dict` instances in-place. """
    for key, v1 in obj1.items():
        if key not in obj2:
            continue
        v2 = obj2[key]
        
        # TODO: also crossover the nested dicts?
        if isinstance(v1, dict):
            crossover(obj1[key], obj2[key])
        else:    
            if random.random() <= swap_p:
                obj2[key] = v1
                obj1[key] = v2
