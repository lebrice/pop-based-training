import copy
import dataclasses
import math
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields
from typing import ClassVar, Dict, Type, TypeVar, Union, cast, Sequence

from utils import T

H = TypeVar("H", bound="HyperParameters")


def hparam(default: T,
          *args,
          min: T=None,
          max: T=None,
        #   choices: Sequence[T]=None,  # TODO
          **kwargs) -> T:
    """ Overload of `dataclasses.field()` used to specify the range of an arg.
    
    Specifying a valid range prevents the mutation operator from changing the
    value to one that doesn't make sense. For instance, negative number of
    layers, etc.
    """
    metadata = kwargs.get("metadata", {})
    metadata.update({
        "min": min,
        "max": max,
        # "choices": choices,  # TODO
    })
    kwargs["metadata"] = metadata

    return dataclasses.field(
        default=default,
        *args, **kwargs, 
    )


@dataclass
class HyperParameters:
    """ Base class that marks the attributes as hyperparameters to be optimized.
    
    Basically, you should implement your own hyperparameter class, and have it
    subclass `HyperParameters`. In doing so, the attributes of that class are
    considered hyperparameters, and might have their values changed during
    training.
    
    To limit the set of possible values or set a desired range, use the
    `hyperparameters.hparam` function, along with the `min`, `max` or `choices`
    (TODO) arguments.

    NOTE: All the prior-related stuff is not related to EPBT. Might take it out.
    It was just for the sake of experimentation.
    """

    def asdict(self) -> Dict:
        return dataclasses.asdict(self)
