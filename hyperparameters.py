import copy
import dataclasses
import math
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields
from typing import ClassVar, Dict, Type, TypeVar, Union, cast, Sequence

from priors import LogUniformPrior, Prior, UniformPrior
from utils import T

H = TypeVar("H", bound="HyperParameters")


def hparam(default: T,
          *args,
          min: T=None,
          max: T=None,
        #   choices: Sequence[T]=None,  # TODO
          prior: Prior=None,
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
        "prior": prior,
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
    def __post_init__(self):
        # TODO: maybe filter through the fields and alert when one isn't compatible or something?
        if not self.sample_from_priors:
            return
        for field in dataclasses.fields(self):
            prior = field.metadata.get("prior")
            if prior is not None:
                value = prior.sample()
                setattr(self, field.name, value)

    sample_from_priors: ClassVar[bool] = False

    @property
    def asdict(self) -> Dict:
        return dataclasses.asdict(self)
    
    @classmethod
    def sample(cls: Type[H]) -> H:
        with cls.use_priors():
            instance = cls()
        return instance

    @classmethod
    @contextmanager
    def use_priors(cls, value: bool=True):
        temp = cls.sample_from_priors
        cls.sample_from_priors = value
        yield
        cls.sample_from_priors = temp


if __name__ == "__main__":
    @dataclass
    class Bob(HyperParameters):
        learning_rate: float = hparam(default=1e-3, min=1e-10, max=1, prior=LogUniformPrior(1e-10, 1))
        n_layers: int = hparam(10, min=1, max=20, prior=UniformPrior(1,20))
        optimizer: str = "ADAM"
        momentum: float = 0.9

    bob = Bob(learning_rate=0.1, n_layers=2)
    print(bob)
    # bob1 = Bob.sample()
    print(bob1)
