import copy
import dataclasses
import math
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields
from typing import ClassVar, Dict, Type, TypeVar, Union, cast

from priors import LogUniformPrior, Prior, UniformPrior
from utils import T

H = TypeVar("H", bound="HyperParameters")


def hparam(default: T,
          *args,
          min: T=None,
          max: T=None,
          prior: Prior=None,
          **kwargs) -> T:
    metadata = kwargs.get("metadata", {})
    metadata.update({
        "min": min,
        "max": max,
        "prior": prior,
        "is_hparam": True,
    })
    kwargs["metadata"] = metadata

    return dataclasses.field(
        default=default,
        *args, **kwargs, 
    )


@dataclass
class HyperParameters:
    sample_from_priors: ClassVar[bool] = False

    def __post_init__(self):
        # TODO: maybe filter through the fields and alert when one isn't compatible or something?
        if not self.sample_from_priors:
            return
        for field in dataclasses.fields(self):
            prior = field.metadata.get("prior")
            if prior is not None:
                value = prior.sample()
                setattr(self, field.name, value)

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
    bob1 = Bob.sample()
    print(bob1)
