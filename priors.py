import math
import random
from abc import abstractmethod
from dataclasses import dataclass, InitVar
from typing import *
from typing import TypeVar

T = TypeVar("T")


@dataclass  # type: ignore
class Prior(Generic[T]):
    @abstractmethod
    def sample(self) -> T:
        pass


@dataclass
class GaussianPrior(Prior[float]):
    mu: float = 0.
    sigma: float = 1.

    def sample(self) -> float:
        value = random.gauss(self.mu, self.sigma)
        return value

@dataclass
class UniformPrior(Prior):
    min: float = 0.
    max: float = 1.

    def sample(self) -> Union[float, int]:
        # TODO: add suport for enums?
        value = random.uniform(self.min, self.max)
        if isinstance(self.min, int) and isinstance(self.max, int):
            return round(value)
        return value


@dataclass
class LogUniformPrior(UniformPrior):
    base: float = 10.0
    def sample(self) -> float:
        log_min = math.log(self.min, self.base)
        log_max = math.log(self.max, self.base)
        log_val = random.uniform(log_min, log_max)
        value = math.pow(self.base, log_val)
        return value
