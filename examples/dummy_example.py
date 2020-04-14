import random
import time
from dataclasses import dataclass
from typing import List

from epbt.candidate import Candidate
from epbt.hyperparameters import HyperParameters, hparam
from epbt.helpers import Config
from epbt.main import epbt



def dummy_evaluation(candidate: Candidate) -> Candidate:
    """Dummy evaluation which just waits for 1 second and sets a random fitness. 
    
    An evaluation function should, given the previous `Candidate`:
    1. Create a new model, using the (potentially mutated) hyperparameters
       stored at `candidate.hparams`.
    2. Load weights (state_dict) from the previous model (stored at
       `candidate.model`) into the new model. 
        - It might be possible that the structure of the model changed, hence
          this should try to reuse as much state as possible.
    3. Train the (new/updated) model on some task/dataset;
    4. Evaluate the model on some validation set to obtain a fitness score.
       (higher is better)
    5. Return a new `Candidate` instance with the new/updated model, the
       corresponding HyperParameters, and the resulting fitness value.
    """
    time.sleep(1)
    candidate.fitness = random.random()
    return candidate


@dataclass
class Bob(HyperParameters):
    learning_rate: float = hparam(1e-3, min=1e-10, max=1)
    n_layers: int = hparam(10, min=1, max=20)
    optimizer: str = "ADAM"
    momentum: float = 0.9


if __name__ == "__main__":
    random.seed(0)
    pop: List[Candidate] = [
        Candidate(model=None, hparams=Bob(learning_rate=1e-2), fitness=0.1),
        Candidate(model=None, hparams=Bob(learning_rate=1e-5), fitness=0.2),
        Candidate(model=None, hparams=Bob(learning_rate=1e-7), fitness=0.5),
        Candidate(model=None, hparams=Bob(learning_rate=1e-8), fitness=0.3),
        Candidate(model=None, hparams=Bob(learning_rate=1e-2), fitness=0.4),
    ]
    generator = epbt(n_generations=5, initial_population=pop, evaluation_function=dummy_evaluation)
    
    for i, best_candidate in enumerate(generator):
        print(f"Best hparams at step {i}: {best_candidate}.")
