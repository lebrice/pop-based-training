import copy
import multiprocessing
import os
import pprint
import random
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Generator, TypeVar, List

from .hyperparameters import HyperParameters, hparam
from .candidate import Candidate
from .operators import (CrossoverOperator, MutationOperator,
                       TournamentSelectionOperator)

C = TypeVar("C", bound=Candidate)


def epbt(n_generations: int,
         initial_population: List[Candidate],
         evaluation_function: Callable[[C], C],
         n_processes: int=None,
         multiprocessing=multiprocessing) -> Generator[Candidate, None, None]:
    """Performs the epbt algorithm, yielding the best candidate at each step.

    An `evaluation_function` should, given the (optional) previous `Candidate`:
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

    For an example, check out the [dummy](dummy_example.py) and
    [mnist](mnist_pytorch_example.py) examples.

    Args:
        n_generations (int): The number of steps or generations to perform.
        initial_population (Population): The initial population of candidates.
        evaluation_function (Callable[[Candidate], Candidate]): The evaluation
        function to use, as described above.
        n_processes (int, optional): The number of processes to use. Defaults to
        None.
        multiprocessing (module, optional): The multiprocessing module to use
        (for instance, could be set to `torch.multiprocessing`). Defaults to 
        multiprocessing.
    
    Returns:
        Generator[List[Candidate], None, None]: A generator that yields the best
        candidate at each step.
    
    Yields:
        Generator[List[Candidate], None, None]: [description]
    """
    population: List[Candidate] = initial_population
    pop_size = len(population)
    elite_count = pop_size // 2
    
    # Create the operators:
    crossover = CrossoverOperator()
    mutate = MutationOperator()
    tournament = TournamentSelectionOperator(
        n_winners=len(population)-elite_count,
        fighters_per_round=2
    )

    # Keep track of the best candidate at each step.
    best_candidate: Any = max(initial_population)
    
    with multiprocessing.Pool(n_processes) as pool:
        for i in range(n_generations):
            # Sort the population by decreasing fitness (best first).
            population.sort(reverse=True)
            # Elitism: Save the best K candidates so they don't get mutated.
            elites = population[:elite_count]

            # Select the fittest descendants with tournament selection.
            descendants = tournament(population)
            # Apply the Mutation and Crossover operators.
            descendants = mutate(descendants)
            descendants = crossover(descendants)

            candidates = elites + descendants
            
            # Evaluate the population.
            population.clear()
            start_time = time.time()
            for candidate in pool.imap_unordered(evaluation_function, candidates):
                population.append(candidate)
                t: float = time.time() - start_time
                print(f"Evaluated candidate {candidate} in {t:.2f}s.")
                new_best_candidate = max(best_candidate, candidate)
                if new_best_candidate is not best_candidate:
                    best_candidate = new_best_candidate
                    print(f"New best candidate found: {best_candidate}")
            yield best_candidate
